#include <Arduino.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "offloading_model.h"
#include "deployment_vectors.h"
#include "boundary_vectors.h"
#include "near_boundary_vectors.h"

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 2 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

struct AnalyticalState {
    float C;
    float L;
    int T;
    int F;
};

const AnalyticalState ANALYTICAL_STATES[] = {
    {20.0f,  20.0f, 0, 1},
    {20.0f, 200.0f, 1, 0},
    {50.0f,  60.0f, 0, 0},
    {50.0f, 160.0f, 1, 1},
    {75.0f,  30.0f, 1, 1},
    {75.0f, 220.0f, 0, 0},
    {95.0f,  40.0f, 1, 0},
    {95.0f, 230.0f, 0, 1}
};

constexpr int NUM_ANALYTICAL_STATES =
    sizeof(ANALYTICAL_STATES) /
    sizeof(ANALYTICAL_STATES[0]);

volatile int analytical_action_sink = 0;

int argmin3(
    int8_t a,
    int8_t b,
    int8_t c
) {
    int action = 0;
    int8_t minimum = a;

    if (b < minimum) {
        minimum = b;
        action = 1;
    }

    if (c < minimum) {
        action = 2;
    }

    return action;
}

bool runInference(
    const int8_t test_input[4],
    int8_t result[3]
) {
    for (int i = 0; i < 4; i++) {
        input->data.int8[i] = test_input[i];
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke() failed.");
        return false;
    }

    for (int i = 0; i < 3; i++) {
        result[i] = output->data.int8[i];
    }

    return true;
}

void runValidationSet(
    const char* set_name,
    const int8_t test_inputs[][4],
    const int8_t expected_outputs[][3],
    const int8_t expected_actions[],
    int num_vectors
) {
    Serial.println();
    Serial.println("============================================================");
    Serial.print("Validation set: ");
    Serial.println(set_name);
    Serial.println("============================================================");

    int exact_vector_matches = 0;
    int individual_output_matches = 0;
    int action_matches = 0;
    int max_raw_difference = 0;

    for (int i = 0; i < num_vectors; i++) {

        int8_t actual[3];

        if (!runInference(test_inputs[i], actual)) {
            Serial.print("Inference failed at vector ");
            Serial.println(i);
            continue;
        }

        bool vector_exact = true;

        for (int j = 0; j < 3; j++) {

            int difference =
                abs(
                    (int)actual[j] -
                    (int)expected_outputs[i][j]
                );

            if (difference == 0) {
                individual_output_matches++;
            } else {
                vector_exact = false;
            }

            if (difference > max_raw_difference) {
                max_raw_difference = difference;
            }
        }

        if (vector_exact) {
            exact_vector_matches++;
        }

        int actual_action =
            argmin3(
                actual[0],
                actual[1],
                actual[2]
            );

        int expected_action =
            expected_actions[i];

        if (actual_action == expected_action) {
            action_matches++;
        }

        if (!vector_exact ||
            actual_action != expected_action) {

            Serial.println();

            Serial.print("Vector ");
            Serial.println(i);

            Serial.print("Input: [");

            for (int j = 0; j < 4; j++) {
                Serial.print(test_inputs[i][j]);

                if (j < 3) {
                    Serial.print(", ");
                }
            }

            Serial.println("]");

            Serial.print("Expected raw: [");
            Serial.print(expected_outputs[i][0]);
            Serial.print(", ");
            Serial.print(expected_outputs[i][1]);
            Serial.print(", ");
            Serial.print(expected_outputs[i][2]);
            Serial.println("]");

            Serial.print("ESP32 raw:    [");
            Serial.print(actual[0]);
            Serial.print(", ");
            Serial.print(actual[1]);
            Serial.print(", ");
            Serial.print(actual[2]);
            Serial.println("]");

            Serial.print("Expected action: ");
            Serial.println(expected_action);

            Serial.print("ESP32 action:    ");
            Serial.println(actual_action);
        }
    }

    Serial.println();
    Serial.println("---------------- SUMMARY ----------------");

    Serial.print("Vectors: ");
    Serial.println(num_vectors);

    Serial.print("Exact raw-output vectors: ");
    Serial.print(exact_vector_matches);
    Serial.print("/");
    Serial.println(num_vectors);

    Serial.print("Individual raw outputs: ");
    Serial.print(individual_output_matches);
    Serial.print("/");
    Serial.println(num_vectors * 3);

    Serial.print("Action agreement: ");
    Serial.print(action_matches);
    Serial.print("/");
    Serial.println(num_vectors);

    Serial.print("Maximum raw INT8 difference: ");
    Serial.println(max_raw_difference);

    Serial.println("-----------------------------------------");
}

void printTensorInformation() {

    Serial.println();
    Serial.println("============================================================");
    Serial.println("TENSOR INFORMATION");
    Serial.println("============================================================");

    Serial.print("Input type: ");
    Serial.println(input->type);

    Serial.print("Input scale: ");
    Serial.println(input->params.scale, 10);

    Serial.print("Input zero point: ");
    Serial.println(input->params.zero_point);

    Serial.println();

    Serial.print("Output type: ");
    Serial.println(output->type);

    Serial.print("Output scale: ");
    Serial.println(output->params.scale, 10);

    Serial.print("Output zero point: ");
    Serial.println(output->params.zero_point);
}

__attribute__((noinline))
int analyticalDecision(
    float C,
    float L,
    int T,
    int F
) {
    const float alpha_f = 0.25f;
    const float beta_f = 0.20f;

    float L_eff =
        L *
        (
            1.0f +
            alpha_f * (1.0f - (float)F)
        );

    float L0 = 2.5f * C + 50.0f;
    float L1 = 0.5f * L_eff + 30.0f;
    float L2 = L_eff + 10.0f;

    float P_comm =
        1.0f +
        beta_f * (1.0f - (float)F);

    float E0 = 0.8f * C;
    float E1 = 15.0f * P_comm;
    float E2 = 25.0f * P_comm;

    float L0n = L0 / 260.0f;
    float L1n = L1 / 260.0f;
    float L2n = L2 / 260.0f;

    float E0n = E0 / 80.0f;
    float E1n = E1 / 80.0f;
    float E2n = E2 / 80.0f;

    float wL;
    float wE;

    if (T == 1) {
        wL = 0.75f;
        wE = 0.25f;
    } else {
        wL = 0.50f;
        wE = 0.50f;
    }

    float J0 =
        wL * L0n +
        wE * E0n;

    float J1 =
        wL * L1n +
        wE * E1n;

    float J2 =
        wL * L2n +
        wE * E2n;

    int action = 0;
    float minimum = J0;

    if (J1 < minimum) {
        minimum = J1;
        action = 1;
    }

    if (J2 < minimum) {
        action = 2;
    }

    return action;
}

double benchmarkInference() {

    constexpr int WARMUP_RUNS = 100;
    constexpr int MEASURED_RUNS = 1000;

    for (int i = 0; i < 4; i++) {
        input->data.int8[i] =
            TEST_INPUTS[0][i];
    }

    for (int i = 0; i < WARMUP_RUNS; i++) {

        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println(
                "Warm-up Invoke() failed."
            );

            return -1.0;
        }
    }

    uint64_t total_individual = 0;
    uint32_t minimum_time = UINT32_MAX;
    uint32_t maximum_time = 0;

    for (int i = 0; i < MEASURED_RUNS; i++) {

        uint32_t start_time =
            micros();

        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println(
                "Measured Invoke() failed."
            );

            return -1.0;
        }

        uint32_t elapsed =
            micros() - start_time;

        total_individual += elapsed;

        if (elapsed < minimum_time) {
            minimum_time = elapsed;
        }

        if (elapsed > maximum_time) {
            maximum_time = elapsed;
        }
    }

    double individual_mean =
        (double)total_individual /
        (double)MEASURED_RUNS;

    uint32_t batch_start =
        micros();

    for (int i = 0; i < MEASURED_RUNS; i++) {

        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println(
                "Batch Invoke() failed."
            );

            return -1.0;
        }
    }

    uint32_t batch_total =
        micros() - batch_start;

    double batch_mean =
        (double)batch_total /
        (double)MEASURED_RUNS;

    Serial.println();
    Serial.println("============================================================");
    Serial.println("MODEL-ONLY INFERENCE LATENCY");
    Serial.println("============================================================");

    Serial.print("Warm-up runs: ");
    Serial.println(WARMUP_RUNS);

    Serial.print("Measured runs: ");
    Serial.println(MEASURED_RUNS);

    Serial.print("Individual mean (us): ");
    Serial.println(individual_mean, 3);

    Serial.print("Minimum (us): ");
    Serial.println(minimum_time);

    Serial.print("Maximum (us): ");
    Serial.println(maximum_time);

    Serial.print("Batch total (us): ");
    Serial.println(batch_total);

    Serial.print("Batch mean (us): ");
    Serial.println(batch_mean, 3);

    Serial.println("============================================================");

    return batch_mean;
}

double benchmarkAnalyticalPolicy() {

    constexpr int WARMUP_RUNS = 1000;
    constexpr int INDIVIDUAL_RUNS = 1000;
    constexpr int BATCH_RUNS = 100000;

    for (int i = 0; i < WARMUP_RUNS; i++) {

        int idx =
            i % NUM_ANALYTICAL_STATES;

        analytical_action_sink =
            analyticalDecision(
                ANALYTICAL_STATES[idx].C,
                ANALYTICAL_STATES[idx].L,
                ANALYTICAL_STATES[idx].T,
                ANALYTICAL_STATES[idx].F
            );
    }

    uint64_t total_individual = 0;
    uint32_t minimum_time = UINT32_MAX;
    uint32_t maximum_time = 0;

    for (int i = 0; i < INDIVIDUAL_RUNS; i++) {

        int idx =
            i % NUM_ANALYTICAL_STATES;

        uint32_t start_time =
            micros();

        analytical_action_sink =
            analyticalDecision(
                ANALYTICAL_STATES[idx].C,
                ANALYTICAL_STATES[idx].L,
                ANALYTICAL_STATES[idx].T,
                ANALYTICAL_STATES[idx].F
            );

        uint32_t elapsed =
            micros() - start_time;

        total_individual += elapsed;

        if (elapsed < minimum_time) {
            minimum_time = elapsed;
        }

        if (elapsed > maximum_time) {
            maximum_time = elapsed;
        }
    }

    double individual_mean =
        (double)total_individual /
        (double)INDIVIDUAL_RUNS;

    uint32_t batch_start =
        micros();

    for (int i = 0; i < BATCH_RUNS; i++) {

        int idx =
            i % NUM_ANALYTICAL_STATES;

        analytical_action_sink =
            analyticalDecision(
                ANALYTICAL_STATES[idx].C,
                ANALYTICAL_STATES[idx].L,
                ANALYTICAL_STATES[idx].T,
                ANALYTICAL_STATES[idx].F
            );
    }

    uint32_t batch_total =
        micros() - batch_start;

    double batch_mean =
        (double)batch_total /
        (double)BATCH_RUNS;

    Serial.println();
    Serial.println("============================================================");
    Serial.println("DIRECT ANALYTICAL POLICY LATENCY - MULTI-STATE");
    Serial.println("============================================================");

    Serial.print("Analytical states: ");
    Serial.println(NUM_ANALYTICAL_STATES);

    Serial.print("Warm-up runs: ");
    Serial.println(WARMUP_RUNS);

    Serial.print("Individual measured runs: ");
    Serial.println(INDIVIDUAL_RUNS);

    Serial.print("Batch measured runs: ");
    Serial.println(BATCH_RUNS);

    Serial.println();

    Serial.print("Individual mean (us): ");
    Serial.println(individual_mean, 3);

    Serial.print("Individual minimum (us): ");
    Serial.println(minimum_time);

    Serial.print("Individual maximum (us): ");
    Serial.println(maximum_time);

    Serial.println();

    Serial.print("Batch total (us): ");
    Serial.println(batch_total);

    Serial.print("Batch mean per decision (us): ");
    Serial.println(batch_mean, 6);

    Serial.print("Final analytical action: ");
    Serial.println(analytical_action_sink);

    Serial.println("============================================================");

    return batch_mean;
}

void printTimingComparison(
    double neural_batch_mean,
    double analytical_batch_mean
) {
    Serial.println();
    Serial.println("============================================================");
    Serial.println("NEURAL VS ANALYTICAL POLICY SUMMARY");
    Serial.println("============================================================");

    Serial.print("INT8 neural batch mean (us): ");
    Serial.println(neural_batch_mean, 3);

    Serial.print("Analytical multi-state batch mean (us): ");
    Serial.println(analytical_batch_mean, 6);

    if (
        neural_batch_mean > 0.0 &&
        analytical_batch_mean > 0.0
    ) {
        double ratio =
            neural_batch_mean /
            analytical_batch_mean;

        Serial.print(
            "Neural / analytical latency ratio: "
        );

        Serial.print(ratio, 3);
        Serial.println(" x");
    } else {
        Serial.println(
            "Latency ratio unavailable."
        );
    }

    Serial.println("============================================================");
}

void setup() {

    Serial.begin(115200);

    delay(1000);

    Serial.println();
    Serial.println("============================================================");
    Serial.println("INT8 OFFLOADING POLICY - ESP32 VALIDATION");
    Serial.println("============================================================");

    model =
        tflite::GetModel(
            offloading_model
        );

    if (
        model->version() !=
        TFLITE_SCHEMA_VERSION
    ) {

        Serial.print(
            "Model schema version: "
        );

        Serial.println(
            model->version()
        );

        Serial.print(
            "Supported schema version: "
        );

        Serial.println(
            TFLITE_SCHEMA_VERSION
        );

        return;
    }

    static
    tflite::MicroMutableOpResolver<1>
        resolver;

    if (
        resolver.AddFullyConnected()
        != kTfLiteOk
    ) {

        Serial.println(
            "AddFullyConnected() failed."
        );

        return;
    }

    static
    tflite::MicroInterpreter
        static_interpreter(
            model,
            resolver,
            tensor_arena,
            kTensorArenaSize
        );

    interpreter =
        &static_interpreter;

    TfLiteStatus allocate_status =
        interpreter->AllocateTensors();

    if (
        allocate_status
        != kTfLiteOk
    ) {

        Serial.println(
            "AllocateTensors() FAILED."
        );

        return;
    }

    Serial.println(
        "AllocateTensors() PASS"
    );

    Serial.print(
        "Configured tensor arena: "
    );

    Serial.print(
        kTensorArenaSize
    );

    Serial.println(
        " bytes"
    );

    input =
        interpreter->input(0);

    output =
        interpreter->output(0);

    if (
        input == nullptr ||
        output == nullptr
    ) {

        Serial.println(
            "Failed to obtain tensors."
        );

        return;
    }

    if (
        input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8
    ) {

        Serial.println(
            "ERROR: Expected INT8 input/output tensors."
        );

        return;
    }

    printTensorInformation();

    runValidationSet(
        "GENERAL DISTRIBUTED CASES",
        TEST_INPUTS,
        EXPECTED_OUTPUTS,
        EXPECTED_ACTIONS,
        NUM_TEST_VECTORS
    );

    runValidationSet(
        "EXACT INT8 TIE CASES",
        BOUNDARY_INPUTS,
        BOUNDARY_EXPECTED_OUTPUTS,
        BOUNDARY_EXPECTED_ACTIONS,
        NUM_BOUNDARY_VECTORS
    );

    runValidationSet(
        "NEAR-BOUNDARY CASES",
        NEAR_BOUNDARY_INPUTS,
        NEAR_BOUNDARY_EXPECTED_OUTPUTS,
        NEAR_BOUNDARY_EXPECTED_ACTIONS,
        NUM_NEAR_BOUNDARY_VECTORS
    );

    double neural_batch_mean =
        benchmarkInference();

    double analytical_batch_mean =
        benchmarkAnalyticalPolicy();

    printTimingComparison(
        neural_batch_mean,
        analytical_batch_mean
    );

    Serial.println();
    Serial.println(
        "Validation and multi-state analytical baseline complete."
    );
}

void loop() {
}
