import os
import numpy as np
import tensorflow as tf

SEED = 42
NUM_SAMPLES = 50000

TRAIN_RATIO = 0.70

CALIBRATION_SAMPLES = 5000

def reconstruct_training_inputs():

    rng = np.random.default_rng(SEED)

    cpu = rng.uniform(
        0.0,
        100.0,
        size=NUM_SAMPLES
    )

    network_latency = rng.uniform(
        0.0,
        250.0,
        size=NUM_SAMPLES
    )

    task_type = rng.integers(
        0,
        2,
        size=NUM_SAMPLES
    )

    feedback = rng.integers(
        0,
        2,
        size=NUM_SAMPLES
    )

    states = np.column_stack([
        cpu,
        network_latency,
        task_type,
        feedback
    ])

    inputs = states.astype(
        np.float32
    ).copy()

    inputs[:, 0] /= 100.0
    inputs[:, 1] /= 250.0

    split_rng = np.random.default_rng(SEED)

    indices = np.arange(
        NUM_SAMPLES
    )

    split_rng.shuffle(
        indices
    )

    train_end = int(
        TRAIN_RATIO * NUM_SAMPLES
    )

    train_indices = indices[
        :train_end
    ]

    x_train = inputs[
        train_indices
    ]

    return x_train

OUTPUT_DIR = "outputs"

KERAS_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "offloading_fp32.keras"
)

INT8_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "offloading_int8.tflite"
)

X_TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "x_test.npy"
)

Y_TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "y_test.npy"
)

ORACLE_ACTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "oracle_actions.npy"
)

FP32_TFLITE_ACTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "fp32_tflite_actions.npy"
)

FP32_TFLITE_PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "fp32_tflite_predictions.npy"
)

EPSILON = 1e-8

def load_artifacts():

    print("Loading frozen artifacts...")

    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH
    )

    x_test = np.load(
        X_TEST_PATH
    ).astype(np.float32)

    y_test = np.load(
        Y_TEST_PATH
    ).astype(np.float32)

    oracle_actions = np.load(
        ORACLE_ACTIONS_PATH
    )

    fp32_actions = np.load(
        FP32_TFLITE_ACTIONS_PATH
    )

    fp32_predictions = np.load(
        FP32_TFLITE_PREDICTIONS_PATH
    ).astype(np.float32)

    print(f"Loaded model       : {KERAS_MODEL_PATH}")
    print(f"Test samples       : {len(x_test)}")

    return (
        model,
        x_test,
        y_test,
        oracle_actions,
        fp32_actions,
        fp32_predictions
    )

def make_representative_dataset(x_train):

    calibration_data = x_train[
        :min(CALIBRATION_SAMPLES, len(x_train))
    ]

    def representative_dataset():

        for sample in calibration_data:

            sample = np.expand_dims(
                sample,
                axis=0
            ).astype(np.float32)

            yield [sample]

    return representative_dataset

def convert_to_int8(model, x_train):

    print("\nConverting model to full-integer INT8 TFLite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = (
    make_representative_dataset(x_train)
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(
        INT8_MODEL_PATH,
        "wb"
    ) as f:
        f.write(tflite_model)

    model_size_bytes = os.path.getsize(
        INT8_MODEL_PATH
    )

    print(
        f"Saved INT8 TFLite model : "
        f"{INT8_MODEL_PATH}"
    )

    print(
        f"INT8 model size         : "
        f"{model_size_bytes} bytes "
        f"({model_size_bytes / 1024:.3f} KB)"
    )

    return model_size_bytes

def create_interpreter():

    interpreter = tf.lite.Interpreter(
        model_path=INT8_MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nINT8 input tensor:")
    print(
        f"  Name        : {input_details[0]['name']}"
    )
    print(
        f"  Shape       : {input_details[0]['shape']}"
    )
    print(
        f"  Data type   : {input_details[0]['dtype']}"
    )
    print(
        f"  Quantization: {input_details[0]['quantization']}"
    )

    print("\nINT8 output tensor:")
    print(
        f"  Name        : {output_details[0]['name']}"
    )
    print(
        f"  Shape       : {output_details[0]['shape']}"
    )
    print(
        f"  Data type   : {output_details[0]['dtype']}"
    )
    print(
        f"  Quantization: {output_details[0]['quantization']}"
    )

    assert input_details[0]["dtype"] == np.int8, (
        "Model input is not INT8."
    )

    assert output_details[0]["dtype"] == np.int8, (
        "Model output is not INT8."
    )

    assert tuple(
        input_details[0]["shape"]
    ) == (1, 4), "Unexpected INT8 input shape."

    assert tuple(
        output_details[0]["shape"]
    ) == (1, 3), "Unexpected INT8 output shape."

    return (
        interpreter,
        input_details,
        output_details
    )

def quantize_input(float_input, scale, zero_point):

    if scale == 0:
        raise ValueError(
            "Input quantization scale is zero."
        )

    quantized = np.round(
        float_input / scale + zero_point
    )

    quantized = np.clip(
        quantized,
        -128,
        127
    )

    return quantized.astype(np.int8)

def dequantize_output(int8_output, scale, zero_point):

    if scale == 0:
        raise ValueError(
            "Output quantization scale is zero."
        )

    return (
        int8_output.astype(np.float32)
        - zero_point
    ) * scale

def run_int8_inference(
    interpreter,
    input_details,
    output_details,
    x_test
):

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_scale, input_zero_point = (
        input_details[0]["quantization"]
    )

    output_scale, output_zero_point = (
        output_details[0]["quantization"]
    )

    int8_outputs = []
    dequantized_predictions = []

    print(
        "\nRunning INT8 TFLite inference on "
        f"{len(x_test)} test samples..."
    )

    for sample in x_test:

        sample = np.expand_dims(
            sample,
            axis=0
        ).astype(np.float32)

        quantized_sample = quantize_input(
            sample,
            input_scale,
            input_zero_point
        )

        interpreter.set_tensor(
            input_index,
            quantized_sample
        )

        interpreter.invoke()

        output_int8 = interpreter.get_tensor(
            output_index
        )

        output_float = dequantize_output(
            output_int8,
            output_scale,
            output_zero_point
        )

        int8_outputs.append(
            output_int8[0].copy()
        )

        dequantized_predictions.append(
            output_float[0].copy()
        )

    return (
        np.asarray(
            int8_outputs,
            dtype=np.int8
        ),
        np.asarray(
            dequantized_predictions,
            dtype=np.float32
        )
    )

def evaluate_int8(
    y_test,
    oracle_actions,
    fp32_actions,
    fp32_predictions,
    int8_predictions
):

    int8_actions = np.argmin(
        int8_predictions,
        axis=1
    )

    fp32_int8_agreement = np.mean(
        fp32_actions == int8_actions
    )

    fp32_int8_disagreements = np.sum(
        fp32_actions != int8_actions
    )

    int8_oracle_agreement = np.mean(
        int8_actions == oracle_actions
    )

    int8_oracle_disagreements = np.sum(
        int8_actions != oracle_actions
    )

    abs_difference = np.abs(
        fp32_predictions
        - int8_predictions
    )

    mean_prediction_difference = np.mean(
        abs_difference
    )

    max_prediction_difference = np.max(
        abs_difference
    )

    indices = np.arange(
        len(y_test)
    )

    selected_cost = y_test[
        indices,
        int8_actions
    ]

    oracle_cost = y_test[
        indices,
        oracle_actions
    ]

    regret = (
        selected_cost
        - oracle_cost
    )

    normalized_regret = (
        regret
        / (oracle_cost + EPSILON)
    )

    print("\n" + "=" * 65)
    print("FULL-INT8 TFLITE VALIDATION RESULTS")
    print("=" * 65)

    print(
        f"FP32 -> INT8 agreement       : "
        f"{fp32_int8_agreement * 100:.6f}%"
    )

    print(
        f"FP32/INT8 disagreements      : "
        f"{fp32_int8_disagreements}"
    )

    print(
        f"INT8 -> oracle agreement     : "
        f"{int8_oracle_agreement * 100:.6f}%"
    )

    print(
        f"INT8/oracle disagreements    : "
        f"{int8_oracle_disagreements}"
    )

    print(
        f"Mean prediction difference   : "
        f"{mean_prediction_difference:.10e}"
    )

    print(
        f"Maximum prediction difference: "
        f"{max_prediction_difference:.10e}"
    )

    print(
        f"Mean INT8 regret             : "
        f"{np.mean(regret):.10e}"
    )

    print(
        f"Median INT8 regret           : "
        f"{np.median(regret):.10e}"
    )

    print(
        f"Mean normalized regret       : "
        f"{np.mean(normalized_regret):.10e}"
    )

    print(
        f"Median normalized regret     : "
        f"{np.median(normalized_regret):.10e}"
    )

    return {
        "int8_actions": int8_actions,
        "regret": regret,
        "normalized_regret": normalized_regret
    }

def save_results(
    int8_raw_outputs,
    int8_predictions,
    results
):

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "int8_raw_outputs.npy"
        ),
        int8_raw_outputs
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "int8_predictions.npy"
        ),
        int8_predictions
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "int8_actions.npy"
        ),
        results["int8_actions"]
    )

    print("\nSaved INT8 validation outputs:")
    print("  outputs/int8_raw_outputs.npy")
    print("  outputs/int8_predictions.npy")
    print("  outputs/int8_actions.npy")

def main():

    print("=" * 65)
    print("Full-Integer INT8 TFLite Conversion and Validation")
    print("=" * 65)

    (
        model,
        x_test,
        y_test,
        oracle_actions,
        fp32_actions,
        fp32_predictions
    ) = load_artifacts()

    convert_to_int8(
        model,
        x_train
    )

    (
        interpreter,
        input_details,
        output_details
    ) = create_interpreter()

    (
        int8_raw_outputs,
        int8_predictions
    ) = run_int8_inference(
        interpreter,
        input_details,
        output_details,
        x_test
    )

    results = evaluate_int8(
        y_test,
        oracle_actions,
        fp32_actions,
        fp32_predictions,
        int8_predictions
    )

    save_results(
        int8_raw_outputs,
        int8_predictions,
        results
    )

    print("\n" + "=" * 65)
    print("Full-INT8 validation completed.")
    print("=" * 65)

print("\nReconstructing training data for INT8 calibration...")

x_train = reconstruct_training_inputs()

print(
    f"Training samples available for calibration: "
    f"{len(x_train)}"
)

print(
    f"Representative calibration samples: "
    f"{min(CALIBRATION_SAMPLES, len(x_train))}"
)

if __name__ == "__main__":
    main()
