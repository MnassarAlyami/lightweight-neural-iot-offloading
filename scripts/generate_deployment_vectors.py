import os
import numpy as np
import tensorflow as tf

OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(
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

ORACLE_PATH = os.path.join(
    OUTPUT_DIR,
    "oracle_actions.npy"
)

NUM_VECTORS = 20

x_test = np.load(X_TEST_PATH).astype(np.float32)
y_test = np.load(Y_TEST_PATH).astype(np.float32)
oracle_actions = np.load(ORACLE_PATH)

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = (
    input_details["quantization"]
)

output_scale, output_zero_point = (
    output_details["quantization"]
)

indices = np.linspace(
    0,
    len(x_test) - 1,
    NUM_VECTORS,
    dtype=int
)

normalized_inputs = []
quantized_inputs = []
raw_outputs = []
dequantized_outputs = []
selected_actions = []
selected_oracle_actions = []

for idx in indices:

    x = x_test[idx]

    q_input = np.round(
        x / input_scale + input_zero_point
    )

    q_input = np.clip(
        q_input,
        -128,
        127
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details["index"],
        q_input.reshape(1, 4)
    )

    interpreter.invoke()

    q_output = interpreter.get_tensor(
        output_details["index"]
    )[0].copy()

    output_float = (
        q_output.astype(np.float32)
        - output_zero_point
    ) * output_scale

    action = int(
        np.argmin(q_output)
    )

    oracle = int(
        oracle_actions[idx]
    )

    normalized_inputs.append(x)
    quantized_inputs.append(q_input)
    raw_outputs.append(q_output)
    dequantized_outputs.append(output_float)
    selected_actions.append(action)
    selected_oracle_actions.append(oracle)

normalized_inputs = np.asarray(
    normalized_inputs,
    dtype=np.float32
)

quantized_inputs = np.asarray(
    quantized_inputs,
    dtype=np.int8
)

raw_outputs = np.asarray(
    raw_outputs,
    dtype=np.int8
)

dequantized_outputs = np.asarray(
    dequantized_outputs,
    dtype=np.float32
)

selected_actions = np.asarray(
    selected_actions,
    dtype=np.int32
)

selected_oracle_actions = np.asarray(
    selected_oracle_actions,
    dtype=np.int32
)

print("=" * 75)
print("DEPLOYMENT VALIDATION VECTORS")
print("=" * 75)

print(f"\nModel: {MODEL_PATH}")
print(f"Number of vectors: {NUM_VECTORS}")

print("\nQuantization parameters")

print(
    f"Input scale       : {input_scale}"
)

print(
    f"Input zero point  : {input_zero_point}"
)

print(
    f"Output scale      : {output_scale}"
)

print(
    f"Output zero point : {output_zero_point}"
)

for i in range(NUM_VECTORS):

    print("\n" + "-" * 75)

    print(
        f"Vector {i + 1} "
        f"(test index {indices[i]})"
    )

    print(
        "Normalized input :",
        normalized_inputs[i]
    )

    print(
        "INT8 input       :",
        quantized_inputs[i]
    )

    print(
        "INT8 output      :",
        raw_outputs[i]
    )

    print(
        "Output costs     :",
        dequantized_outputs[i]
    )

    print(
        "INT8 action      :",
        selected_actions[i]
    )

    print(
        "Oracle action    :",
        selected_oracle_actions[i]
    )

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "deployment_vectors.npz"
    ),
    test_indices=indices,
    normalized_inputs=normalized_inputs,
    quantized_inputs=quantized_inputs,
    raw_outputs=raw_outputs,
    dequantized_outputs=dequantized_outputs,
    selected_actions=selected_actions,
    oracle_actions=selected_oracle_actions
)

header_path = os.path.join(
    OUTPUT_DIR,
    "deployment_vectors.h"
)

with open(
    header_path,
    "w"
) as f:

    f.write(
        "#ifndef DEPLOYMENT_VECTORS_H\n"
    )

    f.write(
        "#define DEPLOYMENT_VECTORS_H\n\n"
    )

    f.write(
        "#include <stdint.h>\n\n"
    )

    f.write(
        f"#define NUM_TEST_VECTORS {NUM_VECTORS}\n\n"
    )

    f.write(
        "const int8_t TEST_INPUTS"
        "[NUM_TEST_VECTORS][4] = {\n"
    )

    for row in quantized_inputs:

        values = ", ".join(
            str(int(v))
            for v in row
        )

        f.write(
            f"    {{{values}}},\n"
        )

    f.write(
        "};\n\n"
    )

    f.write(
        "const int8_t EXPECTED_OUTPUTS"
        "[NUM_TEST_VECTORS][3] = {\n"
    )

    for row in raw_outputs:

        values = ", ".join(
            str(int(v))
            for v in row
        )

        f.write(
            f"    {{{values}}},\n"
        )

    f.write(
        "};\n\n"
    )

    f.write(
        "const int8_t EXPECTED_ACTIONS"
        "[NUM_TEST_VECTORS] = {\n    "
    )

    f.write(
        ", ".join(
            str(int(v))
            for v in selected_actions
        )
    )

    f.write(
        "\n};\n\n"
    )

    f.write(
        "#endif\n"
    )

print("\n" + "=" * 75)

print("Saved:")
print("  outputs/deployment_vectors.npz")
print("  outputs/deployment_vectors.h")

print("\nDeployment-vector generation completed.")
