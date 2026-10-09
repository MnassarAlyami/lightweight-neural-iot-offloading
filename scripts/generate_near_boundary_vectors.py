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

NUM_BOUNDARY_VECTORS = 30

x_test = np.load(
    X_TEST_PATH
).astype(np.float32)

y_test = np.load(
    Y_TEST_PATH
).astype(np.float32)

oracle_actions = np.load(
    ORACLE_PATH
)

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

print("=" * 75)
print("INT8 DECISION-BOUNDARY VECTOR GENERATION")
print("=" * 75)

print(f"\nTest samples : {len(x_test)}")

print(
    f"Input quantization  : "
    f"scale={input_scale}, "
    f"zero_point={input_zero_point}"
)

print(
    f"Output quantization : "
    f"scale={output_scale}, "
    f"zero_point={output_zero_point}"
)

all_quantized_inputs = []
all_raw_outputs = []
all_dequantized_outputs = []
all_actions = []
all_margins = []

for x in x_test:

    q_input = np.round(
        x / input_scale
        + input_zero_point
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

    sorted_output = np.sort(
        q_output.astype(np.int16)
    )

    margin = int(
        sorted_output[1]
        - sorted_output[0]
    )

    all_quantized_inputs.append(
        q_input
    )

    all_raw_outputs.append(
        q_output
    )

    all_dequantized_outputs.append(
        output_float
    )

    all_actions.append(
        action
    )

    all_margins.append(
        margin
    )

all_quantized_inputs = np.asarray(
    all_quantized_inputs,
    dtype=np.int8
)

all_raw_outputs = np.asarray(
    all_raw_outputs,
    dtype=np.int8
)

all_dequantized_outputs = np.asarray(
    all_dequantized_outputs,
    dtype=np.float32
)

all_actions = np.asarray(
    all_actions,
    dtype=np.int32
)

all_margins = np.asarray(
    all_margins,
    dtype=np.int32
)

nonzero_indices = np.where(
    all_margins > 0
)[0]

rank_order = np.argsort(
    all_margins[nonzero_indices],
    kind="stable"
)

boundary_indices = nonzero_indices[
    rank_order[:NUM_BOUNDARY_VECTORS]
]

boundary_inputs_float = x_test[
    boundary_indices
]

boundary_inputs_int8 = all_quantized_inputs[
    boundary_indices
]

boundary_outputs_int8 = all_raw_outputs[
    boundary_indices
]

boundary_outputs_float = all_dequantized_outputs[
    boundary_indices
]

boundary_actions = all_actions[
    boundary_indices
]

boundary_oracle_actions = oracle_actions[
    boundary_indices
]

boundary_margins = all_margins[
    boundary_indices
]

objective_margins = []

for idx in boundary_indices:

    costs = np.sort(
        y_test[idx]
    )

    margin = float(
        costs[1] - costs[0]
    )

    objective_margins.append(
        margin
    )

objective_margins = np.asarray(
    objective_margins,
    dtype=np.float32
)

print("\n" + "=" * 75)
print("SELECTED DECISION-BOUNDARY CASES")
print("=" * 75)

for i in range(NUM_BOUNDARY_VECTORS):

    idx = boundary_indices[i]

    print("\n" + "-" * 75)

    print(
        f"Boundary vector {i + 1} "
        f"(test index {idx})"
    )

    print(
        "Normalized input :",
        boundary_inputs_float[i]
    )

    print(
        "INT8 input       :",
        boundary_inputs_int8[i]
    )

    print(
        "INT8 output      :",
        boundary_outputs_int8[i]
    )

    print(
        "Output costs     :",
        boundary_outputs_float[i]
    )

    print(
        "INT8 margin      :",
        boundary_margins[i]
    )

    print(
        "Objective margin :",
        f"{objective_margins[i]:.10e}"
    )

    print(
        "INT8 action      :",
        boundary_actions[i]
    )

    print(
        "Oracle action    :",
        boundary_oracle_actions[i]
    )

print("\n" + "=" * 75)
print("BOUNDARY SET SUMMARY")
print("=" * 75)

print(
    f"Selected vectors       : "
    f"{NUM_BOUNDARY_VECTORS}"
)

print(
    f"Minimum INT8 margin    : "
    f"{np.min(boundary_margins)}"
)

print(
    f"Maximum INT8 margin    : "
    f"{np.max(boundary_margins)}"
)

print(
    f"Median INT8 margin     : "
    f"{np.median(boundary_margins):.2f}"
)

boundary_oracle_agreement = np.mean(
    boundary_actions
    == boundary_oracle_actions
)

print(
    f"INT8/oracle agreement  : "
    f"{boundary_oracle_agreement * 100:.4f}%"
)

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "near_boundary_vectors.npz"
    ),
    test_indices=boundary_indices,
    normalized_inputs=boundary_inputs_float,
    quantized_inputs=boundary_inputs_int8,
    raw_outputs=boundary_outputs_int8,
    dequantized_outputs=boundary_outputs_float,
    int8_margins=boundary_margins,
    objective_margins=objective_margins,
    selected_actions=boundary_actions,
    oracle_actions=boundary_oracle_actions
)

header_path = os.path.join(
    OUTPUT_DIR,
    "near_boundary_vectors.h"
)

with open(header_path, "w") as f:

    f.write(
        "#ifndef NEAR_BOUNDARY_VECTORS_H\n"
    )
    f.write(
        "#define NEAR_BOUNDARY_VECTORS_H\n\n"
    )

    f.write(
        "#include <stdint.h>\n\n"
    )

    f.write(
        f"#define NUM_NEAR_BOUNDARY_VECTORS "
        f"{NUM_BOUNDARY_VECTORS}\n\n"
    )

    f.write(
        "const int8_t NEAR_BOUNDARY_INPUTS"
        "[NUM_NEAR_BOUNDARY_VECTORS][4] = {\n"
    )

    for row in boundary_inputs_int8:
        values = ", ".join(
            str(int(v)) for v in row
        )
        f.write(
            f"    {{{values}}},\n"
        )

    f.write("};\n\n")

    f.write(
        "const int8_t NEAR_BOUNDARY_EXPECTED_OUTPUTS"
        "[NUM_NEAR_BOUNDARY_VECTORS][3] = {\n"
    )

    for row in boundary_outputs_int8:
        values = ", ".join(
            str(int(v)) for v in row
        )
        f.write(
            f"    {{{values}}},\n"
        )

    f.write("};\n\n")

    f.write(
        "const int8_t NEAR_BOUNDARY_EXPECTED_ACTIONS"
        "[NUM_NEAR_BOUNDARY_VECTORS] = {\n    "
    )

    f.write(
        ", ".join(
            str(int(v)) for v in boundary_actions
        )
    )

    f.write("\n};\n\n")

    f.write(
        "const int16_t NEAR_BOUNDARY_MARGINS"
        "[NUM_NEAR_BOUNDARY_VECTORS] = {\n    "
    )

    f.write(
        ", ".join(
            str(int(v)) for v in boundary_margins
        )
    )

    f.write("\n};\n\n")

    f.write(
        "#endif  // NEAR_BOUNDARY_VECTORS_H\n"
    )

print("\nSaved:")
print("  outputs/near_boundary_vectors.npz")
print("  outputs/near_boundary_vectors.h")

print(
    "\nNear-boundary vector generation completed."
)
