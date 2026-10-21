import os
import numpy as np
import tensorflow as tf

OUTPUT_DIR = "outputs"

KERAS_MODEL_PATH = os.path.join(OUTPUT_DIR, "offloading_fp32.keras")

TFLITE_MODEL_PATH = os.path.join(OUTPUT_DIR, "offloading_fp32.tflite")

X_TEST_PATH = os.path.join(OUTPUT_DIR, "x_test.npy")

Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test.npy")

REFERENCE_ACTIONS_PATH = os.path.join(OUTPUT_DIR, "reference_actions.npy")

EPSILON = 1e-8


def load_artifacts():
    print("Loading generated training artifacts...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    x_test = np.load(X_TEST_PATH).astype(np.float32)
    y_test = np.load(Y_TEST_PATH).astype(np.float32)
    reference_actions = np.load(REFERENCE_ACTIONS_PATH)
    print(f"Loaded Keras model : {KERAS_MODEL_PATH}")
    print(f"Test samples       : {len(x_test)}")
    return model, x_test, y_test, reference_actions


def convert_to_fp32_tflite(model):
    print("\nConverting Keras model to FP32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    model_size_bytes = os.path.getsize(TFLITE_MODEL_PATH)
    print(f"Saved FP32 TFLite model : {TFLITE_MODEL_PATH}")
    print(f"TFLite model size : {model_size_bytes} bytes ({model_size_bytes / 1024:.3f} KiB)")
    return model_size_bytes


def create_interpreter():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("\nTFLite input tensor:")
    print(f" Name : {input_details[0]['name']}")
    print(f" Shape : {input_details[0]['shape']}")
    print(f" Data type : {input_details[0]['dtype']}")
    print(f" Quantization: {input_details[0]['quantization']}")
    print("\nTFLite output tensor:")
    print(f" Name : {output_details[0]['name']}")
    print(f" Shape : {output_details[0]['shape']}")
    print(f" Data type : {output_details[0]['dtype']}")
    print(f" Quantization: {output_details[0]['quantization']}")
    return (interpreter, input_details, output_details)


def run_tflite_inference(interpreter, input_details, output_details, x_test):
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    predictions = []
    print(f"\nRunning FP32 TFLite inference on {len(x_test)} test samples...")
    for sample in x_test:
        sample = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        predictions.append(output[0].copy())

    return np.asarray(predictions, dtype=np.float32)


def evaluate_equivalence(keras_model, x_test, y_test, reference_actions, tflite_predictions):
    print("\nGenerating Keras FP32 reference predictions...")
    keras_predictions = keras_model.predict(x_test, batch_size=64, verbose=0)
    keras_actions = np.argmin(keras_predictions, axis=1)
    tflite_actions = np.argmin(tflite_predictions, axis=1)
    absolute_difference = np.abs(keras_predictions - tflite_predictions)
    max_abs_difference = np.max(absolute_difference)
    mean_abs_difference = np.mean(absolute_difference)
    keras_tflite_agreement = np.mean(keras_actions == tflite_actions)
    number_action_disagreements = np.sum(keras_actions != tflite_actions)
    tflite_reference_agreement = np.mean(tflite_actions == reference_actions)
    tflite_reference_disagreements = np.sum(tflite_actions != reference_actions)
    indices = np.arange(len(y_test))
    selected_cost = y_test[indices, tflite_actions]
    reference_cost = y_test[indices, reference_actions]
    regret = selected_cost - reference_cost
    normalized_regret = regret / (reference_cost + EPSILON)
    print("\n" + "=" * 65)
    print("FP32 TFLITE VALIDATION RESULTS")
    print("=" * 65)
    print(f"Maximum prediction difference : {max_abs_difference:.10e}")
    print(f"Mean prediction difference : {mean_abs_difference:.10e}")
    print(f"Keras -> TFLite agreement : {keras_tflite_agreement * 100:.6f}%")
    print(f"Keras/TFLite disagreements : {number_action_disagreements}")
    print(f"TFLite -> reference agreement : {tflite_reference_agreement * 100:.6f}%")
    print(f"TFLite/reference disagreements : {tflite_reference_disagreements}")
    print(f"Mean TFLite regret : {np.mean(regret):.10e}")
    print(f"Median TFLite regret : {np.median(regret):.10e}")
    print(f"Mean normalized regret : {np.mean(normalized_regret):.10e}")
    print(f"Median normalized regret : {np.median(normalized_regret):.10e}")
    disagreement_indices = np.where(keras_actions != tflite_actions)[0]
    if len(disagreement_indices) > 0:
        print("\nWARNING: Keras and TFLite selected different actions.")
        print("First disagreement indices:", disagreement_indices[:10])
    else:
        print(
            "\nPASS: FP32 Keras and FP32 TFLite "
            "selected identical actions for every "
            "test sample."
        )

    return {
        "keras_predictions": keras_predictions,
        "tflite_predictions": tflite_predictions,
        "keras_actions": keras_actions,
        "tflite_actions": tflite_actions,
        "regret": regret,
        "normalized_regret": normalized_regret
    }


def save_validation_outputs(results):
    np.save(os.path.join(OUTPUT_DIR, "fp32_tflite_predictions.npy"), results["tflite_predictions"])
    np.save(os.path.join(OUTPUT_DIR, "fp32_tflite_actions.npy"), results["tflite_actions"])
    print("\nSaved validation outputs:")
    print(" outputs/fp32_tflite_predictions.npy")
    print(" outputs/fp32_tflite_actions.npy")


def main():
    print("=" * 65)
    print("FP32 TFLite Conversion and Validation")
    print("=" * 65)
    (keras_model, x_test, y_test, reference_actions) = load_artifacts()
    model_size = convert_to_fp32_tflite(keras_model)
    (interpreter, input_details, output_details) = create_interpreter()
    assert (input_details[0]["dtype"] == np.float32), "FP32 TFLite input is not float32."
    assert (output_details[0]["dtype"] == np.float32), "FP32 TFLite output is not float32."
    assert tuple(input_details[0]["shape"]) == (1, 4), ("Unexpected input shape.")
    assert tuple(output_details[0]["shape"]) == (1, 3), ("Unexpected output shape.")
    tflite_predictions = run_tflite_inference(interpreter, input_details, output_details, x_test)
    results = evaluate_equivalence(keras_model, x_test, y_test, reference_actions, tflite_predictions)
    save_validation_outputs(results)
    print("\n" + "=" * 65)
    print("FP32 TFLite validation completed.")
    print("=" * 65)

if __name__ == "__main__":
    main()
