import os
import random
import numpy as np
import tensorflow as tf

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

NUM_SAMPLES = 50000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

HIDDEN_WIDTH = 16

EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

FEEDBACK_LATENCY_PENALTY = 0.25
FEEDBACK_ENERGY_PENALTY = 0.20

LATENCY_SCALE = 260.0
ENERGY_SCALE = 80.0

EPSILON = 1e-8

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_states(num_samples, seed):
    """
    Generate the four-dimensional system state:

        C = CPU utilization (%)
        L = network latency (ms)
        T = task type
            1 = real-time
            0 = delay-tolerant
        F = feedback
            1 = favorable
            0 = degraded

    The neural network receives normalized C and L.
    """

    rng = np.random.default_rng(seed)

    cpu = rng.uniform(0.0, 100.0, size=num_samples)
    network_latency = rng.uniform(0.0, 250.0, size=num_samples)

    task_type = rng.integers(0, 2, size=num_samples)
    feedback = rng.integers(0, 2, size=num_samples)

    states = np.column_stack([
        cpu,
        network_latency,
        task_type,
        feedback
    ])

    return states

def calculate_costs(states):
    """
    Calculate latency, energy, and final objective J
    for all three execution actions.

    Actions:
        0 = Local execution
        1 = Partial offloading
        2 = Full offloading
    """

    cpu = states[:, 0]
    network_latency = states[:, 1]
    task_type = states[:, 2]
    feedback = states[:, 3]

    effective_network_latency = (
        network_latency
        * (
            1.0
            + FEEDBACK_LATENCY_PENALTY * (1.0 - feedback)
        )
    )

    latency_local = (
        2.5 * cpu + 50.0
    )

    latency_partial = (
        0.5 * effective_network_latency + 30.0
    )

    latency_full = (
        effective_network_latency + 10.0
    )

    latency = np.column_stack([
        latency_local,
        latency_partial,
        latency_full
    ])

    energy_local = (
        0.8 * cpu
    )

    communication_penalty = (
        1.0
        + FEEDBACK_ENERGY_PENALTY * (1.0 - feedback)
    )

    energy_partial = 15.0 * communication_penalty
    energy_full = 25.0 * communication_penalty

    energy = np.column_stack([
        energy_local,
        np.full(len(states), energy_partial),
        np.full(len(states), energy_full)
    ])

    latency_normalized = latency / LATENCY_SCALE
    energy_normalized = energy / ENERGY_SCALE

    latency_weight = np.where(
        task_type == 1,
        0.75,
        0.50
    )

    energy_weight = np.where(
        task_type == 1,
        0.25,
        0.50
    )

    latency_weight = latency_weight[:, np.newaxis]
    energy_weight = energy_weight[:, np.newaxis]

    objective = (
        latency_weight * latency_normalized
        + energy_weight * energy_normalized
    )

    return latency, energy, objective

def normalize_inputs(states):
    """
    Convert raw state variables to the representation
    used by the neural network.

        CPU             -> CPU / 100
        Network latency -> latency / 250
        Task type       -> unchanged
        Feedback        -> unchanged
    """

    normalized = states.astype(np.float32).copy()

    normalized[:, 0] /= 100.0
    normalized[:, 1] /= 250.0

    return normalized

def create_dataset(num_samples, seed):
    """
    Generate states and corresponding three-action
    objective targets.
    """

    states = generate_states(
        num_samples=num_samples,
        seed=seed
    )

    _, _, objectives = calculate_costs(states)

    inputs = normalize_inputs(states)

    return inputs.astype(np.float32), objectives.astype(np.float32)

def split_dataset(inputs, targets, seed):
    """
    Randomly split the complete dataset into:

        70% training
        15% validation
        15% testing
    """

    rng = np.random.default_rng(seed)

    indices = np.arange(len(inputs))
    rng.shuffle(indices)

    n = len(indices)

    train_end = int(TRAIN_RATIO * n)
    val_end = train_end + int(VAL_RATIO * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        inputs[train_idx],
        targets[train_idx],
        inputs[val_idx],
        targets[val_idx],
        inputs[test_idx],
        targets[test_idx]
    )

def build_model():
    """
    4 -> 16 -> 16 -> 3

    The output contains the estimated objective for
    each of the three actions.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(4,),
            name="state"
        ),

        tf.keras.layers.Dense(
            HIDDEN_WIDTH,
            activation="relu",
            name="hidden_1"
        ),

        tf.keras.layers.Dense(
            HIDDEN_WIDTH,
            activation="relu",
            name="hidden_2"
        ),

        tf.keras.layers.Dense(
            3,
            activation="linear",
            name="action_costs"
        )
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model

def evaluate_policy(model, inputs, targets):
    """
    Evaluate the model as an offloading policy.

    The oracle selects the action with the minimum true cost.
    The neural policy selects the action with the minimum
    predicted cost.
    """

    predictions = model.predict(
        inputs,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    oracle_actions = np.argmin(
        targets,
        axis=1
    )

    predicted_actions = np.argmin(
        predictions,
        axis=1
    )

    agreement = np.mean(
        predicted_actions == oracle_actions
    )

    selected_indices = np.arange(len(targets))

    predicted_policy_cost = targets[
        selected_indices,
        predicted_actions
    ]

    oracle_cost = targets[
        selected_indices,
        oracle_actions
    ]

    regret = (
        predicted_policy_cost
        - oracle_cost
    )

    normalized_regret = (
        regret
        / (oracle_cost + EPSILON)
    )

    return {
        "agreement": agreement,
        "mean_regret": np.mean(regret),
        "median_regret": np.median(regret),
        "mean_normalized_regret": np.mean(
            normalized_regret
        ),
        "median_normalized_regret": np.median(
            normalized_regret
        ),
        "predictions": predictions,
        "oracle_actions": oracle_actions,
        "predicted_actions": predicted_actions,
        "regret": regret,
        "normalized_regret": normalized_regret
    }

def main():

    print("=" * 60)
    print("Clean Offloading Model Training Pipeline")
    print("=" * 60)

    print("\nGenerating dataset...")

    inputs, targets = create_dataset(
        num_samples=NUM_SAMPLES,
        seed=SEED
    )

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    ) = split_dataset(
        inputs,
        targets,
        seed=SEED
    )

    print(f"Total samples : {len(inputs)}")
    print(f"Training      : {len(x_train)}")
    print(f"Validation    : {len(x_val)}")
    print(f"Testing       : {len(x_test)}")

    model = build_model()

    print("\nModel:")
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    print("\nTraining...")

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\nEvaluating regression performance...")

    test_loss, test_mae = model.evaluate(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    print("Evaluating offloading policy...")

    results = evaluate_policy(
        model,
        x_test,
        y_test
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test MSE                  : {test_loss:.8f}")
    print(f"Test MAE                  : {test_mae:.8f}")
    print(
        f"Oracle agreement          : "
        f"{results['agreement'] * 100:.4f}%"
    )
    print(
        f"Mean regret               : "
        f"{results['mean_regret']:.10e}"
    )
    print(
        f"Median regret             : "
        f"{results['median_regret']:.10e}"
    )
    print(
        f"Mean normalized regret    : "
        f"{results['mean_normalized_regret']:.10e}"
    )
    print(
        f"Median normalized regret  : "
        f"{results['median_normalized_regret']:.10e}"
    )

    model_path = os.path.join(
        OUTPUT_DIR,
        "offloading_fp32.keras"
    )

    model.save(model_path)

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "x_test.npy"
        ),
        x_test
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_test.npy"
        ),
        y_test
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "test_predictions.npy"
        ),
        results["predictions"]
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "oracle_actions.npy"
        ),
        results["oracle_actions"]
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "predicted_actions.npy"
        ),
        results["predicted_actions"]
    )

    print("\nSaved:")
    print(f"  {model_path}")
    print("  outputs/x_test.npy")
    print("  outputs/y_test.npy")
    print("  outputs/test_predictions.npy")
    print("  outputs/oracle_actions.npy")
    print("  outputs/predicted_actions.npy")

    print("\nPipeline completed.")

if __name__ == "__main__":
    main()
