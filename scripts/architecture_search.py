"""Search compact neural cost estimators across width, depth, and five random seeds."""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SEEDS = [1, 2, 3, 4, 5]

torch.set_num_threads(1)

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

HIDDEN_WIDTHS = [
    4, 6, 8, 10, 12,
    16, 20, 24, 32,
    40, 48, 64, 80,
    96, 128
]

NUM_BLOCKS = [
    1, 2, 3, 4, 5, 6, 8
]

# These are allocated MODEL budgets.
# They are NOT total MCU Flash/SRAM limits.

MODEL_FLASH_BUDGET_BYTES = 256 * 1024
MODEL_ACTIVATION_BUDGET_BYTES = 64 * 1024

MODEL_MAC_BUDGET = 1_000_000

NUM_TRAIN_SAMPLES = 5000
NUM_VALIDATION_SAMPLES = 1000
NUM_TEST_SAMPLES = 5000

TRAINING_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

CPU_MAX = 100.0
NETWORK_LATENCY_MAX = 250.0

def normalize_states(states):

    x = states.copy().astype(np.float32)

    x[:, 0] /= CPU_MAX
    x[:, 1] /= NETWORK_LATENCY_MAX

    return x

def generate_contexts(num_samples, seed):

    rng = np.random.default_rng(seed)

    cpu = rng.uniform(
        0,
        100,
        num_samples
    )

    network_latency = rng.uniform(
        5,
        250,
        num_samples
    )

    task_type = rng.integers(
        0,
        2,
        num_samples
    )

    feedback = rng.integers(
        0,
        2,
        num_samples
    )

    return np.column_stack([
        cpu,
        network_latency,
        task_type,
        feedback
    ]).astype(np.float32)

FEEDBACK_LATENCY_PENALTY = 0.25
FEEDBACK_ENERGY_PENALTY = 0.20

def calculate_costs(states):

    cpu = states[:, 0]
    network_latency = states[:, 1]
    feedback = states[:, 3]

    # Degraded feedback increases effective communication latency.

    effective_network_latency = (
        network_latency *
        (
            1.0
            +
            FEEDBACK_LATENCY_PENALTY
            *
            (1.0 - feedback)
        )
    )

    local_latency = (
        2.5 * cpu + 50.0
    )

    partial_latency = (
        0.5 *
        effective_network_latency
        + 30.0
    )

    full_latency = (
        effective_network_latency
        + 10.0
    )

    latency = np.column_stack([
        local_latency,
        partial_latency,
        full_latency
    ])

    local_energy = (
        0.8 * cpu
    )

    communication_multiplier = (
        1.0
        +
        FEEDBACK_ENERGY_PENALTY
        *
        (1.0 - feedback)
    )

    partial_energy = (
        15.0 *
        communication_multiplier
    )

    full_energy = (
        25.0 *
        communication_multiplier
    )

    energy = np.column_stack([
        local_energy,
        partial_energy,
        full_energy
    ])

    return latency, energy

LATENCY_SCALE = 260.0
ENERGY_SCALE = 80.0

def calculate_objective(states):

    latency, energy = calculate_costs(
        states
    )

    task_type = states[:, 2]

    latency_normalized = (
        latency / LATENCY_SCALE
    )

    energy_normalized = (
        energy / ENERGY_SCALE
    )

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

    objective = (
        latency_weight[:, None]
        *
        latency_normalized
        +
        energy_weight[:, None]
        *
        energy_normalized
    )

    return objective.astype(
        np.float32
    )

def reference_policy(states):

    objective = calculate_objective(
        states
    )

    return np.argmin(
        objective,
        axis=1
    )

def create_dataset(
    num_samples,
    seed
):

    states = generate_contexts(
        num_samples,
        seed
    )

    normalized_states = (
        normalize_states(states)
    )

    objective = calculate_objective(
        states
    )

    return (
        torch.tensor(
            normalized_states,
            dtype=torch.float32
        ),

        torch.tensor(
            objective,
            dtype=torch.float32
        )
    )

class TinyMLNetwork(nn.Module):

    def __init__(
        self,
        hidden_width,
        num_blocks
    ):

        super().__init__()

        layers = []

        layers.append(
            nn.Linear(
                4,
                hidden_width
            )
        )

        layers.append(
            nn.ReLU()
        )

        for _ in range(num_blocks):

            layers.append(
                nn.Linear(
                    hidden_width,
                    hidden_width
                )
            )

            layers.append(
                nn.ReLU()
            )

        # Predict objective for each action.
        layers.append(
            nn.Linear(
                hidden_width,
                3
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(x)

def train_model(
    model,
    states,
    objective_targets,
    seed
):

    set_seed(seed)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.MSELoss()

    model.train()

    n = len(states)

    for epoch in range(
        TRAINING_EPOCHS
    ):

        permutation = torch.randperm(n)

        for start in range(
            0,
            n,
            BATCH_SIZE
        ):

            indices = permutation[
                start:start + BATCH_SIZE
            ]

            x = states[
                indices
            ]

            target = objective_targets[
                indices
            ]

            optimizer.zero_grad()

            predicted_objective = (
                model(x)
            )

            loss = loss_function(
                predicted_objective,
                target
            )

            loss.backward()

            optimizer.step()

    return model

def evaluate_model(
    model,
    states_raw
):

    model.eval()

    states_normalized = (
        normalize_states(
            states_raw
        )
    )

    x = torch.tensor(
        states_normalized,
        dtype=torch.float32
    )

    with torch.no_grad():

        predicted_objective = (
            model(x)
            .cpu()
            .numpy()
        )

    predicted_actions = np.argmin(
        predicted_objective,
        axis=1
    )

    true_objective = (
        calculate_objective(
            states_raw
        )
    )

    oracle_actions = np.argmin(
        true_objective,
        axis=1
    )

    indices = np.arange(
        len(states_raw)
    )

    selected_objective = (
        true_objective[
            indices,
            predicted_actions
        ]
    )

    oracle_objective = (
        true_objective[
            indices,
            oracle_actions
        ]
    )

    regret = (
        selected_objective
        -
        oracle_objective
    )

    normalized_regret = (
        regret
        /
        (oracle_objective + 1e-8)
    )

    latency, energy = (
        calculate_costs(
            states_raw
        )
    )

    selected_latency = (
        latency[
            indices,
            predicted_actions
        ]
    )

    selected_energy = (
        energy[
            indices,
            predicted_actions
        ]
    )

    return {

        "reference_agreement":
            np.mean(
                predicted_actions
                ==
                oracle_actions
            ),

        "mean_objective":
            np.mean(
                selected_objective
            ),

        "oracle_objective":
            np.mean(
                oracle_objective
            ),

        "mean_regret":
            np.mean(regret),

        "median_regret":
            np.median(regret),

        "mean_normalized_regret":
            np.mean(
                normalized_regret
            ),

        "mean_latency":
            np.mean(
                selected_latency
            ),

        "median_latency":
            np.median(
                selected_latency
            ),

        "mean_energy":
            np.mean(
                selected_energy
            ),

        "median_energy":
            np.median(
                selected_energy
            ),

        "local_fraction":
            np.mean(
                predicted_actions == 0
            ),

        "partial_fraction":
            np.mean(
                predicted_actions == 1
            ),

        "full_fraction":
            np.mean(
                predicted_actions == 2
            )
    }

def evaluate_int8_screening(
    model,
    states_raw
):

    """
    Screening only.

    Final validation must use full-integer TFLite/TFLite Micro.
    """

    model.eval()

    x = torch.tensor(
        normalize_states(
            states_raw
        ),
        dtype=torch.float32
    )

    with torch.no_grad():

        fp32_output = model(x)

        fp32_actions = torch.argmin(
            fp32_output,
            dim=1
        ).cpu().numpy()

    quantized_model = (
        torch.quantization
        .quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
    )

    quantized_model.eval()

    with torch.no_grad():

        int8_output = quantized_model(x)

        int8_actions = torch.argmin(
            int8_output,
            dim=1
        ).cpu().numpy()

    return np.mean(
        fp32_actions
        ==
        int8_actions
    )

def profile_model(model):

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    bias_count = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "bias" in name
    )

    fp32_bytes = (
        parameters * 4
    )

    int8_bytes = (
        parameters
        +
        bias_count * 4
    )

    macs = 0

    for module in model.modules():

        if isinstance(
            module,
            nn.Linear
        ):

            macs += (
                module.in_features
                *
                module.out_features
            )

    # Activation screening estimate.

    activation_sizes = []

    hooks = []

    def hook(module, inputs, output):

        if isinstance(
            output,
            torch.Tensor
        ):

            activation_sizes.append(
                output.numel()
            )

    for module in model.modules():

        if isinstance(
            module,
            nn.Linear
        ):

            hooks.append(
                module.register_forward_hook(
                    hook
                )
            )

    dummy = torch.zeros(
        1,
        4
    )

    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    if activation_sizes:

        peak_elements = max(
            activation_sizes
        )

    else:

        peak_elements = 0

    peak_activation_bytes = (
        peak_elements
        * 2
        * 4
    )

    return {

        "parameters":
            parameters,

        "fp32_weight_bytes":
            fp32_bytes,

        "estimated_int8_model_bytes":
            int8_bytes,

        "peak_activation_bytes":
            peak_activation_bytes,

        "macs":
            macs,

        "flops_estimate":
            macs * 2
    }

def passes_constraints(profile):

    return (
        profile[
            "estimated_int8_model_bytes"
        ]
        <=
        MODEL_FLASH_BUDGET_BYTES
        and

        profile[
            "peak_activation_bytes"
        ]
        <=
        MODEL_ACTIVATION_BUDGET_BYTES
        and

        profile["macs"]
        <=
        MODEL_MAC_BUDGET
    )

def main():

    output_dir = (
        "tinyml_search_results"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    train_states, train_objective = (
        create_dataset(
            NUM_TRAIN_SAMPLES,
            100
        )
    )

    validation_states_raw = (
        generate_contexts(
            NUM_VALIDATION_SAMPLES,
            200
        )
    )

    test_states_raw = (
        generate_contexts(
            NUM_TEST_SAMPLES,
            300
        )
    )

    results = []

    for width in HIDDEN_WIDTHS:

        for blocks in NUM_BLOCKS:

            prototype = TinyMLNetwork(
                width,
                blocks
            )

            profile = profile_model(
                prototype
            )

            if not passes_constraints(
                profile
            ):

                print(
                    f"SKIP "
                    f"W={width}, "
                    f"B={blocks}"
                )

                continue

            print(
                f"\nArchitecture "
                f"W={width}, B={blocks}"
            )

            for seed in SEEDS:

                print(
                    f"  Seed {seed}"
                )

                model = TinyMLNetwork(
                    width,
                    blocks
                )

                model = train_model(
                    model,
                    train_states,
                    train_objective,
                    seed
                )

                validation_metrics = (
                    evaluate_model(
                        model,
                        validation_states_raw
                    )
                )

                test_metrics = (
                    evaluate_model(
                        model,
                        test_states_raw
                    )
                )

                int8_agreement = (
                    evaluate_int8_screening(
                        model,
                        test_states_raw
                    )
                )

                results.append({

                    "hidden_width":
                        width,

                    "num_blocks":
                        blocks,

                    "seed":
                        seed,

                    **profile,

                    "validation_reference_agreement":
                        validation_metrics[
                            "reference_agreement"
                        ],

                    "validation_mean_regret":
                        validation_metrics[
                            "mean_regret"
