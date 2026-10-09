import numpy as np
from pathlib import Path
import csv

SEED = 42
N_SAMPLES = 50_000

rng = np.random.default_rng(SEED)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

NOMINAL = {
    "E1": 15.0,
    "E2": 25.0,
    "alpha_f": 0.25,
    "beta_f": 0.20,
}

LATENCY_SCALE = 260.0
ENERGY_SCALE = 80.0

REALTIME_WEIGHTS = (0.75, 0.25)
DELAY_TOLERANT_WEIGHTS = (0.50, 0.50)

cpu = rng.uniform(0.0, 100.0, N_SAMPLES)
network_latency = rng.uniform(0.0, 250.0, N_SAMPLES)
task_type = rng.integers(0, 2, N_SAMPLES)
feedback = rng.integers(0, 2, N_SAMPLES)

states = np.column_stack(
    [cpu, network_latency, task_type, feedback]
)

def calculate_costs(states, E1, E2, alpha_f, beta_f):

    C = states[:, 0]
    L_net = states[:, 1]
    T = states[:, 2].astype(int)
    F = states[:, 3]

    L_net_eff = L_net * (
        1.0 + alpha_f * (1.0 - F)
    )

    L0 = 2.5 * C + 50.0
    L1 = 0.5 * L_net_eff + 30.0
    L2 = L_net_eff + 10.0

    communication_penalty = (
        1.0 + beta_f * (1.0 - F)
    )

    E0 = 0.8 * C
    E_partial = E1 * communication_penalty
    E_full = E2 * communication_penalty

    latency = np.column_stack(
        [L0, L1, L2]
    )

    energy = np.column_stack(
        [E0, E_partial, E_full]
    )

    latency_n = latency / LATENCY_SCALE
    energy_n = energy / ENERGY_SCALE

    w_latency = np.where(
        T == 1,
        REALTIME_WEIGHTS[0],
        DELAY_TOLERANT_WEIGHTS[0],
    )

    w_energy = np.where(
        T == 1,
        REALTIME_WEIGHTS[1],
        DELAY_TOLERANT_WEIGHTS[1],
    )

    J = (
        w_latency[:, None] * latency_n
        + w_energy[:, None] * energy_n
    )

    return J

J_nominal = calculate_costs(
    states,
    **NOMINAL
)

nominal_actions = np.argmin(
    J_nominal,
    axis=1
)

def evaluate_configuration(name, parameters):

    J = calculate_costs(
        states,
        **parameters
    )

    actions = np.argmin(J, axis=1)

    agreement = np.mean(
        actions == nominal_actions
    )

    sample_indices = np.arange(N_SAMPLES)

    optimal_cost = J[
        sample_indices,
        actions
    ]

    nominal_policy_cost = J[
        sample_indices,
        nominal_actions
    ]

    regret = (
        nominal_policy_cost
        - optimal_cost
    )

    normalized_regret = regret / (
        optimal_cost + 1e-12
    )

    return {
        "configuration": name,

        "E1": parameters["E1"],
        "E2": parameters["E2"],
        "alpha_f": parameters["alpha_f"],
        "beta_f": parameters["beta_f"],

        "policy_agreement": agreement,

        "local_fraction":
            np.mean(actions == 0),

        "partial_fraction":
            np.mean(actions == 1),

        "full_fraction":
            np.mean(actions == 2),

        "mean_regret":
            np.mean(regret),

        "median_regret":
            np.median(regret),

        "mean_normalized_regret":
            np.mean(normalized_regret),

        "max_regret":
            np.max(regret),
    }

def scaled_parameters(
    E1_factor=1.0,
    E2_factor=1.0,
    alpha_factor=1.0,
    beta_factor=1.0,
):

    return {
        "E1":
            NOMINAL["E1"] * E1_factor,

        "E2":
            NOMINAL["E2"] * E2_factor,

        "alpha_f":
            NOMINAL["alpha_f"] * alpha_factor,

        "beta_f":
            NOMINAL["beta_f"] * beta_factor,
    }

# Structured perturbation scenarios; not empirical confidence intervals.
experiments = [
    (
        "nominal",
        scaled_parameters()
    ),

    (
        "all_-20%",
        scaled_parameters(
            0.8, 0.8, 0.8, 0.8
        )
    ),

    (
        "all_+20%",
        scaled_parameters(
            1.2, 1.2, 1.2, 1.2
        )
    ),

    (
        "all_-40%",
        scaled_parameters(
            0.6, 0.6, 0.6, 0.6
        )
    ),

    (
        "all_+40%",
        scaled_parameters(
            1.4, 1.4, 1.4, 1.4
        )
    ),

    (
        "E1_low_E2_high_20",
        scaled_parameters(
            E1_factor=0.8,
            E2_factor=1.2
        )
    ),

    (
        "E1_high_E2_low_20",
        scaled_parameters(
            E1_factor=1.2,
            E2_factor=0.8
        )
    ),

    (
        "E1_low_E2_high_40",
        scaled_parameters(
            E1_factor=0.6,
            E2_factor=1.4
        )
    ),

    (
        "E1_high_E2_low_40",
        scaled_parameters(
            E1_factor=1.4,
            E2_factor=0.6
        )
    ),

    (
        "feedback_-40%",
        scaled_parameters(
            alpha_factor=0.6,
            beta_factor=0.6
        )
    ),

    (
        "feedback_+40%",
        scaled_parameters(
            alpha_factor=1.4,
            beta_factor=1.4
        )
    ),

    (
        "alpha_low_beta_high",
        scaled_parameters(
            alpha_factor=0.6,
            beta_factor=1.4
        )
    ),

    (
        "alpha_high_beta_low",
        scaled_parameters(
            alpha_factor=1.4,
            beta_factor=0.6
        )
    ),
]

results = []

for name, parameters in experiments:

    result = evaluate_configuration(
        name,
        parameters
    )

    results.append(result)

print("=" * 125)
print(
    "COMBINED COST-MODEL "
    "SENSITIVITY ANALYSIS"
)
print("=" * 125)

print(
    f"{'Configuration':<25}"
    f"{'Agreement':>12}"
    f"{'Local':>10}"
    f"{'Partial':>10}"
    f"{'Full':>10}"
    f"{'Mean Regret':>15}"
    f"{'Norm. Regret':>15}"
)

print("-" * 125)

for r in results:

    print(
        f"{r['configuration']:<25}"
        f"{100*r['policy_agreement']:>11.3f}%"
        f"{100*r['local_fraction']:>9.3f}%"
        f"{100*r['partial_fraction']:>9.3f}%"
        f"{100*r['full_fraction']:>9.3f}%"
        f"{r['mean_regret']:>15.8e}"
        f"{r['mean_normalized_regret']:>15.8e}"
    )

print("=" * 125)

csv_path = (
    RESULTS_DIR
    / "combined_sensitivity.csv"
)

fieldnames = list(
    results[0].keys()
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

print(
    f"\nSaved results to: {csv_path}"
)
