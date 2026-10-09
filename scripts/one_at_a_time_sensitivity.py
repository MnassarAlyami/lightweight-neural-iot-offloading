import numpy as np
from pathlib import Path
import csv

SEED = 42
N_SAMPLES = 50_000

rng = np.random.default_rng(SEED)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

NOMINAL = {
    "E1": 15.0,       # Partial-offload energy-cost proxy
    "E2": 25.0,       # Full-offload energy-cost proxy
    "alpha_f": 0.25,  # Feedback latency penalty
    "beta_f": 0.20,   # Feedback energy penalty
}

LATENCY_SCALE = 260.0
ENERGY_SCALE = 80.0

REALTIME_WEIGHTS = (0.75, 0.25)      # latency, energy
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

    L_net_eff = L_net * (1.0 + alpha_f * (1.0 - F))

    L0 = 2.5 * C + 50.0
    L1 = 0.5 * L_net_eff + 30.0
    L2 = L_net_eff + 10.0

    communication_penalty = 1.0 + beta_f * (1.0 - F)

    E0 = 0.8 * C
    E_partial = E1 * communication_penalty
    E_full = E2 * communication_penalty

    latency = np.column_stack([L0, L1, L2])
    energy = np.column_stack([E0, E_partial, E_full])

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

    objective = (
        w_latency[:, None] * latency_n
        + w_energy[:, None] * energy_n
    )

    return objective

J_nominal = calculate_costs(states, **NOMINAL)
nominal_actions = np.argmin(J_nominal, axis=1)

def evaluate_configuration(name, parameters):
    J_perturbed = calculate_costs(states, **parameters)
    perturbed_actions = np.argmin(J_perturbed, axis=1)

    agreement = np.mean(
        nominal_actions == perturbed_actions
    )

    action_distribution = [
        np.mean(perturbed_actions == action)
        for action in range(3)
    ]

    sample_indices = np.arange(N_SAMPLES)

    optimal_cost = J_perturbed[
        sample_indices,
        perturbed_actions
    ]

    nominal_policy_cost = J_perturbed[
        sample_indices,
        nominal_actions
    ]

    regret = nominal_policy_cost - optimal_cost

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
        "local_fraction": action_distribution[0],
        "partial_fraction": action_distribution[1],
        "full_fraction": action_distribution[2],
        "mean_regret": np.mean(regret),
        "median_regret": np.median(regret),
        "mean_normalized_regret": np.mean(normalized_regret),
        "max_regret": np.max(regret),
    }

variation_factors = [0.60, 0.80, 1.00, 1.20, 1.40]

experiments = []

experiments.append(
    ("nominal", NOMINAL.copy())
)

for parameter_name in ["E1", "E2", "alpha_f", "beta_f"]:

    for factor in variation_factors:

        if factor == 1.0:
            continue

        parameters = NOMINAL.copy()
        parameters[parameter_name] *= factor

        name = f"{parameter_name}_x{factor:.2f}"

        experiments.append(
            (name, parameters)
        )

results = []

for name, parameters in experiments:
    result = evaluate_configuration(
        name,
        parameters
    )

    results.append(result)

print("=" * 120)
print("ONE-AT-A-TIME COST-MODEL SENSITIVITY ANALYSIS")
print("=" * 120)

header = (
    f"{'Configuration':<20}"
    f"{'Agreement':>12}"
    f"{'Local':>10}"
    f"{'Partial':>10}"
    f"{'Full':>10}"
    f"{'Mean Regret':>15}"
    f"{'Norm. Regret':>15}"
)

print(header)
print("-" * 120)

for r in results:
    print(
        f"{r['configuration']:<20}"
        f"{100*r['policy_agreement']:>11.3f}%"
        f"{100*r['local_fraction']:>9.3f}%"
        f"{100*r['partial_fraction']:>9.3f}%"
        f"{100*r['full_fraction']:>9.3f}%"
        f"{r['mean_regret']:>15.8e}"
        f"{r['mean_normalized_regret']:>15.8e}"
    )

print("=" * 120)

csv_path = RESULTS_DIR / "one_at_a_time_sensitivity.csv"

fieldnames = list(results[0].keys())

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to: {csv_path}")

print("\nNominal policy distribution:")

action_names = ["Local", "Partial", "Full"]

for action, action_name in enumerate(action_names):
    count = np.sum(nominal_actions == action)

    print(
        f"{action_name:<8}: "
        f"{count:>6} / {N_SAMPLES} "
        f"({100*count/N_SAMPLES:.3f}%)"
    )
