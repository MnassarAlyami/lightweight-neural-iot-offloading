import numpy as np
import csv
from pathlib import Path

SEED = 42
N_SAMPLES = 50_000

rng = np.random.default_rng(SEED)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

E1 = 15.0
E2 = 25.0
ALPHA_F = 0.25
BETA_F = 0.20

LATENCY_SCALE = 260.0
ENERGY_SCALE = 80.0

NOMINAL_RT = (0.75, 0.25)
NOMINAL_DT = (0.50, 0.50)

cpu = rng.uniform(0.0, 100.0, N_SAMPLES)
network_latency = rng.uniform(0.0, 250.0, N_SAMPLES)
task_type = rng.integers(0, 2, N_SAMPLES)
feedback = rng.integers(0, 2, N_SAMPLES)

states = np.column_stack([
    cpu,
    network_latency,
    task_type,
    feedback
])

def base_costs(states):
    C = states[:, 0]
    L_net = states[:, 1]
    F = states[:, 3]

    L_net_eff = L_net * (
        1.0 + ALPHA_F * (1.0 - F)
    )

    L0 = 2.5 * C + 50.0
    L1 = 0.5 * L_net_eff + 30.0
    L2 = L_net_eff + 10.0

    communication_penalty = (
        1.0 + BETA_F * (1.0 - F)
    )

    E0 = 0.8 * C
    E_partial = E1 * communication_penalty
    E_full = E2 * communication_penalty

    latency = np.column_stack([L0, L1, L2])
    energy = np.column_stack([E0, E_partial, E_full])

    return (
        latency / LATENCY_SCALE,
        energy / ENERGY_SCALE
    )

latency_n, energy_n = base_costs(states)

def objective(rt_weights, dt_weights):
    T = states[:, 2].astype(int)

    w_latency = np.where(
        T == 1,
        rt_weights[0],
        dt_weights[0]
    )

    w_energy = np.where(
        T == 1,
        rt_weights[1],
        dt_weights[1]
    )

    return (
        w_latency[:, None] * latency_n
        + w_energy[:, None] * energy_n
    )

J_nominal = objective(
    NOMINAL_RT,
    NOMINAL_DT
)

nominal_actions = np.argmin(
    J_nominal,
    axis=1
)

def evaluate(name, rt_weights, dt_weights):

    J = objective(
        rt_weights,
        dt_weights
    )

    actions = np.argmin(J, axis=1)

    idx = np.arange(N_SAMPLES)

    optimal_cost = J[idx, actions]
    nominal_policy_cost = J[idx, nominal_actions]

    regret = nominal_policy_cost - optimal_cost

    normalized_regret = regret / (
        optimal_cost + 1e-12
    )

    return {
        "configuration": name,

        "rt_latency_weight": rt_weights[0],
        "rt_energy_weight": rt_weights[1],

        "dt_latency_weight": dt_weights[0],
        "dt_energy_weight": dt_weights[1],

        "policy_agreement":
            np.mean(actions == nominal_actions),

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

# All weight pairs sum to 1; scenarios test local and wider sensitivity.
experiments = [
    (
        "nominal",
        (0.75, 0.25),
        (0.50, 0.50)
    ),

    (
        "RT_70_30",
        (0.70, 0.30),
        (0.50, 0.50)
    ),
    (
        "RT_80_20",
        (0.80, 0.20),
        (0.50, 0.50)
    ),
    (
        "RT_65_35",
        (0.65, 0.35),
        (0.50, 0.50)
    ),
    (
        "RT_85_15",
        (0.85, 0.15),
        (0.50, 0.50)
    ),

    (
        "DT_45_55",
        (0.75, 0.25),
        (0.45, 0.55)
    ),
    (
        "DT_55_45",
        (0.75, 0.25),
        (0.55, 0.45)
    ),
    (
        "DT_40_60",
        (0.75, 0.25),
        (0.40, 0.60)
    ),
    (
        "DT_60_40",
        (0.75, 0.25),
        (0.60, 0.40)
    ),

    (
        "latency_emphasis",
        (0.85, 0.15),
        (0.60, 0.40)
    ),
    (
        "energy_emphasis",
        (0.65, 0.35),
        (0.40, 0.60)
    ),
]

results = [
    evaluate(name, rt, dt)
    for name, rt, dt in experiments
]

print("=" * 125)
print("OBJECTIVE-WEIGHT SENSITIVITY ANALYSIS")
print("=" * 125)

print(
    f"{'Configuration':<22}"
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
        f"{r['configuration']:<22}"
        f"{100*r['policy_agreement']:>11.3f}%"
        f"{100*r['local_fraction']:>9.3f}%"
        f"{100*r['partial_fraction']:>9.3f}%"
        f"{100*r['full_fraction']:>9.3f}%"
        f"{r['mean_regret']:>15.8e}"
        f"{r['mean_normalized_regret']:>15.8e}"
    )

print("=" * 125)

csv_path = RESULTS_DIR / "weight_sensitivity.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved results to: {csv_path}")
