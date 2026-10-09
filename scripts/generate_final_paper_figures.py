import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARCH_PATH = "results/architecture_seed_results.csv"
COMBINED_PATH = "results/combined_sensitivity.csv"
WEIGHT_PATH = "results/weight_sensitivity.csv"

X_TEST_PATH = "outputs/x_test.npy"
Y_TEST_PATH = "outputs/y_test.npy"
ORACLE_ACTIONS_PATH = "outputs/oracle_actions.npy"
INT8_ACTIONS_PATH = "outputs/int8_actions.npy"

OUT = "final_paper_figures"
os.makedirs(OUT, exist_ok=True)

ALPHA_F = 0.25
BETA_F = 0.20

LATENCY_SCALE = 260.0
ENERGY_SCALE = 80.0

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

def save_figure(fig, filename):
    """Save each figure as vector PDF and 300-dpi PNG."""

    fig.savefig(
        os.path.join(OUT, filename + ".pdf"),
        bbox_inches="tight"
    )

    fig.savefig(
        os.path.join(OUT, filename + ".png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

df = pd.read_csv(ARCH_PATH)

arch = (
    df.groupby(["hidden_width", "num_blocks"])
      .agg(
          parameters=("parameters", "first"),
          macs=("macs", "first"),
          model_bytes=("estimated_int8_model_bytes", "first"),
          peak_activation_bytes=("peak_activation_bytes", "first"),

          agreement_mean=("test_reference_agreement", "mean"),
          agreement_sd=("test_reference_agreement", "std"),
          agreement_min=("test_reference_agreement", "min"),

          int8_agreement_mean=("fp32_int8_action_agreement", "mean"),
          int8_agreement_sd=("fp32_int8_action_agreement", "std"),

          regret_mean=("mean_regret", "mean")
      )
      .reset_index()
)

arch["qualifies"] = (
    (arch["model_bytes"] <= 256 * 1024) &
    (arch["peak_activation_bytes"] <= 64 * 1024) &
    (arch["macs"] <= 1_000_000) &
    (arch["agreement_mean"] >= 0.99) &
    (arch["int8_agreement_mean"] >= 0.99) &
    (arch["agreement_sd"] <= 0.005) &
    (arch["int8_agreement_sd"] <= 0.005)
)

print("=" * 70)
print("ARCHITECTURE SEARCH")
print("=" * 70)
print("Architectures evaluated:", len(arch))
print(
    "Architectures satisfying all criteria:",
    int(arch["qualifies"].sum())
)

selected = arch[
    (arch["hidden_width"] == 16) &
    (arch["num_blocks"] == 1)
].iloc[0]

print()
print("Selected architecture: 4 -> 16 -> 16 -> 3")
print("Parameters:", int(selected["parameters"]))
print("MACs:", int(selected["macs"]))
print(
    "Mean reference agreement:",
    f"{selected['agreement_mean'] * 100:.4f}%"
)
print(
    "Mean INT8 agreement:",
    f"{selected['int8_agreement_mean'] * 100:.4f}%"
)

failed = arch[~arch["qualifies"]]
passed = arch[arch["qualifies"]]

fig, ax = plt.subplots(figsize=(7.0, 4.3))

ax.scatter(
    failed["parameters"],
    failed["agreement_mean"] * 100,
    s=28,
    alpha=0.55,
    label="Does not satisfy all criteria"
)

ax.scatter(
    passed["parameters"],
    passed["agreement_mean"] * 100,
    s=34,
    alpha=0.8,
    label="Satisfies all criteria"
)

ax.scatter(
    selected["parameters"],
    selected["agreement_mean"] * 100,
    s=120,
    marker="*",
    zorder=5,
    label=r"Selected: $4\!\rightarrow\!16\!\rightarrow\!16\!\rightarrow\!3$"
)

ax.axhline(
    99.0,
    linestyle="--",
    linewidth=1.0,
    label="99% reference-agreement threshold"
)

ax.annotate(
    "Selected\n(403 parameters)",
    xy=(
        selected["parameters"],
        selected["agreement_mean"] * 100
    ),
    xytext=(35, -42),
    textcoords="offset points",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.8
    )
)

ax.set_xscale("log")

ax.set_xlabel("Trainable Parameters")
ax.set_ylabel("Mean Reference-Policy Agreement (%)")

ax.grid(
    alpha=0.22,
    which="both"
)

ax.legend(
    frameon=False,
    loc="lower right"
)

fig.tight_layout()

save_figure(
    fig,
    "Fig2_architecture_search"
)

x_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

oracle_actions = np.load(
    ORACLE_ACTIONS_PATH
).astype(int)

int8_actions = np.load(
    INT8_ACTIONS_PATH
).astype(int)

N = len(x_test)

assert x_test.shape == (N, 4)
assert y_test.shape == (N, 3)
assert len(oracle_actions) == N
assert len(int8_actions) == N

print()
print("=" * 70)
print("POLICY EVALUATION")
print("=" * 70)
print("Held-out samples:", N)

cpu = x_test[:, 0] * 100.0
network_latency = x_test[:, 1] * 250.0

task_type = np.rint(
    x_test[:, 2]
).astype(int)

feedback = np.rint(
    x_test[:, 3]
).astype(int)

effective_latency = (
    network_latency *
    (1.0 + ALPHA_F * (1 - feedback))
)

communication_penalty = (
    1.0 +
    BETA_F * (1 - feedback)
)

latency_matrix = np.column_stack([
    2.5 * cpu + 50.0,
    0.5 * effective_latency + 30.0,
    effective_latency + 10.0
])

energy_matrix = np.column_stack([
    0.8 * cpu,
    15.0 * communication_penalty,
    25.0 * communication_penalty
])

w_latency = np.where(
    task_type == 1,
    0.75,
    0.50
)

w_energy = np.where(
    task_type == 1,
    0.25,
    0.50
)

objective_matrix = (
    w_latency[:, None] *
    (latency_matrix / LATENCY_SCALE)
    +
    w_energy[:, None] *
    (energy_matrix / ENERGY_SCALE)
)

recomputed_oracle = np.argmin(
    objective_matrix,
    axis=1
)

oracle_agreement = np.mean(
    recomputed_oracle == oracle_actions
)

print(
    "Recomputed/saved oracle agreement:",
    f"{oracle_agreement * 100:.6f}%"
)

if oracle_agreement < 0.999999:
    raise RuntimeError(
        "Oracle mismatch detected. "
        "Do not generate the final figures."
    )

baseline_actions = np.where(
    cpu > 70.0,
    2,
    0
).astype(int)

rows = np.arange(N)

def select_values(matrix, actions):
    return matrix[rows, actions]

policies = {
    "Analytical Reference": oracle_actions,
    "INT8 Neural Policy": int8_actions,
    "Static Baseline": baseline_actions
}

results = {}

oracle_objective = select_values(
    objective_matrix,
    oracle_actions
)

for name, actions in policies.items():

    latency = select_values(
        latency_matrix,
        actions
    )

    energy = select_values(
        energy_matrix,
        actions
    )

    objective = select_values(
        objective_matrix,
        actions
    )

    regret = objective - oracle_objective

    results[name] = {
        "actions": actions,
        "latency": latency,
        "energy": energy,
        "objective": objective,
        "regret": regret
    }

summary_rows = []

for name, r in results.items():

    actions = r["actions"]

    summary_rows.append({
        "Policy": name,

        "Mean Latency (ms)":
            np.mean(r["latency"]),

        "Median Latency (ms)":
            np.median(r["latency"]),

        "Mean Modeled Energy Cost":
            np.mean(r["energy"]),

        "Median Modeled Energy Cost":
            np.median(r["energy"]),

        "Mean Objective":
            np.mean(r["objective"]),

        "Mean Regret":
            np.mean(r["regret"]),

        "Local (%)":
            np.mean(actions == 0) * 100,

        "Partial (%)":
            np.mean(actions == 1) * 100,

        "Full (%)":
            np.mean(actions == 2) * 100
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    os.path.join(
        OUT,
        "final_policy_comparison.csv"
    ),
    index=False
)

print()
print(summary_df.to_string(index=False))

policy_names = list(policies.keys())

short_labels = [
    "Analytical\nReference",
    "INT8 Neural\nPolicy",
    "Static\nBaseline"
]

latency_data = [
    results[p]["latency"]
    for p in policy_names
]

energy_data = [
    results[p]["energy"]
    for p in policy_names
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 3.35)
)

axes[0].boxplot(
    latency_data,
    labels=short_labels,
    showfliers=False
)

axes[0].set_ylabel("Latency (ms)")

axes[0].grid(
    axis="y",
    alpha=0.22
)

axes[0].text(
    0.02,
    0.97,
    "(a)",
    transform=axes[0].transAxes,
    ha="left",
    va="top",
    fontweight="bold"
)

axes[1].boxplot(
    energy_data,
    labels=short_labels,
    showfliers=False
)

axes[1].set_ylabel("Modeled Energy-Cost Proxy")

axes[1].grid(
    axis="y",
    alpha=0.22
)

axes[1].text(
    0.02,
    0.97,
    "(b)",
    transform=axes[1].transAxes,
    ha="left",
    va="top",
    fontweight="bold"
)

fig.tight_layout()

save_figure(
    fig,
    "Fig3_policy_performance"
)

action_names = [
    "Local",
    "Partial",
    "Full"
]

action_percentages = np.array([
    [
        np.mean(policies[p] == action) * 100
        for action in range(3)
    ]
    for p in policy_names
])

x = np.arange(
    len(action_names)
)

width = 0.24

fig, ax = plt.subplots(
    figsize=(6.8, 3.8)
)

for i, policy_name in enumerate(policy_names):

    offset = (
        i -
        (len(policy_names) - 1) / 2
    ) * width

    bars = ax.bar(
        x + offset,
        action_percentages[i],
        width,
        label=policy_name
    )

    for bar, value in zip(
        bars,
        action_percentages[i]
    ):
        ax.text(
            bar.get_x() +
            bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

ax.set_xticks(x)
ax.set_xticklabels(action_names)

ax.set_ylabel("Action Selection (%)")

ax.set_ylim(0, 90)

ax.grid(
    axis="y",
    alpha=0.22
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center"
)

fig.tight_layout()

save_figure(
    fig,
    "Fig4_action_distribution"
)

combined = pd.read_csv(
    COMBINED_PATH
)

combined_order = [
    "nominal",
    "all_-20%",
    "all_+20%",
    "all_-40%",
    "all_+40%",
    "feedback_-40%",
    "feedback_+40%",
    "E1_low_E2_high_20",
    "E1_high_E2_low_20",
    "E1_low_E2_high_40",
    "E1_high_E2_low_40"
]

cdf = combined[
    combined["configuration"].isin(
        combined_order
    )
].copy()

cdf["configuration"] = pd.Categorical(
    cdf["configuration"],
    categories=combined_order,
    ordered=True
)

cdf = cdf.sort_values(
    "configuration"
)

combined_labels = {
    "nominal": "Nominal",

    "all_-20%":
        "All\n-20%",

    "all_+20%":
        "All\n+20%",

    "all_-40%":
        "All\n-40%",

    "all_+40%":
        "All\n+40%",

    "feedback_-40%":
        "Feedback\n-40%",

    "feedback_+40%":
        "Feedback\n+40%",

    "E1_low_E2_high_20":
        "$E_1$ -20%\n$E_2$ +20%",

    "E1_high_E2_low_20":
        "$E_1$ +20%\n$E_2$ -20%",

    "E1_low_E2_high_40":
        "$E_1$ -40%\n$E_2$ +40%",

    "E1_high_E2_low_40":
        "$E_1$ +40%\n$E_2$ -40%"
}

labels = [
    combined_labels[x]
    for x in cdf[
        "configuration"
    ].astype(str)
]

x = np.arange(len(cdf))

agreement = (
    cdf["policy_agreement"].to_numpy()
    * 100
)

regret_pct = (
    cdf[
        "mean_normalized_regret"
    ].to_numpy()
    * 100
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.2, 5.8),
    sharex=True
)

axes[0].bar(
    x,
    agreement
)

axes[0].set_ylabel(
    "Policy Agreement (%)"
)

axes[0].set_ylim(
    max(70, agreement.min() - 4),
    101
)

axes[0].grid(
    axis="y",
    alpha=0.22
)

axes[0].text(
    0.01,
    0.95,
    "(a)",
    transform=axes[0].transAxes,
    ha="left",
    va="top",
    fontweight="bold"
)

axes[1].bar(
    x,
    regret_pct
)

axes[1].set_ylabel(
    "Mean Normalized Regret (%)"
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    labels,
    rotation=0
)

axes[1].grid(
    axis="y",
    alpha=0.22
)

axes[1].text(
    0.01,
    0.95,
    "(b)",
    transform=axes[1].transAxes,
    ha="left",
    va="top",
    fontweight="bold"
)

fig.tight_layout()

save_figure(
    fig,
    "Fig5_cost_model_sensitivity"
)

weights = pd.read_csv(
    WEIGHT_PATH
)

weight_order = [
    "nominal",
    "RT_70_30",
    "RT_80_20",
    "RT_65_35",
    "RT_85_15",
    "DT_45_55",
    "DT_55_45",
    "DT_40_60",
    "DT_60_40",
    "latency_emphasis",
    "energy_emphasis"
]

wdf = weights[
    weights["configuration"].isin(
        weight_order
    )
].copy()

wdf["configuration"] = pd.Categorical(
    wdf["configuration"],
    categories=weight_order,
    ordered=True
)

wdf = wdf.sort_values(
    "configuration"
)

weight_labels = {
    "nominal": "Nominal",
    "RT_70_30": "RT\n70/30",
    "RT_80_20": "RT\n80/20",
    "RT_65_35": "RT\n65/35",
    "RT_85_15": "RT\n85/15",
    "DT_45_55": "DT\n45/55",
    "DT_55_45": "DT\n55/45",
    "DT_40_60": "DT\n40/60",
    "DT_60_40": "DT\n60/40",
    "latency_emphasis":
        "Latency\nemphasis",
    "energy_emphasis":
        "Energy\nemphasis"
}

labels = [
    weight_labels[x]
    for x in wdf[
        "configuration"
    ].astype(str)
]

x = np.arange(len(wdf))

agreement = (
    wdf[
        "policy_agreement"
    ].to_numpy()
    * 100
)

regret_pct = (
    wdf[
        "mean_normalized_regret"
    ].to_numpy()
    * 100
)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.2, 5.8),
    sharex=True
)

axes[0].bar(
    x,
    agreement
)

axes[0].set_ylabel(
    "Policy Agreement (%)"
)

axes[0].set_ylim(
    max(90, agreement.min() - 2),
    101
)

axes[0].grid(
    axis="y",
    alpha=0.22
)

axes[0].text(
    0.01,
    0.95,
    "(a)",
    transform=axes[0].transAxes,
    ha="left",
    va="top",
    fontweight="bold"
)

axes[1].bar(
    x,
    regret_pct
)

axes[1].set_ylabel(
    "Mean Normalized Regret (%)"
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(
    labels
)

axes[1].grid(
    axis="y",
    alpha=0.22
)

axes[1].text(
    0.01,
    0.95,
    "(b)",
    transform=axes[1].transAxes,
    ha="left",
    va="top",
    fontweight="bold"
)

fig.tight_layout()

save_figure(
    fig,
    "Fig6_objective_weight_sensitivity"
)

int8 = results[
    "INT8 Neural Policy"
]

baseline = results[
    "Static Baseline"
]

latency_reduction = (
    1 -
    np.mean(int8["latency"]) /
    np.mean(baseline["latency"])
) * 100

energy_reduction = (
    1 -
    np.mean(int8["energy"]) /
    np.mean(baseline["energy"])
) * 100

objective_reduction = (
    1 -
    np.mean(int8["objective"]) /
    np.mean(baseline["objective"])
) * 100

print()
print("=" * 70)
print("INT8 POLICY VS STATIC BASELINE")
print("=" * 70)

print(
    "Mean latency reduction:",
    f"{latency_reduction:.3f}%"
)

print(
    "Mean modeled energy-cost reduction:",
    f"{energy_reduction:.3f}%"
)

print(
    "Mean objective reduction:",
    f"{objective_reduction:.3f}%"
)

print()
print(
    "All final figures saved to:",
    os.path.abspath(OUT)
)
