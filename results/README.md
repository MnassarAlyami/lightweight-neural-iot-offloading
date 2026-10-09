# Results

This directory contains the frozen numerical outputs used to reproduce the main reported analyses.

## Architecture search

- `architecture_seed_results.csv`: 525 architecture-seed runs (105 architectures × 5 seeds).
- `architecture_summary.csv`: architecture-level aggregation with the complete final qualification criteria.
- `architecture_shortlist.csv`: 42 architectures satisfying the final resource, agreement, and stability criteria.

## Sensitivity analysis

- `one_at_a_time_sensitivity.csv`
- `combined_sensitivity.csv`
- `weight_sensitivity.csv`

The sensitivity perturbations are systematic robustness scenarios, not empirical confidence intervals.

The 7,500-state held-out arrays are deterministic outputs of `scripts/train_offloading_model.py` and are not committed separately. Their verified hashes are recorded in `ARTIFACTS.md`.
