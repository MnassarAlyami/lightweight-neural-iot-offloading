# Reproducibility notes

## Architecture-search screening

The supplied architecture-search source originally created its shortlist using only mean FP32 reference agreement and mean FP32-to-INT8 agreement. The final manuscript criteria also require standard deviations no greater than 0.5 percentage points for both agreement measures, in addition to the resource envelope. Applying the complete final criteria to the supplied 525 seed-level runs yields 42 qualifying architectures, rather than 45.

The repository version of `scripts/architecture_search.py` applies the complete final criteria and uses parameter count, then MACs, then mean regret as the selection order. This change aligns the script with the finalized methodology; it does not alter the supplied seed-level experimental results.

The architecture-search INT8 metric is a screening metric based on PyTorch dynamic quantization. Final deployment quality is evaluated separately with the fully integer TensorFlow Lite model.

## Standalone deployment model

The final standalone model uses TensorFlow/Keras and is evaluated on the supplied 7,500-state held-out set. The frozen FP32 Keras and FP32 TFLite artifacts are included alongside the frozen INT8 deployment model.

## Sensitivity analysis

The one-at-a-time, combined, and objective-weight sensitivity scripts use a fixed 50,000-state population with seed 42. Their perturbations are systematic robustness scenarios, not empirical confidence intervals.

## ESP32 proof of concept

The Wokwi/ESP32 measurements validate firmware integration, numerical consistency, tensor-arena allocation, and model-inference timing in a simulated environment. They are not physical-device power or end-to-end communication measurements.


## Exact INT8 tie-case generator

The original source used to generate the frozen 30 exact INT8 tie cases was not recovered. The repository therefore includes `scripts/generate_exact_tie_vectors.py`, reconstructed from the supplied near-boundary generator and the frozen `boundary_vectors.h`.

The reconstructed script selects held-out samples for which the two smallest raw INT8 outputs are equal (decision margin 0) and takes the first 30 cases in held-out-set order. This matches the documented frozen exact-tie set. The frozen header remains the authoritative artifact for the reported ESP32 consistency experiment.
