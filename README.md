# A Lightweight Context-Aware Neural Offloading Strategy for IoT Edge Devices

This repository contains the code, frozen model artifacts, evaluation data, and simulated ESP32 proof-of-concept files associated with the manuscript **“A Lightweight Context-Aware Neural Offloading Strategy for IoT Edge Devices.”**

## Overview

The method treats task offloading as a context-dependent cost-minimization problem over three execution options: local execution, partial offloading, and full offloading. The input context contains CPU utilization, network latency, task type, and network-condition feedback. Analytical latency and modeled energy-cost functions define the reference action costs, and a compact neural network learns to estimate those costs.

The selected network is `4 -> 16 -> 16 -> 3`, with 403 trainable parameters and 368 MACs per inference. The final fully quantized INT8 model occupies 2,776 bytes (2.71 KiB) and achieves 99.17% action agreement with the analytical reference policy on 7,500 held-out states.

## Repository structure

```text
scripts/
  architecture_search.py
  train_offloading_model.py
  export_validate_fp32_tflite.py
  export_validate_int8_tflite.py
  one_at_a_time_sensitivity.py
  combined_sensitivity.py
  weight_sensitivity.py
  generate_deployment_vectors.py
  generate_near_boundary_vectors.py
  generate_final_paper_figures.py
  generate_model_header.py

models/
  offloading_fp32.tflite
  offloading_int8.tflite

results/
  architecture_seed_results.csv
  architecture_summary.csv
  architecture_shortlist.csv
  one_at_a_time_sensitivity.csv
  combined_sensitivity.csv
  weight_sensitivity.csv

deployment/esp32/
  sketch.ino
  diagram.json
  deployment_vectors.h
  boundary_vectors.h
  near_boundary_vectors.h
  WOKWI_PROJECT.txt
```

## Environment

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

See `ENVIRONMENT.md` for the environment metadata recoverable from the frozen artifacts and for commands to capture the original package versions if the original environment is still available.

## Main workflow

Run scripts from the repository root.

Architecture search:

```bash
python scripts/architecture_search.py
```

Train the selected deployment model:

```bash
python scripts/train_offloading_model.py
```

Export and validate FP32 and INT8 TensorFlow Lite models:

```bash
python scripts/export_validate_fp32_tflite.py
python scripts/export_validate_int8_tflite.py
```

Run sensitivity analyses:

```bash
python scripts/one_at_a_time_sensitivity.py
python scripts/combined_sensitivity.py
python scripts/weight_sensitivity.py
```

Generate deployment vectors:

```bash
python scripts/generate_deployment_vectors.py
python scripts/generate_near_boundary_vectors.py
```

The exact INT8 tie cases used in the proof-of-concept evaluation are included as `deployment/esp32/boundary_vectors.h`. The supplied `generate_boundary_vectors.ipynb` generates the non-tied near-boundary set, so the original generator for the exact-tie set is not currently available in the repository.

Generate the C header for the frozen INT8 model before compiling the ESP32 firmware:

```bash
python scripts/generate_model_header.py
```

Generate the main evaluation figures:

```bash
python scripts/generate_final_paper_figures.py
```

## Main reported deployment results

| Characteristic | Result |
|---|---:|
| Selected architecture | `4 -> 16 -> 16 -> 3` |
| Trainable parameters | 403 |
| MACs per inference | 368 |
| INT8 model size | 2,776 B (2.71 KiB) |
| INT8/reference-policy agreement | 99.17% |
| Minimum successful tensor arena tested | 1,680 B |
| Practical tensor arena | 2 KiB |
| Simulated ESP32 model inference | ~0.854 ms |
| Direct analytical decision | ~31.81 us |

For the current three-action formulation, direct analytical evaluation is faster than neural inference. The neural model is evaluated as a compact learned representation of the analytical policy rather than as a faster substitute for the underlying equations.

The modeled energy values are analytical energy-cost proxies, not physical energy measurements. The ESP32 measurements are from a simulated proof-of-concept environment and should not be interpreted as physical-device power, energy, or end-to-end communication measurements.

## Reproducibility notes

See `REPRODUCIBILITY_NOTES.md` for the architecture-search screening correction and other methodological details relevant to reproducing the final manuscript results.

## License

This project is released under the MIT License.