# A Lightweight Context-Aware Neural Offloading Strategy for IoT Edge Devices

This repository contains the training, conversion, evaluation, and proof-of-concept deployment code associated with the manuscript **“A Lightweight Context-Aware Neural Offloading Strategy for IoT Edge Devices.”**

## Overview

The method treats each task-offloading decision as a context-dependent cost-minimization problem over three execution options: local execution, partial offloading, and full offloading. The system context contains CPU utilization, network latency, task type, and network-condition feedback. Analytical latency and modeled energy-cost functions define the reference action costs, and a compact neural network learns to approximate those costs.

The selected network has a `4 -> 16 -> 16 -> 3` topology with 403 trainable parameters and 368 MACs per inference.

After full INT8 quantization, the frozen deployment model occupies 2,776 bytes (2.71 KiB). On 7,500 held-out states, the INT8 policy achieves 99.17% action agreement with the analytical reference policy.

## Repository structure

```text
scripts/
  train_offloading_model.py
  export_validate_fp32_tflite.py
  export_validate_int8_tflite.py
  generate_deployment_vectors.py
  generate_near_boundary_vectors.py
  generate_final_paper_figures.py
  generate_model_header.py

deployment/esp32/
  sketch.ino
  diagram.json
  deployment_vectors.h
  boundary_vectors.h
  near_boundary_vectors.h
  WOKWI_PROJECT.txt

models/
  offloading_int8.tflite

results/
  README.md
```

The firmware includes `offloading_model.h` at compile time. Generate this header from the frozen INT8 model using `scripts/generate_model_header.py`; the generated header itself is not stored in the repository.

## Environment

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Run the Python scripts from the repository root so that generated files are written to `outputs/`.

## Reproducing the model pipeline

1. Train the selected network:

```bash
python scripts/train_offloading_model.py
```

2. Export and validate the FP32 TensorFlow Lite model:

```bash
python scripts/export_validate_fp32_tflite.py
```

3. Export and validate the fully quantized INT8 model:

```bash
python scripts/export_validate_int8_tflite.py
```

4. Generate the general deployment vectors and cases close to an INT8 decision tie:

```bash
python scripts/generate_deployment_vectors.py
python scripts/generate_near_boundary_vectors.py
```

The exact-tie validation header used in the proof-of-concept firmware is included under `deployment/esp32/`. Its original generator was not part of the supplied project files.

5. Generate the C header from the frozen INT8 model before compiling the ESP32 firmware:

```bash
python scripts/generate_model_header.py
```

## Main reported results

| Characteristic | Result |
|---|---:|
| Selected architecture | `4 -> 16 -> 16 -> 3` |
| Trainable parameters | 403 |
| MACs per inference | 368 |
| INT8 model size | 2,776 B (2.71 KiB) |
| INT8/reference-policy agreement | 99.17% |
| Practical tensor arena | 2 KiB |
| Simulated ESP32 model inference | ~0.854 ms |
| Direct analytical decision | ~31.81 us |

The modeled energy values are analytical energy-cost proxies rather than physical energy measurements. The ESP32 timing results are from a simulated proof-of-concept environment and should not be interpreted as physical-hardware power, energy, or end-to-end communication measurements.

## Figures and sensitivity results

`scripts/generate_final_paper_figures.py` reproduces the main evaluation figures when the architecture-search and sensitivity CSV files are available. See `results/README.md` for the required filenames.

## Frozen deployment artifact

The exact INT8 model used in the proof-of-concept firmware is included in `models/`. See `ARTIFACTS.md` for its SHA-256 hash and artifact notes.

## License

This project is released under the MIT License.
