# A Lightweight Context-Aware Neural Offloading Strategy for IoT Edge Devices

This repository contains the code and deployment artifacts associated with the manuscript **“A Lightweight Context-Aware Neural Offloading Strategy for IoT Edge Devices.”**

## Overview

The method treats each task-offloading decision as a context-dependent cost-minimization problem over three execution options:

- local execution
- partial offloading
- full offloading

The system context contains CPU utilization, network latency, task type, and network-condition feedback. Analytical latency and modeled energy-cost functions define the reference action costs, and a compact neural network learns to approximate those costs.

The selected network has the topology:

```text
4 -> 16 -> 16 -> 3
```

It contains 403 trainable parameters and requires 368 MACs per inference.

## Final Deployment Model

The final model is fully quantized to INT8 for TensorFlow Lite for Microcontrollers.

| Characteristic | Result |
|---|---:|
| INT8 model size | 2,776 bytes (2.71 KiB) |
| Parameters | 403 |
| MACs per inference | 368 |
| Held-out analytical-policy agreement | 99.17% |
| Practical tensor arena | 2 KiB |
| Simulated ESP32 model inference | ~0.854 ms |

The ESP32 measurements are from a proof-of-concept simulation and should not be interpreted as physical-hardware power, energy, or end-to-end communication measurements.

## Analytical Baseline

The same three-action analytical policy was also evaluated directly in the simulated ESP32 environment. Direct analytical evaluation required about 31.81 microseconds per decision, compared with about 853.53 microseconds for INT8 neural inference.

For the present cost formulation, direct analytical evaluation is faster. The neural model is therefore evaluated as a compact learned representation of the analytical policy rather than as a faster replacement for the underlying equations.

## Evaluation

The repository is intended to support reproduction of the main experiments reported in the manuscript, including:

- architecture search and model selection
- standalone model training
- FP32 TensorFlow Lite conversion
- full INT8 quantization
- analytical-reference and static-baseline evaluation
- cost-model and objective-weight sensitivity analysis
- simulated ESP32 proof-of-concept evaluation

The modeled energy values used in the experiments are analytical energy-cost proxies, not physical energy measurements.

## Repository Structure

The final repository will organize the reproducibility material as follows:

```text
training/
evaluation/
deployment/
models/
results/
figures/
```

Only the scripts and artifacts needed to reproduce or verify the reported experiments will be retained.

## License

This project is released under the MIT License.
