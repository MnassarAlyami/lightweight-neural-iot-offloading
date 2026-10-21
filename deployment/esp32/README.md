# Simulated ESP32 proof of concept

This folder contains the final TensorFlow Lite for Microcontrollers firmware and the frozen validation vectors used in the manuscript.

The firmware performs three validation checks:

- 20 general validation vectors
- 30 exact INT8 tie cases
- 30 cases one quantization level from a tie

It then benchmarks the TensorFlow Lite Micro `Invoke()` operation and a direct analytical implementation of the same three-action decision rule. The neural benchmark times `Invoke()` only with the input tensor preloaded, whereas the analytical benchmark evaluates the formulas and selects the action across eight representative states.

The firmware is configured with a 2 KiB TFLM tensor arena.

Before compiling the firmware, generate `offloading_model.h` from the frozen model:

```bash
python scripts/generate_model_header.py
```

The exact-tie header `boundary_vectors.h` is included because it was used in the reported proof-of-concept evaluation. The original generator was not recovered; the reconstructed helper is documented in the repository root.

The Wokwi project used for the final analytical-baseline comparison is listed in `WOKWI_PROJECT.txt`.
