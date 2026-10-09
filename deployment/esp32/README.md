# Simulated ESP32 proof of concept

This folder contains the final TensorFlow Lite for Microcontrollers firmware and the frozen validation vectors used in the manuscript.

The firmware performs three validation checks:

- 20 general validation vectors
- 30 exact INT8 tie cases
- 30 cases one quantization level from a tie

It then benchmarks the TensorFlow Lite Micro `Invoke()` operation and a direct analytical implementation of the same three-action decision rule.

The configured tensor arena is 2 KiB. The smallest experimentally successful allocation in the reported sweep was 1,680 bytes.

Before compiling the firmware, generate `offloading_model.h` from the frozen model:

```bash
python scripts/generate_model_header.py
```

The exact-tie header `boundary_vectors.h` is included because it was used in the reported proof-of-concept evaluation. Its original generator was not included among the supplied source files.

The Wokwi project used for the final analytical-baseline comparison is listed in `WOKWI_PROJECT.txt`.
