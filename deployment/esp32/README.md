# Simulated ESP32 proof of concept

This folder contains the final INT8 TensorFlow Lite for Microcontrollers proof-of-concept firmware and the frozen validation vectors used in the manuscript.

The firmware performs three validation checks:

- 20 general validation vectors
- 30 exact INT8 tie cases
- 30 cases one quantization level from a tie

It then benchmarks the TensorFlow Lite Micro `Invoke()` operation and a direct analytical implementation of the same three-action decision rule.

The configured tensor arena is 2 KiB. The smallest experimentally successful allocation in the reported sweep was 1,680 bytes.

The Wokwi project used for the final analytical-baseline comparison is listed in `WOKWI_PROJECT.txt`.
