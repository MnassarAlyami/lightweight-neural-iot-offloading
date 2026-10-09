# Reproducibility environment

The reported experiments were run with the following software environment:

| Component | Version |
|---|---|
| Python | 3.7.6 |
| TensorFlow | 2.11.0 |
| PyTorch | 1.13.1+cpu |
| NumPy | 1.21.6 |
| pandas | 1.3.5 |
| Matplotlib | 3.5.3 |

The original Python build was:

```text
Python 3.7.6 (default, Jan 8 2020, 20:23:39) [MSC v.1916 64 bit (AMD64)]
```

This indicates a 64-bit Windows CPython environment built with Microsoft Visual C++ 2017.

The frozen Keras artifact also records `keras_version=2.11.0` and the TensorFlow backend, while the FP32 TFLite artifact records a minimum runtime version of `2.11.0`. These artifact metadata values are consistent with the reported TensorFlow environment.

For a close reproduction of the Python-side experiments, use the package versions above. The exact PyTorch build used for the architecture search was the CPU build reported as `1.13.1+cpu`.

A full historical `pip freeze` is not required for the manuscript results because the principal numerical dependencies and framework versions used by the reported pipeline are now recorded explicitly.
