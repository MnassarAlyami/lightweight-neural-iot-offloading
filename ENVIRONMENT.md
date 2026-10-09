# Reproducibility environment

The frozen Keras artifact records `keras_version=2.11.0` and the TensorFlow backend. The FP32 TFLite artifact records a minimum runtime version of `2.11.0`. These metadata values do not fully identify the original Python environment or exact package builds.

If the original training/search environment is still available, record it with:

```bash
python --version
python -c "import tensorflow as tf, torch, numpy as np, pandas as pd, matplotlib; print('tensorflow', tf.__version__); print('torch', torch.__version__); print('numpy', np.__version__); print('pandas', pd.__version__); print('matplotlib', matplotlib.__version__)"
python -m pip freeze > environment-freeze.txt
```

Run these commands in the same virtual environment, Conda environment, or saved runtime that was used for the reported experiments. If that environment no longer exists, do not replace the missing versions with versions from a new environment; keep the verified artifact metadata above and treat the remaining versions as unavailable.
