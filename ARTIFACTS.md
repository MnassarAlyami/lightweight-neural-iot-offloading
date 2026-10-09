# Model artifacts

The frozen INT8 deployment artifact included in this repository was reconstructed byte-for-byte from the deployed C header.

| Artifact | Size | SHA-256 |
|---|---:|---|
| `models/offloading_int8.tflite` | 2,776 B | `b1ced22604827a45512871007a4a73ca5064198dc59f6c1f61c6485c8853b3e2` |
| `deployment/esp32/offloading_model.h` | 18,018 B | `2b7e619600301a8a7b6aba8ad65ce0c52529560491b5766e3eb7f65c7a7171bb` |

The training and conversion scripts generate the FP32 Keras, FP32 TFLite, and INT8 TFLite files in `outputs/`. Exact binary hashes can depend on the TensorFlow version and conversion environment, so the included INT8 artifact should be used when reproducing the proof-of-concept firmware results.
