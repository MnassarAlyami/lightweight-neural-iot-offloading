# Model artifacts

The repository includes the frozen INT8 TensorFlow Lite model used in the proof-of-concept evaluation.

| Artifact | Size | SHA-256 |
|---|---:|---|
| `models/offloading_int8.tflite` | 2,776 B | `b1ced22604827a45512871007a4a73ca5064198dc59f6c1f61c6485c8853b3e2` |

The ESP32 firmware embeds this binary through `offloading_model.h`. Run:

```bash
python scripts/generate_model_header.py
```

to generate that header from the frozen model before compilation. The textual header can differ in formatting or line endings while embedding the same 2,776 model bytes, so the TFLite binary hash above is the canonical deployment-artifact identifier.

The training and conversion scripts also generate FP32 Keras, FP32 TFLite, and INT8 TFLite files in `outputs/`. Exact binary hashes of regenerated artifacts can depend on the TensorFlow version and conversion environment; use the included frozen INT8 model when reproducing the reported proof-of-concept firmware behavior.
