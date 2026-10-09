# Frozen artifacts

The repository includes the frozen model and held-out evaluation artifacts used in the final study.

| Artifact | Size | SHA-256 |
|---|---:|---|
| `models/offloading_fp32.keras` | 36,672 B | `ffb636ef4faee5c9713c35167198f3647f5b89ac342343e2c07f2549d336267b` |
| `models/offloading_fp32.tflite` | 3,512 B | `4510014b7e2b2af794f93c954a6919945a6dc0540a2a076215ba92949d5bd3bf` |
| `models/offloading_int8.tflite` | 2,776 B | `b1ced22604827a45512871007a4a73ca5064198dc59f6c1f61c6485c8853b3e2` |
| `results/held_out/x_test.npy` | 120,128 B | `ef5f4bc0229035fcbb39b6aa5260ff0bbedd6cda257b7ff5773c9f04a511b70e` |
| `results/held_out/y_test.npy` | 90,128 B | `e89a7ad0958d547192880e12250f04d0db8ee266420cb8034b1757d24f5ac99e` |
| `results/held_out/oracle_actions.npy` | 60,128 B | `b7999f74d66a528c0078461fd8ac7442924fa70dbf184eeb73929cb83b3b4cb3` |

The ESP32 firmware embeds the INT8 model through `offloading_model.h`. Generate that header from the frozen INT8 model before compilation:

```bash
python scripts/generate_model_header.py
```

The TFLite binary hash is the canonical deployment-artifact identifier. Text formatting or line endings can change the generated C header without changing the embedded model bytes.