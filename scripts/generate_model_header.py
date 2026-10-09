from pathlib import Path

MODEL_PATH = Path("models/offloading_int8.tflite")
HEADER_PATH = Path("deployment/esp32/offloading_model.h")

data = MODEL_PATH.read_bytes()

lines = []
for i in range(0, len(data), 12):
    chunk = data[i:i + 12]
    lines.append("    " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")

header = (
    "#ifndef OFFLOADING_MODEL_H\n"
    "#define OFFLOADING_MODEL_H\n\n"
    "#include <stdint.h>\n\n"
    "alignas(16) const unsigned char offloading_model[] = {\n"
    + "\n".join(lines)
    + "\n};\n\n"
    f"const unsigned int offloading_model_len = {len(data)};\n\n"
    "#endif\n"
)

HEADER_PATH.write_text(header, encoding="utf-8")
print(f"Saved {HEADER_PATH}")
