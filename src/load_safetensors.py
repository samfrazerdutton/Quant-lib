import struct, os, numpy as np
from safetensors.torch import load_file

MODEL_DIR  = os.path.expanduser("~/tinyllama")
OUTPUT_BIN = os.path.expanduser("~/tinyllama/weights.bin")

print(f"Loading from {MODEL_DIR}...")
weights = {}
for f in sorted(os.listdir(MODEL_DIR)):
    if f.endswith(".safetensors"):
        print(f"  reading {f}")
        weights.update(load_file(os.path.join(MODEL_DIR, f)))

print(f"Found {len(weights)} tensors, writing binary...")

with open(OUTPUT_BIN, "wb") as out:
    out.write(struct.pack("<I", len(weights)))
    for name, tensor in weights.items():
        arr = tensor.float().numpy()
        if arr.ndim == 1:
            rows, cols = 1, arr.shape[0]
        else:
            rows, cols = arr.shape[0], int(np.prod(arr.shape[1:]))
        arr = arr.reshape(rows, cols)
        name_b = name.encode("utf-8")
        out.write(struct.pack("<I", len(name_b)))
        out.write(name_b)
        out.write(struct.pack("<II", rows, cols))
        out.write(arr.astype(np.float32).tobytes())
        print(f"  {name:60s} [{rows:6d} x {cols:6d}]")

print(f"\nWrote {os.path.getsize(OUTPUT_BIN)/1e6:.0f} MB -> {OUTPUT_BIN}")
