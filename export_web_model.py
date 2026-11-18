"""
Turn your trained model into docs/model.json for the browser version.

Usage (run in the project folder, where model_weights.npz and actions.json are):
    python export_web_model.py

Needs only numpy. Writes docs/model.json (about 1 MB). The weights are stored as
exact 32-bit floats, so the browser computes the same numbers as asl_model.py.
"""
import base64
import json
import os
import sys

import numpy as np

SRC_WEIGHTS = "model_weights.npz"
SRC_ACTIONS = "actions.json"
OUT = os.path.join("docs", "model.json")

for path in (SRC_WEIGHTS, SRC_ACTIONS):
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run this from the folder that contains it.")

weights = np.load(SRC_WEIGHTS)
with open(SRC_ACTIONS) as f:
    actions = json.load(f)

expected = [f"lstm{i}_{p}" for i in range(3) for p in ("kernel", "recurrent", "bias")]
expected += [f"dense{i}_{p}" for i in range(3) for p in ("kernel", "bias")]
missing = [k for k in expected if k not in weights.files]
if missing:
    sys.exit(f"model_weights.npz is missing: {missing}")

n_out = weights["dense2_bias"].shape[0]
if n_out != len(actions):
    sys.exit(f"The model has {n_out} outputs but actions.json lists {len(actions)} classes.")
if weights["lstm0_kernel"].shape[0] != 63:
    sys.exit("The first LSTM layer should take 63 inputs (21 landmarks x 3).")

out = {"actions": actions, "weights": {}}
for name in expected:
    arr = np.ascontiguousarray(weights[name], dtype="<f4")
    out["weights"][name] = {"shape": list(arr.shape),
                            "b64": base64.b64encode(arr.tobytes()).decode("ascii")}

os.makedirs("docs", exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB), classes: {len(actions)}")
