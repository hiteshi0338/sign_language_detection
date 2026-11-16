import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "actions.json")) as f:
    ACTIONS = json.load(f)                      
W = np.load(os.path.join(HERE, "model_weights.npz"))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lstm(x, kernel, recurrent, bias, return_sequences):
    """Keras LSTM forward pass (gate order: input, forget, cell, output)."""
    units = recurrent.shape[0]
    z = x @ kernel + bias                       
    h = np.zeros(units, dtype=np.float32)
    c = np.zeros(units, dtype=np.float32)
    outputs = []
    for t in range(x.shape[0]):
        g = z[t] + h @ recurrent
        i, fg, cell, o = np.split(g, 4)
        c = _sigmoid(fg) * c + _sigmoid(i) * np.tanh(cell)
        h = _sigmoid(o) * np.tanh(c)
        outputs.append(h)
    return np.stack(outputs) if return_sequences else h


def predict_probs(sequence):
    """sequence: (SEQUENCE_LENGTH, 63) array -> probability for each entry of ACTIONS."""
    x = np.asarray(sequence, dtype=np.float32)
    x = _lstm(x, W["lstm0_kernel"], W["lstm0_recurrent"], W["lstm0_bias"], True)
    x = _lstm(x, W["lstm1_kernel"], W["lstm1_recurrent"], W["lstm1_bias"], True)
    x = _lstm(x, W["lstm2_kernel"], W["lstm2_recurrent"], W["lstm2_bias"], False)
    x = np.maximum(x @ W["dense0_kernel"] + W["dense0_bias"], 0)   
    x = np.maximum(x @ W["dense1_kernel"] + W["dense1_bias"], 0)   
    logits = x @ W["dense2_kernel"] + W["dense2_bias"]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def display_name(label):
    return "(no sign)" if label == "NOSIGN" else label
