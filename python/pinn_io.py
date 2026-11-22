"""Plain-text storage of the DD-PINN weights (weights.json).

The file maps the names of the torch state dict (layer_in.weight, layers_middle.0.weight, ...,
layer_out.bias) to nested lists. The values are float32, written with the shortest decimal
representation that reads back to the same float32, so the round trip is exact.
"""
import json
import numpy as np
import torch


def _format(a):
    if a.ndim == 1:
        return '[' + ','.join(np.format_float_positional(x, unique=True, trim='-') if x == 0 or abs(x) >= 1e-4
                              else np.format_float_scientific(x, unique=True, trim='-') for x in a) + ']'
    return '[\n' + ',\n'.join(_format(row) for row in a) + '\n]'


def write_weights(state_dict, path):
    with open(path, 'w') as f:
        f.write('{\n' + ',\n'.join(f'"{name}": {_format(t.detach().cpu().numpy().astype(np.float32))}'
                                   for name, t in state_dict.items()) + '\n}\n')


def read_weights(path):
    with open(path) as f:
        return {name: torch.tensor(np.array(value, dtype=np.float32)) for name, value in json.load(f).items()}
