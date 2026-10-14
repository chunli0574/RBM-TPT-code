#!/usr/bin/env python3
"""
export_keras_weights.py

Convert a new-format Keras 3 .keras model file to a .weights.bin binary file
that the C++ simulation can load natively (no TensorFlow dependency at runtime).

Usage:
    python3 export_keras_weights.py <model_grad.keras> [<model_grad.keras> ...]

Output:
    For each input file foo_grad.keras, writes foo_grad.weights.bin to the same
    directory.

Binary layout (all float64, row-major / C order, Keras kernel convention):
    W1: (DIM_K,     NN_HIDDEN)   input → hidden
    W2: (NN_HIDDEN, NN_HIDDEN)   hidden → hidden
    W3: (NN_HIDDEN, NN_HIDDEN)   hidden → hidden
    W4: (NN_HIDDEN, DIM_K)       hidden → output
Total: (2*DIM_K*NN_HIDDEN + 2*NN_HIDDEN^2) * 8 bytes
       CC3 (DIM_K=3,  NN_HIDDEN=100): 164800 bytes
       CC10 (DIM_K=10, NN_HIDDEN=100): 165600 bytes

Architecture expected:
    4 Dense layers, no bias, ELU activation between layers.
    HDF5 paths: dense_layers/dense{_N}/vars/0

Dependencies: h5py, numpy (standard on any scientific Python install)
    pip install h5py numpy
"""

import sys
import os
import zipfile
import tempfile

try:
    import h5py
    import numpy as np
except ImportError:
    print("ERROR: install h5py and numpy first:  pip install h5py numpy")
    sys.exit(1)


def export_one(keras_path: str) -> None:
    if not keras_path.endswith(".keras"):
        print(f"  SKIP (not a .keras file): {keras_path}")
        return

    out_path = keras_path[:-len(".keras")] + ".weights.bin"
    print(f"  Input : {keras_path}")
    print(f"  Output: {out_path}")

    # .keras is a ZIP archive containing model.weights.h5
    with zipfile.ZipFile(keras_path, "r") as zf:
        if "model.weights.h5" not in zf.namelist():
            print(f"  ERROR: model.weights.h5 not found inside {keras_path}")
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extract("model.weights.h5", tmpdir)
            h5_path = os.path.join(tmpdir, "model.weights.h5")

            with h5py.File(h5_path, "r") as f:
                # Layer paths: dense_layers/dense/vars/0
                #              dense_layers/dense_1/vars/0  etc.
                W1 = f["dense_layers/dense/vars/0"][()]
                W2 = f["dense_layers/dense_1/vars/0"][()]
                W3 = f["dense_layers/dense_2/vars/0"][()]
                W4 = f["dense_layers/dense_3/vars/0"][()]

    # Verify dtypes
    for name, W in [("W1", W1), ("W2", W2), ("W3", W3), ("W4", W4)]:
        if W.dtype != np.float64:
            print(f"  WARNING: {name} dtype is {W.dtype}, converting to float64")
            W = W.astype(np.float64)

    dim_k      = W1.shape[0]   # auto-detected (3 for CC3, 10 for CC10)
    nn_hidden  = W1.shape[1]

    # Verify shapes
    expected = {
        "W1": (dim_k,    nn_hidden),
        "W2": (nn_hidden, nn_hidden),
        "W3": (nn_hidden, nn_hidden),
        "W4": (nn_hidden, dim_k),
    }
    ok = True
    for name, W, shape in [("W1", W1, expected["W1"]),
                            ("W2", W2, expected["W2"]),
                            ("W3", W3, expected["W3"]),
                            ("W4", W4, expected["W4"])]:
        if W.shape != shape:
            print(f"  ERROR: {name} shape {W.shape} != expected {shape}")
            ok = False
    if not ok:
        return

    print(f"  Architecture: DIM_K={dim_k}, NN_HIDDEN={nn_hidden}")

    # Ensure row-major (C order) before writing
    W1 = np.ascontiguousarray(W1)
    W2 = np.ascontiguousarray(W2)
    W3 = np.ascontiguousarray(W3)
    W4 = np.ascontiguousarray(W4)

    with open(out_path, "wb") as f_out:
        f_out.write(W1.tobytes())
        f_out.write(W2.tobytes())
        f_out.write(W3.tobytes())
        f_out.write(W4.tobytes())

    total_bytes = os.path.getsize(out_path)
    expected_bytes = (dim_k * nn_hidden + nn_hidden * nn_hidden +
                      nn_hidden * nn_hidden + nn_hidden * dim_k) * 8
    if total_bytes != expected_bytes:
        print(f"  ERROR: wrote {total_bytes} bytes, expected {expected_bytes}")
        return
    print(f"  OK: {total_bytes} bytes written.\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for path in sys.argv[1:]:
        export_one(path)


if __name__ == "__main__":
    main()
