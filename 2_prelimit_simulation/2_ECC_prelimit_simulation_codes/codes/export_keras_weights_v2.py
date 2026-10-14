#!/usr/bin/env python3
"""
export_keras_weights_v2.py

Convert Keras 3 .keras model files to .weights.bin binary files that the C++
simulation can load natively (no TensorFlow dependency at runtime or convert
time). Uses zip + h5py only.

Supports arbitrary NN_DEPTH by auto-detecting the number of Dense layers in
the HDF5 archive. Works for:
    CC3   (DIM_K=3,  NN_HIDDEN=100, NN_DEPTH=4)  - 164,800 bytes
    CC10  (DIM_K=10, NN_HIDDEN=200, NN_DEPTH=5)  - 992,000 bytes
    (and anything else with the same layer-naming convention)

Usage:
    python3 export_keras_weights_v2.py <model_grad.keras> [...]
    python3 export_keras_weights_v2.py /path/to/models/*.keras

Output:
    For each input file foo_grad.keras, writes foo_grad.weights.bin to the
    same directory.

Binary layout (all float64, row-major / C order, Keras kernel convention):
    W_0:         (DIM_K,     NN_HIDDEN)    input  -> hidden
    W_1..N-2:    (NN_HIDDEN, NN_HIDDEN)    hidden -> hidden
    W_{N-1}:     (NN_HIDDEN, DIM_K)        hidden -> output
    (N = NN_DEPTH = number of Dense layers)

Architecture expected:
    N Dense layers, no bias, ELU activation between hidden layers, linear output.
    HDF5 paths (inside the .keras ZIP archive's model.weights.h5):
        dense_layers/dense/vars/0
        dense_layers/dense_1/vars/0
        ...
        dense_layers/dense_{N-1}/vars/0

Dependencies: h5py, numpy
    pip install h5py numpy
"""

import sys
import os
import re
import zipfile
import tempfile

try:
    import h5py
    import numpy as np
except ImportError:
    print("ERROR: install h5py and numpy first:  pip install h5py numpy")
    sys.exit(1)


def collect_dense_layer_names(h5_file):
    """
    Return the ordered list of Dense-layer group names under 'dense_layers/':
        ['dense', 'dense_1', 'dense_2', ...]
    Sorted so that 'dense' is first and 'dense_<N>' is ordered by N.
    """
    if "dense_layers" not in h5_file:
        return []
    group = h5_file["dense_layers"]
    names = list(group.keys())

    def sort_key(n):
        # 'dense' -> 0, 'dense_1' -> 1, 'dense_2' -> 2, ...
        if n == "dense":
            return 0
        m = re.fullmatch(r"dense_(\d+)", n)
        if m:
            return int(m.group(1))
        return 10**9  # unknown names go to the end

    return sorted(names, key=sort_key)


def export_one(keras_path: str) -> bool:
    if not keras_path.endswith(".keras"):
        print(f"  SKIP (not a .keras file): {keras_path}")
        return False

    out_path = keras_path[:-len(".keras")] + ".weights.bin"
    print(f"  Input : {keras_path}")
    print(f"  Output: {out_path}")

    # .keras is a ZIP archive containing model.weights.h5
    with zipfile.ZipFile(keras_path, "r") as zf:
        if "model.weights.h5" not in zf.namelist():
            print(f"  ERROR: model.weights.h5 not found inside {keras_path}")
            return False
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extract("model.weights.h5", tmpdir)
            h5_path = os.path.join(tmpdir, "model.weights.h5")

            with h5py.File(h5_path, "r") as f:
                layer_names = collect_dense_layer_names(f)
                if not layer_names:
                    print(f"  ERROR: no 'dense_layers' group found in {keras_path}")
                    return False

                kernels = []
                for name in layer_names:
                    key = f"dense_layers/{name}/vars/0"
                    if key not in f:
                        print(f"  ERROR: missing kernel at {key}")
                        return False
                    W = f[key][()]
                    kernels.append(W)

    nn_depth = len(kernels)
    if nn_depth < 3:
        print(f"  ERROR: expected >= 3 Dense layers, found {nn_depth}")
        return False

    # Auto-detect architecture
    dim_k      = kernels[0].shape[0]
    nn_hidden  = kernels[0].shape[1]
    print(f"  Architecture: DIM_K={dim_k}, NN_HIDDEN={nn_hidden}, NN_DEPTH={nn_depth}")

    # Verify shapes:
    #   W_0           : (DIM_K,    NN_HIDDEN)
    #   W_1..N-2      : (NN_HIDDEN, NN_HIDDEN)
    #   W_{N-1}       : (NN_HIDDEN, DIM_K)
    ok = True
    for i, W in enumerate(kernels):
        if i == 0:
            expected = (dim_k, nn_hidden)
        elif i == nn_depth - 1:
            expected = (nn_hidden, dim_k)
        else:
            expected = (nn_hidden, nn_hidden)
        name = f"W{i+1}"
        if W.shape != expected:
            print(f"  ERROR: {name} shape {W.shape} != expected {expected}")
            ok = False
        else:
            print(f"    {name}: {W.shape}  dtype={W.dtype}")
    if not ok:
        return False

    # Convert to float64, row-major
    arrays = []
    for i, W in enumerate(kernels):
        if W.dtype != np.float64:
            print(f"  Converting W{i+1} from {W.dtype} to float64")
            W = W.astype(np.float64)
        arrays.append(np.ascontiguousarray(W))

    # Write binary
    with open(out_path, "wb") as f_out:
        for W in arrays:
            f_out.write(W.tobytes())

    total_bytes    = os.path.getsize(out_path)
    expected_bytes = sum(W.nbytes for W in arrays)
    if total_bytes != expected_bytes:
        print(f"  ERROR: wrote {total_bytes} bytes, expected {expected_bytes}")
        return False

    print(f"  OK: {total_bytes} bytes written.\n")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    success = 0
    fail    = 0
    for path in sys.argv[1:]:
        if export_one(path):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} converted, {fail} failed/skipped.")
    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
