#!/usr/bin/env python3
"""
export_savedmodel_weights.py

Convert TensorFlow SavedModel directories (*_grad/) to .weights.bin binary files
that the C++ simulation can load natively (no TensorFlow dependency at runtime).

Usage:
    python3 export_savedmodel_weights.py <model_grad_dir> [<model_grad_dir> ...]

    Supports shell globs:
    python3 export_savedmodel_weights.py /path/to/models/*dim=10*_grad

Output:
    For each input directory foo_grad/, writes foo_grad.weights.bin in the same
    parent directory.

Binary layout (all float64, row-major / C order, Keras kernel convention):
    W1:   (DIM_K,     NN_HIDDEN)   input -> hidden
    W2..N-1: (NN_HIDDEN, NN_HIDDEN)   hidden -> hidden
    WN:   (NN_HIDDEN, DIM_K)       hidden -> output
    (N = NN_DEPTH = number of Dense layers)

    CC3  (DIM_K=3,  NN_HIDDEN=100, NN_DEPTH=4):  164,800 bytes
    CC10 (DIM_K=10, NN_HIDDEN=200, NN_DEPTH=5):  992,000 bytes

Architecture expected:
    NN_DEPTH Dense layers, no bias, ELU activation between hidden layers, linear output.
    CC3: 100_100_100_D3 → 4 Dense layers (3 hidden + 1 output)
    CC10: 200_200_200_D4 → 5 Dense layers (4 hidden + 1 output)
    Keras kernel convention: output = input @ W  (no transpose needed).

Dependencies: tensorflow (or tensorflow-cpu), numpy
    conda activate tensorflow-test/env   (or your TF environment)
"""

import sys
import os
import numpy as np


def export_one(savedmodel_dir: str) -> bool:
    """Convert one SavedModel directory to .weights.bin. Returns True on success."""

    # Strip trailing slash
    savedmodel_dir = savedmodel_dir.rstrip("/")

    if not os.path.isdir(savedmodel_dir):
        print(f"  SKIP (not a directory): {savedmodel_dir}")
        return False

    if not os.path.exists(os.path.join(savedmodel_dir, "saved_model.pb")):
        print(f"  SKIP (no saved_model.pb): {savedmodel_dir}")
        return False

    out_path = savedmodel_dir + ".weights.bin"
    print(f"  Input : {savedmodel_dir}/")
    print(f"  Output: {out_path}")

    # Load model via tf.keras
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(savedmodel_dir, compile=False)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return False

    # Extract Dense layer kernels (no bias expected)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    nn_depth = len(dense_layers)
    if nn_depth < 3:
        print(f"  ERROR: expected >= 3 Dense layers, found {nn_depth}")
        print(f"         layers: {[l.name for l in model.layers]}")
        return False

    kernels = []
    for i, layer in enumerate(dense_layers):
        weights = layer.get_weights()
        if len(weights) == 1:
            kernels.append(weights[0])
        elif len(weights) == 2:
            print(f"  WARNING: layer {layer.name} has bias (will be ignored)")
            kernels.append(weights[0])
        else:
            print(f"  ERROR: layer {layer.name} has {len(weights)} weight arrays")
            return False

    # Auto-detect architecture from first and last kernels
    dim_k     = kernels[0].shape[0]
    nn_hidden = kernels[0].shape[1]
    print(f"  Architecture: DIM_K={dim_k}, NN_HIDDEN={nn_hidden}, NN_DEPTH={nn_depth}")

    # Verify shapes: W1(dim_k, hidden), W2..N-1(hidden, hidden), WN(hidden, dim_k)
    ok = True
    for i, W in enumerate(kernels):
        if i == 0:
            expected_shape = (dim_k, nn_hidden)
        elif i == nn_depth - 1:
            expected_shape = (nn_hidden, dim_k)
        else:
            expected_shape = (nn_hidden, nn_hidden)
        name = f"W{i+1}"
        if W.shape != expected_shape:
            print(f"  ERROR: {name} shape {W.shape} != expected {expected_shape}")
            ok = False
        else:
            print(f"    {name}: {W.shape}  dtype={W.dtype}")
    if not ok:
        return False

    # Convert to float64 row-major
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

    total_bytes = os.path.getsize(out_path)
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

    # Suppress TF warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Import TF once here so any import error is caught early
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}\n")
    except ImportError:
        print("ERROR: TensorFlow not available. Activate your TF environment first.")
        print("       e.g.:  conda activate tensorflow-test/env")
        sys.exit(1)

    success = 0
    fail = 0
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
