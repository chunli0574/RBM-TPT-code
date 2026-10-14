"""Closed-form backup reflection for the Rybko--Stolyar workload model."""

from __future__ import annotations

import numpy as np


RS_CONTROL_ORDER = ("L1", "L2", "U1", "U2")

M11 = 20.0 / 19.0
M21 = 40.0 / 19.0

G_CONTROLS = np.array(
    [
        [1.0, 0.0, -M11, -M21],
        [0.0, 1.0, -M21, -M11],
    ],
    dtype=np.float64,
)

def validate_rs_backup_configuration(eta_upper: float) -> float:
    """Return eta_upper as a float, rejecting non-finite or non-positive values."""
    eta = float(eta_upper)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta_upper must be a finite positive RS state-space scale")
    return eta


def reflect_rs_closed_form(w, eta_upper: float):
    """Apply the vectorized three-step RS reflection.

    Args:
        w: One point with shape ``(2,)`` or a batch with shape ``(batch, 2)``.
        eta_upper: Scale of the RS state space. This is deliberately not ``rho``.

    Returns:
        ``(y, dpush)`` with control order ``(L1, L2, U1, U2)``. A scalar input
        returns shapes ``(2,)`` and ``(4,)``; a batch returns ``(batch, 2)`` and
        ``(batch, 4)``.
    """
    eta = validate_rs_backup_configuration(eta_upper)
    points = np.asarray(w, dtype=np.float64)
    scalar_input = points.ndim == 1
    if scalar_input:
        if points.shape != (2,):
            raise ValueError(f"w must have shape (2,) or (batch, 2); got {points.shape}")
        points_2d = points.reshape(1, 2)
    else:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"w must have shape (2,) or (batch, 2); got {points.shape}")
        points_2d = points

    if not np.all(np.isfinite(points_2d)):
        raise ValueError("w must contain only finite values")

    w1 = points_2d[:, 0]
    w2 = points_2d[:, 1]
    b1 = (20.0 / 3.0) * eta
    b2 = (80.0 / 9.0) * eta

    # Each expression below acts on the complete batch; there is no per-sample loop.
    s = np.maximum.reduce(
        (
            np.zeros_like(w1),
            (w1 - b1) / M11,
            (w2 - b1) / M21,
        )
    )
    wc1 = w1 - M11 * s
    wc2 = w2 - M21 * s
    y1 = np.maximum.reduce((wc1, np.zeros_like(wc1), 2.0 * wc2 - b2))
    y2 = np.maximum.reduce((wc2, np.zeros_like(wc2), 2.0 * y1 - b2))

    y = np.column_stack((y1, y2))
    dpush = np.column_stack(
        (
            np.maximum(y1 - wc1, 0.0),
            np.maximum(y2 - wc2, 0.0),
            s,
            np.zeros_like(s),
        )
    )

    if scalar_input:
        return y[0], dpush[0]
    return y, dpush


def expand_rs_boundary_pushing_cost(raw_cost, v1: float, v2: float) -> list[float]:
    """Expand ``[add_L, add_U_ratio]`` to the four-control backup cost."""
    raw = np.asarray(raw_cost, dtype=np.float64)
    if raw.shape != (2,) or not np.all(np.isfinite(raw)):
        raise ValueError("raw boundary_pushing_cost must be two finite values")
    v1_value = float(v1)
    v2_value = float(v2)
    if not np.isfinite(v1_value) or not np.isfinite(v2_value):
        raise ValueError("v1 and v2 must be finite")
    add_l, add_u_ratio = raw
    return [
        float(add_l),
        float(add_l),
        float((1.0 + add_u_ratio) * v1_value),
        v2_value,
    ]
