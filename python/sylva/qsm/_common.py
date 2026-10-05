# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Helpers shared by the QSM modules."""

from __future__ import annotations

import numpy as np


def _check_weights(weights, n: int, what: str = "weights") -> np.ndarray:
    """Per-point weights as a float array, checked: one per point, in [0, 1]."""
    w = np.asarray(weights)
    if w.dtype.kind not in "biuf":
        raise ValueError(f"{what} must be numbers in [0, 1]")
    w = np.ascontiguousarray(w, dtype=float)
    if w.ndim != 1 or len(w) != n:
        raise ValueError(f"{what} must have one value per point ({w.size} for {n} points)")
    bad = ~((w >= 0.0) & (w <= 1.0))
    if bad.any():
        raise ValueError(f"{what} must be finite and in [0, 1] (point {int(np.argmax(bad))} has {w[bad][0]})")
    return w
