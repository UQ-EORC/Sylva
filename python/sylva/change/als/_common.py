# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Argument checks shared by the airborne change functions."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from ...als import Catalog
from ...raster import Raster


def _num(name: str, v, lo: float | None = None, strict: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    if lo is not None and (v < lo or (strict and v == lo)):
        bound = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {bound} {lo}, got {v!r}")
    return float(v)


def _pos(name: str, v) -> float:
    return _num(name, v, 0.0, strict=True)


def _count(name: str, v, lo: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, Integral) or v < lo:
        raise ValueError(f"{name} must be an integer of at least {lo}, got {v!r}")
    return int(v)


def _share(name: str, v) -> float:
    v = _num(name, v)
    if not 0 <= v <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {v}")
    return v


def _confidence(v) -> float:
    v = _num("confidence", v)
    if not 0 < v < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {v}")
    return v


def _chunk_size(v) -> float | None:
    return None if v is None else _pos("chunk_size", v)


def _raster_arg(r: Raster, name: str) -> tuple:
    if not isinstance(r, Raster):
        raise ValueError(f"{name} must be a Raster, got {type(r).__name__}")
    return (np.ascontiguousarray(r.data, dtype=np.float64), float(r.xmin), float(r.ymin),
            float(r.resolution))


def _raster(cat: Catalog, d: dict) -> Raster:
    return cat._raster(d)
