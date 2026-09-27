# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Retro-reflective targets: the strongest registration feature, where present.

A target is a point, not a fitted cylinder, so it is located to millimetres,
and three shared targets fix all six degrees of freedom where a stem map fixes
four. Where scans saw targets they are tried before stems.

Targets come from what the scanner or RiSCAN PRO already found (a RIEGL
``.tpl`` tie-point list beside each scan, or a RiSCAN ``.rfl`` reflector
list), or are detected in the cloud by their return strength: retro-reflectors
return far more energy than anything natural. Reading, detection and matching
run in the Rust core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core

__all__ = [
    "Reflector",
    "ReflectorMatch",
    "detect_reflectors",
    "match_reflectors",
    "read_reflector_list",
    "read_tiepoint_list",
]


@dataclass
class Reflector:
    """One retro-reflective target, in the scan's own frame."""

    x: float
    y: float
    z: float
    reflectance: float = float("nan")
    diameter: float = float("nan")
    n_points: int = 0
    name: str = ""

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


def _positions(reflectors: list[Reflector]) -> np.ndarray:
    return np.array([r.position for r in reflectors]) if reflectors else np.zeros((0, 3))


def read_tiepoint_list(path: str | Path) -> list[Reflector]:
    """Read a RIEGL ``.tpl`` tie-point list (JSON).

    Parameters
    ----------
    path
        The ``.tpl`` file.

    Returns
    -------
    list of Reflector
        Empty for a missing or unreadable file: a position that found no
        targets is normal.
    """
    return [Reflector(*r) for r in _core.coreg_read_tiepoint_list(Path(path))]


def read_reflector_list(path: str | Path) -> list[Reflector]:
    """Read a RiSCAN PRO ``.rfl`` reflector list (``RieglRflID``).

    Each ``ReflectorN=`` line holds the fields named by the ``ReflectorIdx=``
    line (name, index, status, x, y, z, ..., reflectance, diameter, points),
    with x, y, z in the scanner's own frame. The position-level
    ``ScanPosNNN.rfl`` holds the fine-scanned targets; a scan's own
    ``<timestamp>.rfl`` those found while scanning.

    Parameters
    ----------
    path
        The ``.rfl`` file.

    Returns
    -------
    list of Reflector
        Empty for a missing or unreadable file.
    """
    return [Reflector(*r) for r in _core.coreg_read_reflector_list(Path(path))]


def detect_reflectors(
    xyz: np.ndarray,
    reflectance: np.ndarray | None,
    *,
    min_reflectance: float = 5.0,
    cluster_radius: float = 0.15,
    min_points: int = 8,
    max_extent: float = 0.5,
) -> list[Reflector]:
    """Find retro-reflective targets in a scan by their return strength.

    ``min_reflectance`` is the setting that matters and it is not universal:
    in a scan with targets the reflectance histogram is bimodal, with a wide
    gap between the scene and the targets; check it first.

    Parameters
    ----------
    xyz
        ``(n, 3)`` points.
    reflectance
        ``(n,)`` reflectance (dB).
    min_reflectance
        Returns at least this bright are target candidates.
    cluster_radius
        Single-link clustering distance (m).
    min_points
        Returns a target needs.
    max_extent
        Clusters larger than this (m) are not targets (a wet surface, a sign).

    Returns
    -------
    list of Reflector
    """
    if reflectance is None or len(xyz) == 0:
        return []
    found = _core.coreg_detect_reflectors(
        np.ascontiguousarray(np.asarray(xyz, dtype=np.float64).reshape(-1, 3)),
        np.ascontiguousarray(np.asarray(reflectance, dtype=np.float64).reshape(-1)),
        float(min_reflectance),
        float(cluster_radius),
        int(min_points),
        float(max_extent),
    )
    return [Reflector(*r) for r in found]


@dataclass
class ReflectorMatch:
    """Alignment of two target sets; ``transform`` maps source into target."""

    transform: np.ndarray
    n_inliers: int
    rmse: float
    correspondences: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))
    success: bool = False

    def __repr__(self) -> str:
        return (
            f"ReflectorMatch(success={self.success}, inliers={self.n_inliers}, "
            f"rmse={self.rmse * 1000:.1f} mm)"
        )


def match_reflectors(
    source: list[Reflector],
    target: list[Reflector],
    *,
    tolerance: float = 0.05,
    min_inliers: int = 3,
    distance_tolerance: float = 0.03,
) -> ReflectorMatch:
    """Align two target sets with no initial guess.

    Triangles are searched exhaustively (targets are few), and three
    correspondences give a full 6-DoF transform. ``distance_tolerance`` is
    tight because target positions are good to millimetres, which is what
    makes the match unambiguous.

    Parameters
    ----------
    source, target
        Targets of the two scans.
    tolerance
        How far (m) a matched target may sit from its partner.
    min_inliers
        Targets that must correspond.
    distance_tolerance
        Agreement (m) of triangle sides.

    Returns
    -------
    ReflectorMatch
        Its ``correspondences`` are ordered by source index.
    """
    d = _core.coreg_match_reflectors(
        np.ascontiguousarray(_positions(source), dtype=np.float64),
        np.ascontiguousarray(_positions(target), dtype=np.float64),
        float(tolerance),
        int(min_inliers),
        float(distance_tolerance),
    )
    return ReflectorMatch(
        d["transform"],
        int(d["n_inliers"]),
        float(d["rmse"]),
        np.asarray(d["correspondences"], dtype=int).reshape(-1, 2),
        bool(d["success"]),
    )
