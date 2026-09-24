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
return far more energy than anything natural.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from .transforms import kabsch, transform_points

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
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    out = []
    for entry in payload:
        cartesian = (entry.get("positionCartesian") if isinstance(entry, dict) else None) or {}
        try:
            x, y, z = float(cartesian["x"]), float(cartesian["y"]), float(cartesian["z"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(
            Reflector(
                x,
                y,
                z,
                float(entry.get("reflectance", float("nan"))),
                float(entry.get("diameter", float("nan"))),
                int(entry.get("pointcount", 0)),
                str(entry.get("name", "")),
            )
        )
    return out


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
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return []
    columns: list[str] = []
    out = []
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip() == "ReflectorIdx":
            columns = [c.strip().lower() for c in value.split(",")]
            continue
        if not key.strip().startswith("Reflector") or not columns:
            continue
        row = dict(zip(columns, (v.strip() for v in value.split(",")), strict=False))
        try:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        except (KeyError, ValueError):
            continue

        def num(k: str, default=float("nan"), row=row):
            try:
                return float(row[k])
            except (KeyError, ValueError):
                return default

        out.append(
            Reflector(
                x,
                y,
                z,
                num("reflectance"),
                num("diameter"),
                int(num("points", 0)),
                row.get("name", ""),
            )
        )
    return out


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
    values = np.asarray(reflectance, dtype=np.float64)
    bright = values >= min_reflectance
    if not bright.any():
        return []
    points = np.ascontiguousarray(np.asarray(xyz, dtype=np.float64)[bright])
    values = values[bright]
    labels = np.asarray(_core.euclidean_clusters(points, float(cluster_radius), int(min_points)))
    out = []
    for label in np.unique(labels[labels >= 0]):
        members = np.flatnonzero(labels == label)
        cluster = points[members]
        extent = cluster.max(axis=0) - cluster.min(axis=0)
        if extent.max() > max_extent:
            continue
        centre = np.average(cluster, axis=0, weights=values[members] - values[members].min() + 1.0)
        out.append(
            Reflector(
                float(centre[0]),
                float(centre[1]),
                float(centre[2]),
                float(np.median(values[members])),
                float(np.linalg.norm(extent[:2])),
                len(members),
            )
        )
    return out


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
    """
    src, dst = _positions(source), _positions(target)
    failure = ReflectorMatch(np.eye(4), 0, float("inf"))
    if len(src) < 3 or len(dst) < 3:
        return failure
    dst_d = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=2)
    best = failure
    for a in range(len(src)):
        for b in range(a + 1, len(src)):
            d_ab = float(np.linalg.norm(src[a] - src[b]))
            for c in range(b + 1, len(src)):
                d_ac = float(np.linalg.norm(src[a] - src[c]))
                d_bc = float(np.linalg.norm(src[b] - src[c]))
                if min(d_ab, d_ac, d_bc) < 0.3 or _collinear(src[[a, b, c]]):
                    continue
                for i, j, k in _congruent_triangles(dst_d, d_ab, d_ac, d_bc, distance_tolerance):
                    cand = _score(src, dst, [a, b, c], [i, j, k], tolerance)
                    if cand.n_inliers > best.n_inliers or (
                        cand.n_inliers == best.n_inliers and cand.rmse < best.rmse
                    ):
                        best = cand
    if best.n_inliers < min_inliers:
        return failure
    best.success = True
    return best


def _collinear(points: np.ndarray, tolerance: float = 0.05) -> bool:
    a, b, c = points
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
    longest = max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b))
    return area < tolerance * longest**2


def _congruent_triangles(
    distances: np.ndarray, d_ab: float, d_ac: float, d_bc: float, tolerance: float
):
    n = len(distances)
    for i in range(n):
        for j in range(n):
            if j == i or abs(distances[i, j] - d_ab) > tolerance:
                continue
            for k in range(n):
                if (
                    k in (i, j)
                    or abs(distances[i, k] - d_ac) > tolerance
                    or abs(distances[j, k] - d_bc) > tolerance
                ):
                    continue
                yield i, j, k


def _score(
    src: np.ndarray, dst: np.ndarray, src_index: list[int], dst_index: list[int], tolerance: float
) -> ReflectorMatch:
    try:
        T = kabsch(src[src_index], dst[dst_index])
    except (ValueError, np.linalg.LinAlgError):
        return ReflectorMatch(np.eye(4), 0, float("inf"))
    gaps = np.linalg.norm(transform_points(T, src)[:, None, :] - dst[None, :, :], axis=2)
    nearest = np.argmin(gaps, axis=1)
    smallest = gaps[np.arange(len(src)), nearest]
    hit = smallest <= tolerance
    if hit.sum() < 3:
        return ReflectorMatch(T, int(hit.sum()), float("inf"))
    pairs, seen = [], set()
    for s in np.argsort(smallest):  # one target cannot stand in for two
        if hit[s] and nearest[s] not in seen:
            seen.add(int(nearest[s]))
            pairs.append((int(s), int(nearest[s])))
    if len(pairs) < 3:
        return ReflectorMatch(T, len(pairs), float("inf"))
    p = np.array(pairs)
    refined = kabsch(src[p[:, 0]], dst[p[:, 1]])
    residual = np.linalg.norm(transform_points(refined, src[p[:, 0]]) - dst[p[:, 1]], axis=1)
    return ReflectorMatch(refined, len(pairs), float(np.sqrt(np.mean(residual**2))), p)
