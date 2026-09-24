# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Stem maps: the features a coarse alignment is found from.

Forests defeat the usual registration cues: there are almost no large planes,
and keypoint descriptors are unstable because foliage changes between
viewpoints and with wind. Stems are stable, sparse, roughly vertical and
described by three numbers (x, y, diameter), so reducing each scan to a stem
map turns registration into a small 2-D pattern-matching problem that can be
solved with no initial guess.

The detector is the Rust one behind :func:`sylva.trees.detect_stems`, run in
its tlsalign mode (0.4 m slice spacing, no prefilter, over-wide clusters
skipped, RANSAC in blocks of 32), so the stem maps are tlsalign's.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .. import _core
from .ground import GroundModel, fit_ground
from .transforms import transform_points, transform_vectors

__all__ = ["Stem", "StemDetectionConfig", "StemMap", "detect_stems", "stem_map_from_arrays"]


@dataclass
class Stem:
    """A detected stem.

    ``x``, ``y``, ``z`` are the axis at ``reference_height`` above the ground,
    in the scan's own frame (``z`` is an elevation in that frame, not a
    height). ``axis`` points up.
    """

    x: float
    y: float
    z: float
    dbh: float
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    reference_height: float = 1.3
    n_slices: int = 0
    n_points: int = 0
    rmse: float = 0.0
    coverage: float = 0.0
    lean_deg: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def radius(self) -> float:
        return 0.5 * self.dbh

    @property
    def quality(self) -> float:
        """0-1 confidence from the fit residual, the arc seen and the slices linked."""
        residual_term = 1.0 / (1.0 + self.rmse / 0.01)
        slice_term = min(self.n_slices / 6.0, 1.0)
        return float(np.clip(residual_term * self.coverage * slice_term, 0.0, 1.0))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axis"] = [float(v) for v in self.axis]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Stem:
        names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in names}
        d["axis"] = np.asarray(d.get("axis", [0.0, 0.0, 1.0]), dtype=float)
        return cls(**d)


@dataclass
class StemMap:
    """The stems of one scan."""

    stems: list[Stem]
    name: str = ""
    ground: GroundModel | None = None

    def __len__(self) -> int:
        return len(self.stems)

    def __iter__(self):
        return iter(self.stems)

    def __getitem__(self, i):
        return self.stems[i]

    def __repr__(self) -> str:
        return f"StemMap(name={self.name!r}, n_stems={len(self.stems)})"

    @property
    def positions(self) -> np.ndarray:
        """``(n, 3)`` stem positions at the reference height."""
        return np.array([s.position for s in self.stems]) if self.stems else np.zeros((0, 3))

    @property
    def xy(self) -> np.ndarray:
        return self.positions[:, :2]

    @property
    def diameters(self) -> np.ndarray:
        return np.array([s.dbh for s in self.stems]) if self.stems else np.zeros(0)

    @property
    def qualities(self) -> np.ndarray:
        return np.array([s.quality for s in self.stems]) if self.stems else np.zeros(0)

    @property
    def axes(self) -> np.ndarray:
        return np.array([s.axis for s in self.stems]) if self.stems else np.zeros((0, 3))

    def sorted_by_quality(self) -> StemMap:
        """Best first (numpy's default sort, as tlsalign, so ties fall the same way)."""
        order = np.argsort(-self.qualities)
        return StemMap([self.stems[i] for i in order], name=self.name, ground=self.ground)

    def top(self, n: int) -> StemMap:
        """The ``n`` best stems; matching cost is quadratic in the count."""
        return StemMap(self.sorted_by_quality().stems[:n], name=self.name, ground=self.ground)

    def transformed(self, T: np.ndarray) -> StemMap:
        """Every stem position and axis moved by a rigid transform."""
        if not self.stems:
            return StemMap([], name=self.name)
        pos = transform_points(T, self.positions)
        ax = transform_vectors(T, self.axes)
        out = []
        for stem, p, a in zip(self.stems, pos, ax, strict=True):
            new = Stem(**{**asdict(stem), "axis": a})
            new.x, new.y, new.z = (float(v) for v in p)
            out.append(new)
        return StemMap(out, name=self.name)

    def save(self, path: str | Path) -> Path:
        """Write the stems as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"name": self.name, "stems": [s.to_dict() for s in self.stems]}, indent=2)
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> StemMap:
        """Read stems written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        return cls([Stem.from_dict(s) for s in payload["stems"]], name=payload.get("name", ""))


@dataclass
class StemDetectionConfig:
    """Settings of :func:`detect_stems`; the defaults are tlsalign's.

    They suit plot-scale TLS (10-30 m range, stems 5-120 cm). In dense
    understorey raise ``min_slices`` and ``min_coverage``; for buttressed
    stems raise ``slice_min_height`` above the buttresses.
    """

    slice_min_height: float = 1.0
    slice_max_height: float = 5.0
    slice_thickness: float = 0.3
    slice_step: float = 0.4
    reference_height: float = 1.3
    min_radius: float = 0.025
    max_radius: float = 0.60
    cluster_cell: float = 0.06
    min_cluster_points: int = 12
    max_cluster_extent: float = 2.0
    """Clusters wider than this (m) are too big to be one stem and are skipped."""
    ransac_iterations: int = 120
    ransac_tolerance: float = 0.02
    max_circles_per_cluster: int = 3
    min_circle_inliers: int = 10
    min_coverage: float = 0.12
    max_circle_rmse: float = 0.02
    link_radius: float = 0.20
    link_radius_ratio: float = 0.45
    min_slices: int = 3
    max_lean_deg: float = 25.0
    seed: int = 0


def _detector_kwargs(cfg: StemDetectionConfig) -> dict:
    """The Rust detector's arguments for this config, in tlsalign mode."""
    kw = dict(_core.stems_tlsalign_defaults())
    kw.update(
        slice_min=cfg.slice_min_height,
        slice_max=cfg.slice_max_height,
        slice_thickness=cfg.slice_thickness,
        slice_step=cfg.slice_step,
        reference_height=cfg.reference_height,
        min_radius=cfg.min_radius,
        max_radius=cfg.max_radius,
        cluster_cell=cfg.cluster_cell,
        min_cluster_points=cfg.min_cluster_points,
        max_cluster_extent=cfg.max_cluster_extent,
        ransac_iterations=cfg.ransac_iterations,
        ransac_tolerance=cfg.ransac_tolerance,
        max_circles_per_cluster=cfg.max_circles_per_cluster,
        min_circle_inliers=cfg.min_circle_inliers,
        min_coverage=cfg.min_coverage,
        max_circle_rmse=cfg.max_circle_rmse,
        link_radius=cfg.link_radius,
        link_radius_ratio=cfg.link_radius_ratio,
        min_slices=cfg.min_slices,
        max_lean_deg=cfg.max_lean_deg,
        seed=cfg.seed,
    )
    return kw


def detect_stems(
    points: np.ndarray,
    ground: GroundModel | None = None,
    config: StemDetectionConfig | None = None,
    *,
    name: str = "",
    heights: np.ndarray | None = None,
) -> StemMap:
    """Detect the stems of one scan.

    Horizontal slices through the stem band are clustered and fitted with
    RANSAC circles, the circles are linked up the stem, and each chain gives
    an axis, a diameter at breast height and a quality.

    Parameters
    ----------
    points
        ``(n, 3)`` points in the scan's own (levelled) frame.
    ground
        Terrain model; fitted if None.
    config
        Detector settings.
    name
        Name of the stem map.
    heights
        Heights above ground of ``points``, if already known.

    Returns
    -------
    StemMap
        Stems, best first, with ``z`` on the scan's own terrain.
    """
    cfg = config or StemDetectionConfig()
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    if len(points) < 100:
        return StemMap([], name=name, ground=ground)
    if heights is None:
        if ground is None:
            ground = fit_ground(points)
        heights = ground.normalise(points)
    found = _core.detect_stems(
        points, np.ascontiguousarray(heights, dtype=np.float64), **_detector_kwargs(cfg)
    )
    stems = []
    for t in found:
        axis = np.asarray(t.get("axis", (0.0, 0.0, 1.0)), dtype=float)
        stems.append(
            Stem(
                x=float(t["x"]),
                y=float(t["y"]),
                z=float(t.get("z", cfg.reference_height)),
                dbh=float(t["dbh"]),
                axis=axis,
                reference_height=cfg.reference_height,
                n_slices=int(t["n_slices"]),
                n_points=int(t["n_points"]),
                rmse=float(t["rmse"]),
                coverage=float(t.get("coverage", 0.0)),
                lean_deg=float(t["lean_deg"]),
            )
        )
    # Circles are fitted in height-above-ground, so z comes back as the
    # reference height; lift it into the scan's own datum, or the vertical
    # offset between two scans could never be estimated.
    if ground is not None and stems:
        terrain = ground.height_at(np.array([[s.x, s.y] for s in stems]))
        for stem, z0 in zip(stems, terrain, strict=True):
            stem.z = float(z0 + cfg.reference_height)
    return StemMap(stems, name=name, ground=ground).sorted_by_quality()


def stem_map_from_arrays(xy: np.ndarray, dbh=None, z=None, name: str = "") -> StemMap:
    """A stem map from plain arrays (tests and external stem lists).

    Parameters
    ----------
    xy
        ``(n, 2)`` positions.
    dbh
        Diameters (m); 0.3 if None.
    z
        Elevations; 0 if None.
    name
        Name of the map.

    Returns
    -------
    StemMap
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    dbh = np.full(len(xy), 0.3) if dbh is None else np.asarray(list(dbh), dtype=float)
    z = np.zeros(len(xy)) if z is None else np.asarray(list(z), dtype=float)
    return StemMap(
        [
            Stem(
                float(p[0]),
                float(p[1]),
                float(zz),
                float(d),
                n_slices=5,
                n_points=200,
                rmse=0.005,
                coverage=0.5,
            )
            for p, d, zz in zip(xy, dbh, z, strict=True)
        ],
        name=name,
    )
