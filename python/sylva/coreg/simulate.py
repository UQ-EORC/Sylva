# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Synthetic plots and virtual scans with known poses, for testing registration.

Field data rarely comes with ground truth. These plots have known stems, and
the scans reproduce what breaks forest registration: point density and noise
that fall off with range, incidence drop-out, and above all mutual occlusion,
which is why two scans of one plot may share only a fraction of their stems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pointcloud import PointCloud
from .transforms import invert, se3_exp, yaw_transform

__all__ = ["Plot", "Survey", "Tree", "scan_plot", "simulate_plot", "simulate_survey"]


@dataclass
class Tree:
    """A leaning, tapering stem carrying a crown."""

    x: float
    y: float
    base_z: float
    dbh: float
    height: float
    lean: np.ndarray
    taper: float
    crown_base: float

    def radius_at(self, h):
        """Stem radius at height ``h`` above the base."""
        return np.maximum(0.5 * self.dbh - self.taper * (np.asarray(h, dtype=float) - 1.3), 0.01)

    def axis_point(self, h) -> np.ndarray:
        """Axis position at height ``h`` above the base."""
        h = np.atleast_1d(np.asarray(h, dtype=float))
        return (
            np.array([self.x, self.y, self.base_z])
            + (h / max(self.lean[2], 1e-6))[:, None] * self.lean
        )


@dataclass
class Plot:
    """A plot on an analytic terrain surface."""

    trees: list[Tree]
    size: float
    terrain_coeffs: np.ndarray
    seed: int = 0

    def terrain(self, xy: np.ndarray) -> np.ndarray:
        """Terrain elevation at ``(n, 2)`` locations."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        a = self.terrain_coeffs
        return (
            a[0]
            + a[1] * xy[:, 0]
            + a[2] * xy[:, 1]
            + a[3] * np.sin(xy[:, 0] / 12.0 + a[5])
            + a[4] * np.cos(xy[:, 1] / 9.0 + a[6])
        )

    @property
    def stem_positions(self) -> np.ndarray:
        """True stem axis positions 1.3 m above the base."""
        return (
            np.array([t.axis_point(1.3)[0] for t in self.trees]) if self.trees else np.zeros((0, 3))
        )

    @property
    def diameters(self) -> np.ndarray:
        return np.array([t.dbh for t in self.trees])


def simulate_plot(
    size: float = 40.0,
    n_trees: int = 45,
    *,
    seed: int = 0,
    dbh_mean: float = 0.28,
    dbh_sigma: float = 0.45,
    min_spacing: float = 1.8,
    slope: float = 0.06,
    relief: float = 1.2,
    max_lean_deg: float = 6.0,
) -> Plot:
    """A plot of ``n_trees`` on undulating terrain.

    Positions by dart throwing with a minimum spacing, DBH log-normal.

    Returns
    -------
    Plot
    """
    rng = np.random.default_rng(seed)
    coeffs = np.array(
        [
            0.0,
            rng.uniform(-slope, slope),
            rng.uniform(-slope, slope),
            rng.uniform(0.0, relief),
            rng.uniform(0.0, relief),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0, 2 * np.pi),
        ]
    )
    plot = Plot(trees=[], size=size, terrain_coeffs=coeffs, seed=seed)
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < n_trees and attempts < n_trees * 400:
        attempts += 1
        p = rng.uniform(-size / 2, size / 2, size=2)
        if positions and np.min(np.linalg.norm(np.array(positions) - p, axis=1)) < min_spacing:
            continue
        positions.append(p)
    for p in positions:
        dbh = float(np.clip(rng.lognormal(np.log(dbh_mean), dbh_sigma), 0.07, 0.95))
        height = float(np.clip(1.3 + 45.0 * dbh**0.6 + rng.normal(0, 2.0), 6.0, 40.0))
        tilt = np.radians(rng.uniform(0, max_lean_deg))
        azimuth = rng.uniform(0, 2 * np.pi)
        lean = np.array(
            [np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)]
        )
        plot.trees.append(
            Tree(
                float(p[0]),
                float(p[1]),
                float(plot.terrain(p.reshape(1, 2))[0]),
                dbh,
                height,
                lean,
                float(rng.uniform(0.004, 0.012)),
                float(np.clip(height * rng.uniform(0.35, 0.6), 2.5, height - 1.0)),
            )
        )
    return plot


def scan_plot(
    plot: Plot,
    scanner_xy: np.ndarray,
    *,
    seed: int = 0,
    max_range: float = 30.0,
    scanner_height: float = 1.6,
    angular_step: float = 0.0012,
    range_noise: float = 0.004,
    ground_density: float = 150.0,
    foliage_density: float = 2500.0,
    understorey_density: float = 70.0,
    occlusion: bool = True,
    world_frame: bool = True,
    min_stem_returns: int = 150,
) -> tuple[PointCloud, np.ndarray, np.ndarray]:
    """A virtual scan of ``plot`` from ``scanner_xy``.

    Returns
    -------
    cloud : sylva.PointCloud
        With ``label`` (tree index, -1 ground, -2 foliage, -3 understorey)
        and ``range`` attributes.
    visible : numpy.ndarray
        Trees that kept at least ``min_stem_returns`` stem points after
        occlusion: the fair recall denominator, since a stem grazed by a few
        beams cannot be fitted across several slices.
    scanner : numpy.ndarray
        ``(3,)`` scanner position in the plot frame.
    """
    rng = np.random.default_rng(seed)
    scanner_xy = np.asarray(scanner_xy, dtype=float).reshape(2)
    scanner = np.array(
        [*scanner_xy, float(plot.terrain(scanner_xy.reshape(1, 2))[0]) + scanner_height]
    )
    blockers = (
        np.array([[t.x, t.y, 0.5 * t.dbh] for t in plot.trees]) if plot.trees else np.zeros((0, 3))
    )
    chunks, labels = [], []
    visible = np.zeros(len(plot.trees), dtype=bool)
    for i, tree in enumerate(plot.trees):
        pts = _sample_stem(tree, scanner, rng, max_range, angular_step)
        if len(pts) == 0:
            continue
        if occlusion:
            pts = pts[_visible_mask(pts, scanner, blockers, skip=i)]
        visible[i] = len(pts) >= min_stem_returns
        if len(pts):
            chunks.append(pts)
            labels.append(np.full(len(pts), i, dtype=np.int32))
    foliage = _sample_foliage(plot, scanner, rng, max_range, foliage_density)
    if occlusion and len(foliage):
        foliage = foliage[_visible_mask(foliage, scanner, blockers, skip=-1, fraction=0.55)]
    ground = _sample_ground(plot, scanner, rng, max_range, ground_density)
    if occlusion and len(ground):
        ground = ground[_visible_mask(ground, scanner, blockers, skip=-1)]
    under = _sample_understorey(plot, scanner, rng, max_range, understorey_density)
    for part, label in ((foliage, -2), (ground, -1), (under, -3)):
        if len(part):
            chunks.append(part)
            labels.append(np.full(len(part), label, dtype=np.int32))
    if not chunks:
        return PointCloud(np.zeros((0, 3))), visible, scanner
    xyz = np.vstack(chunks)
    ranges = np.linalg.norm(xyz - scanner, axis=1)
    xyz = xyz + rng.normal(0, range_noise, size=xyz.shape) * (1.0 + ranges[:, None] / max_range)
    if not world_frame:
        xyz = xyz - scanner
    return PointCloud(xyz, {"label": np.concatenate(labels), "range": ranges}), visible, scanner


def _sample_stem(tree: Tree, scanner, rng, max_range, angular_step) -> np.ndarray:
    """The scanner-facing half of a stem at a realistic angular density."""
    distance = float(np.hypot(tree.x - scanner[0], tree.y - scanner[1]))
    if distance > max_range or distance < 0.3:
        return np.zeros((0, 3))
    top = min(tree.crown_base, tree.height)
    n_vert = int(np.clip(top / (distance * angular_step), 8, 4000))
    n_horiz = int(np.clip(tree.dbh * np.pi / (distance * angular_step), 4, 800))
    n = int(np.clip(n_vert * n_horiz, 0, 60000))
    if n < 10:
        return np.zeros((0, 3))
    h = rng.uniform(0.05, top, size=n)
    axis_pts = tree.axis_point(h)
    radius = tree.radius_at(h)
    to_scanner = scanner[:2] - axis_pts[:, :2]
    bearing = np.arctan2(to_scanner[:, 1], to_scanner[:, 0])
    offset = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    keep = rng.random(n) < np.cos(offset) ** 0.7  # grazing incidence drops out
    theta = bearing + offset
    xyz = np.column_stack(
        [
            axis_pts[:, 0] + radius * np.cos(theta),
            axis_pts[:, 1] + radius * np.sin(theta),
            axis_pts[:, 2],
        ]
    )
    return xyz[keep]


def _sample_ground(plot: Plot, scanner, rng, max_range, density) -> np.ndarray:
    n = int(density * np.pi * max_range**2)
    r = rng.uniform(
        0.5, max_range, size=n
    )  # uniform in radius: the 1/r thinning of grazing returns
    phi = rng.uniform(0, 2 * np.pi, size=n)
    xy = scanner[:2] + np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    xy = xy[np.all(np.abs(xy) <= plot.size / 2 + 5.0, axis=1)]
    return np.column_stack([xy, plot.terrain(xy) + rng.normal(0, 0.015, len(xy))])


def _sample_foliage(plot: Plot, scanner, rng, max_range, density) -> np.ndarray:
    out = []
    for tree in plot.trees:
        d = float(np.hypot(tree.x - scanner[0], tree.y - scanner[1]))
        if d > max_range:
            continue
        n = int(np.clip(density * (tree.height - tree.crown_base) / max(d / 8.0, 1.0), 0, 8000))
        if n < 5:
            continue
        crown_r = 0.25 * tree.height * rng.uniform(0.6, 1.0)
        h = rng.uniform(tree.crown_base, tree.height, size=n)
        centre = tree.axis_point(h)
        spread = (
            crown_r
            * (1.0 - (h - tree.crown_base) / max(tree.height - tree.crown_base, 1e-6)) ** 0.5
        )
        angle = rng.uniform(0, 2 * np.pi, n)
        rad = spread * np.sqrt(rng.random(n))
        out.append(
            np.column_stack(
                [
                    centre[:, 0] + rad * np.cos(angle),
                    centre[:, 1] + rad * np.sin(angle),
                    centre[:, 2] + rng.normal(0, 0.3, n),
                ]
            )
        )
    return np.vstack(out) if out else np.zeros((0, 3))


def _sample_understorey(plot: Plot, scanner, rng, max_range, density) -> np.ndarray:
    """Low vegetation: the main source of false stems."""
    n = int(density * np.pi * max_range**2)
    r = rng.uniform(0.5, max_range, size=n)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    xy = scanner[:2] + np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    xy = xy[np.all(np.abs(xy) <= plot.size / 2 + 5.0, axis=1)]
    return np.column_stack([xy, plot.terrain(xy) + rng.exponential(0.45, len(xy))])


def _visible_mask(points, scanner, blockers, skip: int = -1, fraction: float = 1.0) -> np.ndarray:
    """Points not shadowed by a stem (opaque vertical cylinders, in plan view)."""
    if len(blockers) == 0 or len(points) == 0:
        return np.ones(len(points), dtype=bool)
    d = points[:, :2] - scanner[:2]
    length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    direction = d / length[:, None]
    visible = np.ones(len(points), dtype=bool)
    for j, (bx, by, br) in enumerate(blockers):
        if j == skip:
            continue
        to_blocker = np.array([bx, by]) - scanner[:2]
        along = direction @ to_blocker
        perp = np.abs(direction[:, 0] * to_blocker[1] - direction[:, 1] * to_blocker[0])
        hit = (perp < br) & (along > 0.05) & (along < length - br)
        if fraction < 1.0:
            hit &= np.random.default_rng(j).random(len(points)) < fraction
        visible &= ~hit
    return visible


@dataclass
class Survey:
    """Simulated scans, each in its own scanner frame, with their true poses.

    Attributes
    ----------
    true_transforms
        ``world_from_scan`` of each scan: what a registration should recover.
    """

    clouds: list[PointCloud]
    true_transforms: list[np.ndarray]
    scanner_positions: np.ndarray
    plot: Plot
    visible: list[np.ndarray] = field(default_factory=list)

    def overlap_stems(self, i: int, j: int) -> int:
        """Trees detectable in both scans: an upper bound on matches."""
        return int((self.visible[i] & self.visible[j]).sum())


def simulate_survey(
    n_scans: int = 4,
    *,
    plot: Plot | None = None,
    seed: int = 0,
    scan_radius: float = 12.0,
    max_range: float = 30.0,
    levelled: bool = True,
    tilt_deg: float = 0.5,
    **scan_kwargs,
) -> Survey:
    """A multi-scan survey of one plot.

    Each cloud is in its own scanner frame with an unknown yaw, as a levelled
    but unoriented TLS survey. ``levelled=False`` adds a little roll and pitch.

    Returns
    -------
    Survey
    """
    rng = np.random.default_rng(seed)
    plot = plot or simulate_plot(seed=seed)
    angles = np.linspace(0, 2 * np.pi, n_scans, endpoint=False) + rng.uniform(0, 1.0)
    positions = np.column_stack([scan_radius * np.cos(angles), scan_radius * np.sin(angles)])
    positions += rng.normal(0, 1.5, positions.shape)
    if n_scans > 3:
        positions[0] = rng.normal(0, 1.0, 2)  # one central scan ties the survey together
    clouds, transforms, visible = [], [], []
    for i, p in enumerate(positions):
        cloud, vis, scanner = scan_plot(
            plot, p, seed=seed * 100 + i, max_range=max_range, world_frame=True, **scan_kwargs
        )
        yaw = rng.uniform(-np.pi, np.pi)
        if levelled:
            world_from_scan = yaw_transform(yaw, *scanner)
        else:
            xi = np.concatenate([np.radians(rng.normal(0, tilt_deg, 2)), [yaw]])
            world_from_scan = se3_exp(np.r_[xi, np.zeros(3)])
            world_from_scan[:3, 3] = scanner
        clouds.append(cloud.transform(invert(world_from_scan)))
        transforms.append(world_from_scan)
        visible.append(vis)
    return Survey(clouds, transforms, positions, plot, visible)
