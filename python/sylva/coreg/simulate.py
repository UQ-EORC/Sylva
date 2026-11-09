# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Synthetic plots and virtual scans with known poses, for testing registration.

Field data rarely comes with ground truth. These plots have known stems, and
the scans reproduce what breaks forest registration: point density and noise
that fall off with range, incidence drop-out, and above all mutual occlusion,
which is why two scans of one plot may share only a fraction of their stems.

The plot (a few dozen trees and seven terrain coefficients) is drawn here; the
points of a scan are sampled by the core (``sylva_rs::coreg::simulate``), with
the same random stream as the NumPy implementation it replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import _core
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
        """Stem radius at height ``h`` above the base.

        Parameters
        ----------
        h
            Height above the stem base (m), a scalar or an array.

        Returns
        -------
        numpy.ndarray
            Radius (m): half the DBH at 1.3 m, narrowing by ``taper`` m per metre of height,
            never below 0.01 m.
        """
        return np.maximum(0.5 * self.dbh - self.taper * (np.asarray(h, dtype=float) - 1.3), 0.01)

    def axis_point(self, h) -> np.ndarray:
        """Axis position at height ``h`` above the base.

        Parameters
        ----------
        h
            Height above the stem base (m), a scalar or ``(n,)`` array.

        Returns
        -------
        numpy.ndarray
            ``(n, 3)`` points on the stem axis in the plot frame: the base displaced along the
            unit ``lean`` vector until it has risen ``h``.
        """
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
        """Terrain elevation at ``(n, 2)`` locations.

        Parameters
        ----------
        xy
            ``(n, 2)`` horizontal positions (m) in the plot frame.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` elevation (m): a plane plus two sinusoids, from ``terrain_coeffs``.
        """
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
        """``(n,)`` DBH of every simulated tree, m."""
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

    Parameters
    ----------
    size
        Side of the square plot (m), centred on the origin of the plot frame.
    n_trees
        Trees to place; fewer if the spacing cannot be met in ``400 * n_trees`` throws.
    seed
        Seed of every random draw.
    dbh_mean, dbh_sigma
        Median DBH (m) and the standard deviation of its logarithm; DBH is clipped to
        0.07-0.95 m, and tree height follows DBH.
    min_spacing
        Least distance between stems (m).
    slope
        Largest terrain gradient along x and along y (rise over run), each drawn uniformly
        from ``[-slope, slope]``.
    relief
        Largest amplitude (m) of each of the two sinusoidal undulations (wavelengths of
        about 75 m in x and 57 m in y).
    max_lean_deg
        Largest stem lean from vertical (degrees), drawn uniformly from 0.

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

    Parameters
    ----------
    plot
        The plot to scan.
    scanner_xy
        ``(2,)`` horizontal scanner position (m) in the plot frame; the scanner stands
        ``scanner_height`` above the terrain there.
    seed
        Seed of the sampling and the noise.
    max_range
        Trees further away (m) are not seen; ground and understorey are sampled out to this
        range, and the range noise doubles between the scanner and it.
    scanner_height
        Scanner height above the ground (m).
    angular_step
        Beam spacing (radians): a stem returns about one point per ``angular_step`` squared
        of the solid angle it fills, before grazing-incidence drop-out; 0.0012 rad is about
        0.07 degrees.
    range_noise
        Standard deviation (m) of the Gaussian noise added to each coordinate at zero
        range; it grows linearly to twice that at ``max_range``.
    ground_density
        Ground returns per m² of the disc of radius ``max_range`` round the scanner, drawn
        uniform in range (so denser near the scanner), with 0.015 m of vertical noise.
    foliage_density
        Crown points per metre of crown depth for a tree within 8 m, falling as 1/range
        beyond that and capped at 8000 per tree.
    understorey_density
        Low-vegetation returns per m² of the same disc as ``ground_density``, their heights
        exponential with a 0.45 m mean: the source of false stems.
    occlusion
        Drop points shadowed by stems (opaque vertical cylinders in plan view); foliage is
        shadowed with probability 0.55, as a leaky crown.
    world_frame
        Return the points in the plot frame. If False, the scanner position is subtracted,
        giving a scanner-centred frame with the plot's orientation (no yaw; see
        :func:`simulate_survey`).
    min_stem_returns
        Stem points a tree needs after occlusion to count as visible.

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
    trees = (
        np.array(
            [[t.x, t.y, t.base_z, t.dbh, t.height, *np.asarray(t.lean, dtype=float).reshape(3),
              t.taper, t.crown_base] for t in plot.trees],
            dtype=float,
        ).reshape(-1, 10)
    )
    scanner_xy = np.asarray(scanner_xy, dtype=float).reshape(2)
    xyz, label, ranges, visible, scanner = _core.coreg_simulate_scan(
        trees, [float(c) for c in plot.terrain_coeffs], float(plot.size),
        [float(scanner_xy[0]), float(scanner_xy[1])], int(seed),
        max_range=float(max_range), scanner_height=float(scanner_height),
        angular_step=float(angular_step), range_noise=float(range_noise),
        ground_density=float(ground_density), foliage_density=float(foliage_density),
        understorey_density=float(understorey_density), occlusion=bool(occlusion),
        world_frame=bool(world_frame), min_stem_returns=int(min_stem_returns),
    )
    scanner = np.array(scanner)
    if len(xyz) == 0:
        return PointCloud(np.zeros((0, 3))), visible, scanner
    return PointCloud(xyz, {"label": label, "range": ranges}), visible, scanner


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
        """Trees detectable in both scans: an upper bound on matches.

        Parameters
        ----------
        i, j
            Scan indices into ``clouds``.

        Returns
        -------
        int
            Trees with at least ``min_stem_returns`` stem points in both scans
            (``visible[i] & visible[j]``).
        """
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

    Parameters
    ----------
    n_scans
        Scan positions, spread evenly round a circle of ``scan_radius`` and jittered by
        1.5 m; with more than 3 scans the first is moved to the plot centre instead.
    plot
        The plot to scan; ``simulate_plot(seed=seed)`` if None.
    seed
        Seed of the positions, yaws and tilts, and of each scan's sampling.
    scan_radius
        Radius (m) of the circle of scan positions about the plot centre.
    max_range
        Scanner range (m), passed to :func:`scan_plot`.
    levelled
        Each pose is a yaw and a translation only, as a levelled scanner gives.
    tilt_deg
        With ``levelled=False``, standard deviation (degrees) of the roll and pitch drawn
        for each pose.
    **scan_kwargs
        Further keyword arguments of :func:`scan_plot` (densities, noise, occlusion).

    Returns
    -------
    Survey
        ``clouds[k]`` is in scanner ``k``'s frame; ``true_transforms[k]`` maps it back
        into the plot frame.
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
