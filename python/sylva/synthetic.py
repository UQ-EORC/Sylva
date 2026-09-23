# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Small synthetic scenes for examples, tutorials and tests.

Nothing here is a forest model; the shapes are simple enough that the right
answer is known (stem positions, diameters, heights, leaf area), which is what
an example needs.
"""

from __future__ import annotations

import numpy as np

from .pointcloud import PointCloud
from .shots import Shots

__all__ = ["terrain_height", "tree", "forest", "scan", "leaf_area", "DEFAULT_TREES", "LEAF_RADIUS"]

#: Radius (m) of the leaf discs of :func:`tree`; each disc is 12 points.
LEAF_RADIUS = 0.08
_POINTS_PER_LEAF = 12

#: ``(x, y, dbh, height)`` of the trees in :func:`forest`.
DEFAULT_TREES = [(5.0, 5.0, 0.30, 12.0), (14.0, 6.0, 0.20, 9.0), (8.0, 15.0, 0.45, 15.0),
                 (15.5, 15.0, 0.25, 11.0)]


def terrain_height(x, y, slope: float = 0.05) -> np.ndarray:
    """Ground elevation of the synthetic scenes: a gentle slope with a ripple.

    Parameters
    ----------
    x, y
        Coordinates (m).
    slope
        Rise per metre along x.

    Returns
    -------
    numpy.ndarray
        ``slope * x + 0.2 * sin(y / 3)``.
    """
    return slope * np.asarray(x) + 0.2 * np.sin(np.asarray(y) / 3)


def _cylinder(rng, start, axis, length, r0, r1, n, noise=0.003):
    """Points on a tapered cylinder surface."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = rng.uniform(0, 1, n)
    a = rng.uniform(0, 2 * np.pi, n)
    r = r0 + (r1 - r0) * t + rng.normal(0, noise, n)
    return (np.asarray(start) + np.outer(t * length, axis)
            + (np.cos(a) * r)[:, None] * u + (np.sin(a) * r)[:, None] * v)


def tree(x: float = 0.0, y: float = 0.0, dbh: float = 0.3, height: float = 12.0, z0: float = 0.0,
         n_branches: int = 6, leaf_points: int = 18000, seed: int = 0) -> PointCloud:
    """One tree: a tapered stem, ``n_branches`` limbs in the upper half and
    small flat leaf discs around the limb ends.

    The ``classification`` attribute is 5 for wood and 4 for leaves.

    Parameters
    ----------
    x, y, z0
        Stem base position.
    dbh
        Diameter at the base (m); the stem tapers to a quarter of it.
    height
        Tree height (m).
    n_branches
        Limbs.
    leaf_points
        Approximate number of leaf points (12 per leaf disc).
    seed
        Random seed.

    Returns
    -------
    PointCloud
    """
    rng = np.random.default_rng(seed)
    r = dbh / 2
    stem_points = int(2500 * height)
    parts = [_cylinder(rng, [x, y, z0], [0, 0, 1], height * 0.9, r * 1.05, r * 0.25, stem_points)]
    leaves = []
    for b in range(n_branches):
        h = height * (0.45 + 0.45 * (b + 0.5) / n_branches)
        az = 2.4 * b + rng.uniform(-0.3, 0.3)
        length = height * rng.uniform(0.16, 0.26)
        axis = [np.cos(az), np.sin(az), rng.uniform(0.25, 0.6)]
        rb = r * 0.3 * (1 - 0.5 * b / n_branches)
        limb = _cylinder(rng, [x, y, z0 + h], axis, length, rb, rb * 0.3, int(1500 * length))
        parts.append(limb)
        tip = limb[np.argmax(np.linalg.norm(limb - [x, y, z0 + h], axis=1))]
        # Leaves: small randomly oriented discs scattered around the limb tip.
        k = leaf_points // n_branches
        centres = tip + rng.normal(0, height * 0.07, (k // _POINTS_PER_LEAF, 3))
        for c in centres:
            normal = rng.normal(size=3)
            normal /= np.linalg.norm(normal)
            e1 = np.cross(normal, [0, 0, 1.0])
            e1 /= np.linalg.norm(e1) + 1e-12
            e2 = np.cross(normal, e1)
            rad = LEAF_RADIUS * np.sqrt(rng.uniform(0, 1, _POINTS_PER_LEAF))
            ang = rng.uniform(0, 2 * np.pi, _POINTS_PER_LEAF)
            leaves.append(c + (rad * np.cos(ang))[:, None] * e1 + (rad * np.sin(ang))[:, None] * e2)
    wood = np.vstack(parts)
    leaf = np.vstack(leaves) if leaves else np.zeros((0, 3))
    cls = np.concatenate([np.full(len(wood), 5), np.full(len(leaf), 4)]).astype(np.uint8)
    return PointCloud(np.vstack([wood, leaf]), {"classification": cls})


def leaf_area(cloud: PointCloud) -> float:
    """True one-sided leaf area of a synthetic cloud.

    Parameters
    ----------
    cloud
        From :func:`tree` or :func:`forest`.

    Returns
    -------
    float
        Area (m²) of the leaf discs, counted from ``classification == 4``
        points; what leaf-area estimates can be checked against.
    """
    n_leaves = np.sum(cloud.attrs["classification"] == 4) / _POINTS_PER_LEAF
    return float(n_leaves * np.pi * LEAF_RADIUS**2)


def forest(trees=None, size: float = 20.0, ground_points: int = 40000, margin: float = 4.0,
           seed: int = 0) -> PointCloud:
    """Sloped terrain with a few trees standing on it (z is not normalised).
    The ground extends ``margin`` m beyond the ``size`` m square so that no
    crown overhangs the edge of the terrain.

    ``trees`` is a list of ``(x, y, dbh, height)``, by default
    :data:`DEFAULT_TREES`. Attributes: ``classification`` (2 ground, 4 leaf,
    5 wood) and ``tree_id`` (0 for ground, then 1.. in list order) as ground
    truth to compare results against.

    Parameters
    ----------
    trees
        ``(x, y, dbh, height)`` per tree.
    size
        Side of the plot square (m).
    ground_points
        Points on the terrain.
    margin
        Terrain beyond the plot edge (m).
    seed
        Random seed.

    Returns
    -------
    PointCloud
    """
    rng = np.random.default_rng(seed)
    trees = DEFAULT_TREES if trees is None else trees
    xy = rng.uniform(-margin, size + margin, (ground_points, 2))
    z = terrain_height(xy[:, 0], xy[:, 1]) + rng.normal(0, 0.01, ground_points)
    ground = np.column_stack([xy, z])
    clouds = [PointCloud(ground, {"classification": np.full(ground_points, 2, np.uint8),
                                  "tree_id": np.zeros(ground_points, np.int32)})]
    for i, (x, y, dbh, h) in enumerate(trees, start=1):
        t = tree(x, y, dbh, h, z0=float(terrain_height(x, y)), seed=seed + i)
        clouds.append(t.with_attrs(tree_id=np.full(len(t), i, np.int32)))
    return PointCloud.concatenate(clouds)


def scan(cloud: PointCloud, origin=(10.0, 10.0, 1.5), resolution_deg: float = 0.25,
         max_zenith_deg: float = 130.0, max_echoes: int = 2, echo_separation: float = 0.5) -> Shots:
    """A pseudo terrestrial scan of ``cloud`` from ``origin``.

    Pulses are fired on a regular zenith / azimuth grid. The points falling in
    a pulse's angular cell are its candidate targets: the nearest gives the
    first echo, and further ones at least ``echo_separation`` m apart give up
    to ``max_echoes`` echoes. Cells without a point are pulses with no return,
    which a real scanner's point stream would not show, but which carry the
    free-space information ray tracing needs. Echo attributes are copied from
    the points, so ``classification`` and ``tree_id`` survive.

    Parameters
    ----------
    cloud
        Scene to scan, e.g. :func:`forest`.
    origin
        Scanner position.
    resolution_deg
        Angular step in zenith and azimuth (degrees).
    max_zenith_deg
        Pulses are fired from straight up to this zenith.
    max_echoes
        Echoes per pulse.
    echo_separation
        Minimum range (m) between echoes of one pulse.

    Returns
    -------
    Shots
        One pulse per angular cell, misses included.
    """
    origin = np.asarray(origin, float)
    d = cloud.xyz - origin
    rng_ = np.linalg.norm(d, axis=1)
    zen = np.degrees(np.arccos(np.clip(d[:, 2] / np.maximum(rng_, 1e-12), -1, 1)))
    az = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0
    n_zen = int(round(max_zenith_deg / resolution_deg))
    n_az = int(round(360.0 / resolution_deg))
    iz = np.floor(zen / resolution_deg).astype(np.int64)
    ia = np.minimum(np.floor(az / resolution_deg).astype(np.int64), n_az - 1)
    ok = (iz < n_zen) & (rng_ > 0.1)
    cell = iz * n_az + ia

    # Nearest-first within each cell.
    idx = np.flatnonzero(ok)
    idx = idx[np.lexsort((rng_[idx], cell[idx]))]
    count = np.zeros(n_zen * n_az, np.int64)
    echo_points = []
    last_cell, last_range, taken = -1, 0.0, 0
    for i in idx:
        c = cell[i]
        if c != last_cell:
            last_cell, taken, last_range = c, 0, -np.inf
        if taken < max_echoes and rng_[i] - last_range >= echo_separation:
            echo_points.append(i)
            count[c] += 1
            taken += 1
            last_range = rng_[i]
    echo_points = np.asarray(echo_points, np.int64)

    zc = np.radians((np.arange(n_zen) + 0.5) * resolution_deg)
    ac = np.radians((np.arange(n_az) + 0.5) * resolution_deg)
    Z, A = np.meshgrid(zc, ac, indexing="ij")
    direction = np.column_stack([(np.sin(Z) * np.sin(A)).ravel(), (np.sin(Z) * np.cos(A)).ravel(),
                                 np.cos(Z).ravel()])
    # Aim each pulse that has echoes at its farthest echo so echo_xyz reproduces the points.
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    has = count > 0
    far = echo_points[(start + count - 1)[has]]
    direction[has] = d[far] / rng_[far][:, None]
    return Shots(np.tile(origin, (len(count), 1)), direction, start, count, rng_[echo_points],
                 {k: v[echo_points] for k, v in cloud.attrs.items()})
