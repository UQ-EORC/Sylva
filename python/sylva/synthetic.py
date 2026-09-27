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

from . import _core
from .pointcloud import PointCloud
from .shots import Shots

__all__ = ["terrain_height", "tree", "forest", "scan", "leaf_area", "DEFAULT_TREES", "LEAF_RADIUS"]

#: Radius (m) of the leaf discs of :func:`tree`; each disc is 12 points.
LEAF_RADIUS = 0.08

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
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    z = _core.synthetic_terrain_height(np.ascontiguousarray(x).ravel(), np.ascontiguousarray(y).ravel(),
                                       float(slope)).reshape(x.shape)
    return z[()] if z.ndim == 0 else z


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
    xyz, attrs = _core.synthetic_tree(float(x), float(y), float(dbh), float(height), float(z0), int(n_branches),
                                      int(leaf_points), int(seed))
    return PointCloud(xyz, attrs)


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
    return _core.synthetic_leaf_area(np.asarray(cloud.attrs["classification"], dtype=np.float64))


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
    trees = DEFAULT_TREES if trees is None else trees
    xyz, attrs = _core.synthetic_forest([tuple(float(v) for v in t) for t in trees], float(size), int(ground_points),
                                        float(margin), int(seed))
    return PointCloud(xyz, attrs)


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
    o = tuple(float(v) for v in origin)
    return Shots._from_core(_core.synthetic_scan(cloud.xyz, cloud.attrs, o, float(resolution_deg),
                                                 float(max_zenith_deg), int(max_echoes), float(echo_separation)))
