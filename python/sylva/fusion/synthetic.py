# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""A synthetic terrestrial scan of a scene, for tests and examples."""

from __future__ import annotations

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..shots import Shots


def synthetic_scan(scene: PointCloud, origins=(10.0, 10.0, 1.5), resolution_deg: float = 0.25,
                   max_zenith_deg: float = 130.0, target_radius: float = 0.03,
                   terrain_slope: float = 0.05, min_range: float = 0.1, max_range: float = 200.0):
    """Scan a synthetic scene from the ground as :func:`sylva.synthetic.als_flight` flies it.

    Every scene point is a sphere of ``target_radius`` (points of class 2,
    the scene's ground, are ignored), and the ground is the analytic
    terrain of :func:`sylva.synthetic.terrain_height`. Pulses leave the
    scanner on a regular zenith and azimuth grid as thin rays; a ray stops
    at the first sphere or the terrain, whichever it meets first, and gives
    one echo there, or none if it leaves the scene. A layer of spheres of
    ``n`` per m³ is then a turbid medium of plant area density ``2 π r² n``
    (spherical leaf angles) for both scanners, so their estimates can be
    checked against the same foliage. (:func:`sylva.synthetic.scan` instead
    lets the points in each pulse's angular cell be hit, which suits
    geometry but not densities.)

    Parameters
    ----------
    scene
        From :func:`sylva.synthetic.forest` or
        :func:`sylva.synthetic.crown_forest`.
    origins
        One scanner position ``(x, y, z)`` or an ``(n, 3)`` array.
    resolution_deg
        Angular step in zenith and azimuth (degrees).
    max_zenith_deg
        Pulses are fired from straight up to this zenith.
    target_radius
        Sphere radius (m); use the ``target_radius`` of the flight.
    terrain_slope
        Slope of the terrain (``terrain_slope`` of the flight).
    min_range, max_range
        Echoes nearer than ``min_range`` are ignored; rays are followed to
        ``max_range`` (m).

    Returns
    -------
    Shots or list of Shots
        One per origin, every pulse fired (misses included), with the
        echo attributes ``classification`` (2 for the terrain) and
        ``tree_id`` when the scene has it.

    Raises
    ------
    ValueError
        For settings out of range or an origin below the terrain.
    """
    o = np.asarray(origins, dtype=float)
    single = o.ndim == 1
    o = np.ascontiguousarray(o.reshape(-1, 3))
    out = _core.fusion_scan_spheres(np.ascontiguousarray(scene.xyz), scene.attrs, o, float(resolution_deg),
                                    float(max_zenith_deg), float(target_radius), float(terrain_slope),
                                    float(min_range), float(max_range))
    shots = [Shots._from_core(d) for d in out]
    return shots[0] if single else shots
