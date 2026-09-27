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

from dataclasses import dataclass

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .shots import Shots

__all__ = ["terrain_height", "tree", "forest", "scan", "leaf_area", "als_flight", "ALSFlight",
           "DEFAULT_TREES", "LEAF_RADIUS"]

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
    xs, ys = np.ascontiguousarray(x).ravel(), np.ascontiguousarray(y).ravel()
    z = _core.synthetic_terrain_height(xs, ys, float(slope)).reshape(x.shape)
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
    xyz, attrs = _core.synthetic_tree(float(x), float(y), float(dbh), float(height), float(z0),
                                      int(n_branches), int(leaf_points), int(seed))
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
    rows = [tuple(float(v) for v in t) for t in trees]
    xyz, attrs = _core.synthetic_forest(rows, float(size), int(ground_points), float(margin),
                                        int(seed))
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
    d = _core.synthetic_scan(cloud.xyz, cloud.attrs, o, float(resolution_deg),
                             float(max_zenith_deg), int(max_echoes), float(echo_separation))
    return Shots._from_core(d)


@dataclass
class ALSFlight:
    """A simulated airborne survey, from :func:`als_flight`.

    Parameters
    ----------
    points
        The returns, in firing order, with the LAS attributes ``gps_time``,
        ``return_number``, ``number_of_returns``, ``scan_angle``,
        ``intensity``, ``point_source_id`` (the flight line, from 1) and
        ``classification`` (what was hit: 2 ground, otherwise the class of
        the scene point), plus ``tree_id`` when the scene has it.
    trajectory
        The platform's path while the laser is on, as a table of equal
        length arrays: ``time`` (s, the same clock as ``gps_time``), ``x``,
        ``y``, ``z`` (m), ``roll``, ``pitch``, ``heading`` (degrees) and
        ``line``. Turns between lines are not included.
    n_pulses
        Pulses fired, including those clipped away or with no return.
    """

    points: PointCloud
    trajectory: dict[str, np.ndarray]
    n_pulses: int

    def sensor_positions(self, gps_time) -> np.ndarray:
        """Where the scanner was when each pulse left.

        Parameters
        ----------
        gps_time
            Times, e.g. ``points.attrs["gps_time"]``.

        Returns
        -------
        numpy.ndarray
            ``(N, 3)`` positions, interpolated linearly within the flight
            line being flown at each time. This is exact (to rounding), since
            lines are straight and flown at constant speed; NaN for times
            outside every line. A return lies on the ray from this position
            through the return, so ``(point - position)`` normalised is the
            beam direction.
        """
        t = self.trajectory
        times = np.ascontiguousarray(np.asarray(gps_time, dtype=np.float64).ravel())
        return _core.synthetic_trajectory_positions(
            t["time"], t["x"], t["y"], t["z"], np.asarray(t["line"], dtype=np.uint16), times)

    def write_tiles(self, directory, size: float = 50.0, format: str = "laz", origin=None,
                    epsg: int | None = None):
        """Write the returns as square LAS/LAZ tiles.

        Parameters
        ----------
        directory
            Where to write ``<xmin>_<ymin>.<ext>`` files.
        size
            Tile side (m).
        format
            ``"las"`` or ``"laz"``.
        origin
            ``(x, y)`` of a tile corner; by default the minimum snapped down
            to a multiple of ``size``.
        epsg
            EPSG code to record in the files.

        Returns
        -------
        sylva.als.Catalog
            The tiles written (point format 6, 1 mm quantisation).
        """
        from . import als
        return als.write_tiles(self.points, directory, size, origin=origin, format=format,
                               epsg=epsg)


def als_flight(forest: PointCloud, altitude: float = 80.0, speed: float = 10.0,
               line_spacing: float = 40.0, heading: float = 0.0,
               scan_pattern: str = "oscillating", scan_angle: float = 30.0,
               scan_rate: float = 80.0, pulse_rate: float = 50_000.0,
               beam_divergence: float = 1.0, footprint_samples: int = 7, max_returns: int = 5,
               min_separation: float = 1.0, detection_threshold: float = 0.1,
               range_noise: float = 0.02, attitude=(0.5, 0.3, 0.2), target_radius: float = 0.03,
               terrain_slope: float = 0.05, bounds=None, clip: bool = True, margin: float = 10.0,
               turn_time: float = 10.0, start_time: float = 0.0, trajectory_rate: float = 100.0,
               seed: int = 0) -> ALSFlight:
    """Fly a simulated airborne laser scanner over a synthetic scene.

    The platform flies parallel flight lines, alternating in direction, at
    constant height and speed; a mirror sweeps the beam across the track and
    pulses leave at ``pulse_rate``. A pulse is a cone of the beam divergence
    sampled by ``footprint_samples`` equal-energy sub-beams, each stopping
    at the first scene point it passes within ``target_radius`` of, or at the
    terrain. Hits closer in range than ``min_separation`` merge into one
    return (at their energy-weighted mean range), returns carrying less
    than ``detection_threshold`` of the pulse energy are lost, and at most
    ``max_returns`` are kept, nearest first.

    The ground is the analytic surface of :func:`terrain_height` (with
    ``terrain_slope``), not the scene's ground points (``classification ==
    2``), which are ignored; so the true DTM is known exactly. Use
    :func:`forest` for a scene; larger scenes are a ``forest`` with more
    trees over a larger ``size``.

    Geometry and timing, which the ray-based methods rely on:

    - Frames: map x east, y north, z up; body x forward, y right, z down.
      Heading is clockwise from north; roll positive with the right wing
      down; pitch positive nose up. A body vector ``b`` points along
      ``M Rz(heading) Ry(pitch) Rx(roll) b`` in the map, with ``M`` taking
      north-east-down to east-north-up.
    - The beam leaves the scanner (at the trajectory position; no lever arm
      or boresight) along ``b = (0, sin a, cos a)`` for mirror angle ``a``
      (positive to the right). Every return of a pulse lies on that axis:
      ``point = position(gps_time) + range * direction``; see
      :meth:`ALSFlight.sensor_positions`.
    - ``scan_angle`` is the LAS scan angle ``a - roll`` (degrees from the
      vertical, including roll); the mirror angle is ``scan_angle + roll``.
    - Line ``k`` (from 0) starts at ``start_time + k * (T + turn_time)``,
      ``T`` being the time to fly a line, and fires pulses at
      ``1 / pulse_rate`` intervals from its start; every return of a pulse
      has the pulse's ``gps_time``. Roll, pitch and heading are sinusoids
      about level flight along the line (periods 7.3, 11.1 and 13.7 s,
      amplitudes ``attitude``, phases from ``seed``).
    - ``"oscillating"``: the mirror sweeps left to right and back at a
      constant angular rate, ``scan_rate`` sweeps per second (a zigzag on
      the ground). ``"rotating"``: a polygon mirror sweeps left to right
      only and jumps back (parallel scan lines).

    Parameters
    ----------
    forest
        The scene, e.g. :func:`forest`: points, with ``classification`` (2
        is ignored; 4 leaf and 5 wood set the reflectance) and optionally
        ``tree_id``, carried to the returns.
    altitude
        Flying height above z = 0 (m).
    speed
        Ground speed (m/s).
    line_spacing
        Distance between flight lines (m). The lines are centred on the
        area; there are ``floor(width / line_spacing) + 1`` of them.
    heading
        Direction of the first line, degrees clockwise from north.
    scan_pattern : {"oscillating", "rotating"}
        Mirror motion.
    scan_angle
        Largest mirror angle either side of nadir (degrees, at most 75).
    scan_rate
        Sweeps across the track per second.
    pulse_rate
        Pulses per second.
    beam_divergence
        Full divergence (mrad): the footprint radius at range R is
        ``R * beam_divergence / 2000`` m.
    footprint_samples : {1, 7, 19}
        Sub-beams per pulse.
    max_returns
        Returns kept per pulse (1-15).
    min_separation
        Range resolution (m): hits closer than this form one return.
    detection_threshold
        Share of the pulse energy a return needs (0-1).
    range_noise
        Standard deviation of Gaussian noise on each return's range (m).
    attitude
        Amplitudes of roll, pitch and heading (degrees, at most 10).
    target_radius
        Radius (m) of the sphere each scene point stands for. The leaf
        points of :func:`tree` are about 4 cm apart.
    terrain_slope
        Slope of the terrain (at most 0.5); 0.05 is that of :func:`forest`.
    bounds
        ``(xmin, ymin, xmax, ymax)`` to cover; the scene's extent if None.
    clip
        Keep only returns inside ``bounds``.
    margin
        Distance (m) flown beyond the area at each end of a line.
    turn_time
        Seconds between lines, laser off.
    start_time
        GPS time of the first pulse (s).
    trajectory_rate
        Trajectory samples per second.
    seed
        Random seed (NumPy's ``default_rng``): the same seed gives the same
        flight, whatever the number of threads.

    Returns
    -------
    ALSFlight

    Raises
    ------
    ValueError
        For settings out of range, or an altitude that would take the
        platform through the scene or the terrain.

    Examples
    --------
    >>> from sylva import synthetic
    >>> flight = synthetic.als_flight(synthetic.forest(), pulse_rate=20_000)  # doctest: +SKIP
    >>> cat = flight.write_tiles("tiles/", size=10.0)                         # doctest: +SKIP
    """
    b = None if bounds is None else tuple(float(v) for v in bounds)
    if b is not None and len(b) != 4:
        raise ValueError(f"bounds must be (xmin, ymin, xmax, ymax), got {bounds}")
    if len(tuple(attitude)) != 3:
        raise ValueError(f"attitude must be (roll, pitch, heading) amplitudes, got {attitude}")
    xyz, attrs, traj, n = _core.synthetic_als_flight(
        forest.xyz, forest.attrs, float(altitude), float(speed), float(line_spacing),
        float(heading), str(scan_pattern), float(scan_angle), float(scan_rate), float(pulse_rate),
        float(beam_divergence), int(footprint_samples), int(max_returns), float(min_separation),
        float(detection_threshold), float(range_noise), tuple(float(a) for a in attitude),
        float(target_radius), float(terrain_slope), b, bool(clip), float(margin),
        float(turn_time), float(start_time), float(trajectory_rate), int(seed))
    return ALSFlight(PointCloud(xyz, attrs), dict(traj), int(n))
