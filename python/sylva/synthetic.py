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
           "stand", "crown_forest", "forest_trees",
           "forest_epochs", "ForestEpochs", "DEFAULT_TREES", "LEAF_RADIUS", "waveforms"]

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


def _rpy(roll: float, pitch: float, yaw: float, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    """``Rz(yaw) Ry(pitch) Rx(roll)`` (degrees) followed by a translation."""
    r, p, y = np.radians([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    m[:3, 3] = t
    return m


@dataclass
class ForestEpochs:
    """Two scanned epochs of one synthetic plot and their truth; see
    :func:`forest_epochs`.

    Attributes
    ----------
    clouds
        The echoes of all scans of each epoch, with ``classification`` (2
        ground, 4 leaf, 5 wood), ``tree_id`` (0 for ground; the same tree
        keeps its id in both epochs), ``branch_id`` (limb index within the
        tree, -1 on stems and ground), ``scan_id`` and the scan attributes of
        :meth:`sylva.Shots.to_pointcloud`. Epoch 2 is in its displaced frame.
    shots
        The pulses of each epoch, misses included, in the same frames.
    trees
        Per epoch, a table (dict of equal-length arrays) of the trees
        standing: ``tree_id``, ``x``, ``y``, ``z0`` (terrain at the stem),
        ``dbh`` (1.3 m above ``z0``), ``height``, ``stem_volume``,
        ``wood_volume`` (stem and limbs), ``leaf_area`` and ``n_limbs``. Both
        tables are in the frame of epoch 1.
    changes
        The known changes, one dict each, with ``kind`` and ``tree_id``:
        ``growth`` (``d_dbh``, ``d_height``, ``d_stem_volume``, ``shift``),
        ``death`` and ``recruit`` (``x``, ``y``, ``dbh``, ``height``,
        ``wood_volume``; a recruit next to a felled tree has ``replaces``),
        ``branch_removed`` (``branch_id``, ``volume``, ``leaf_area``, base and
        tip coordinates) and ``foliage_thinned`` (per tree ``leaf_area`` and
        ``discs``; for ``tree_id`` -1 the box, ``fraction`` and total
        ``leaf_area``).
    transform
        ``(4, 4)`` transform taking epoch 2 onto epoch 1: the registration
        an alignment should recover.
    origins
        Scanner positions of each epoch, in its delivered frame.
    range_noise
        Range noise (m) of each epoch.
    """

    clouds: list
    shots: list
    trees: list
    changes: list
    transform: np.ndarray
    origins: list
    range_noise: tuple

    def of_kind(self, kind: str) -> list[dict]:
        """The changes of one kind.

        Parameters
        ----------
        kind
            ``growth``, ``death``, ``recruit``, ``branch_removed`` or
            ``foliage_thinned``.

        Returns
        -------
        list of dict
        """
        return [c for c in self.changes if c["kind"] == kind]


def forest_epochs(n_trees: int = 16, size: float = 30.0, deaths: int = 2, recruits: int = 2,
                  replaced: int = 1, small_increments: int = 2,
                  dbh_increment: tuple[float, float] = (0.012, 0.004),
                  height_increment: tuple[float, float] = (0.6, 0.2), branch_removals: int = 1,
                  foliage_box=None, foliage_fraction: float = 0.5, tree_shift: float = 0.0,
                  offset=None, range_noise: tuple[float, float] = (0.003, 0.005),
                  scan_positions=None, scan_jitter: float = 0.5, resolution_deg: float = 0.25,
                  max_echoes: int = 1, ground_density: float = 60.0, min_spacing: float = 2.5,
                  seed: int = 0) -> ForestEpochs:
    """Two epochs of one synthetic plot with known changes, each scanned.

    Every tree is a fixed structure (a tapered stem, limbs at fixed heights,
    leaf discs at fixed offsets from the limb tips) sampled afresh in each
    epoch. Between the epochs survivors grow by a known DBH and height
    increment (a uniform layer of wood along the stem, a longer leader and
    proportionally longer limbs); ``deaths`` trees are gone, ``recruits`` new
    small trees stand (``replaced`` of them 0.3 to 0.6 m from a dead stem, as
    after felling), ``branch_removals`` survivors lose their largest limb,
    ``small_increments`` survivors grow by only 0.5 mm in DBH and 1 cm in
    height, and ``foliage_fraction`` of the leaf discs in ``foliage_box``
    disappear. Each epoch is scanned from ``scan_positions`` with
    :func:`scan` (so occlusion is real) and Gaussian range noise; the epoch-2
    scanners stand ``scan_jitter`` m (one sigma) from the epoch-1 ones, and
    epoch 2 is delivered in a frame displaced by ``offset``, as an
    independently registered revisit would be.

    Parameters
    ----------
    n_trees
        Trees in epoch 1, with DBH drawn from 0.15 to 0.5 m and height from
        an allometry (about 12 to 20 m).
    size
        Side of the square plot (m); stems stand at least 2 m inside it.
    deaths, recruits, replaced
        Trees lost, trees gained, and of those the felled-and-replaced pairs.
    small_increments
        Survivors with increments far below what a scan can detect.
    dbh_increment, height_increment
        Mean and standard deviation (m) of the survivors' increments.
    branch_removals
        Survivors that lose their largest limb and its leaves.
    foliage_box
        ``(xmin, ymin, zmin, xmax, ymax, zmax)`` of the thinned foliage;
        by default the crown of one survivor.
    foliage_fraction
        Share of the leaf discs in the box that disappear.
    tree_shift
        Horizontal displacement (m, random direction) of every survivor's
        stem between the epochs, to test matching; 0 keeps stems in place.
    offset
        ``(4, 4)`` rigid transform from the true frame to the frame epoch 2
        is delivered in; by default a rotation of 1.5 degrees about z with
        tilts of 0.1 and -0.15 degrees, and a shift of (0.8, -0.5, 0.15) m.
    range_noise
        Range noise (m, one sigma) of epoch 1 and epoch 2.
    scan_positions
        Scanner positions as ``(n, 2)`` fractions of the plot side (1.5 m
        above ground); by default the centre and four positions at 0.2 and
        0.8. Positions within 1 m of a stem are moved away from it.
    scan_jitter
        Standard deviation (m) of the epoch-2 scanners' offsets.
    resolution_deg
        Angular step of the scans (degrees).
    max_echoes
        Echoes recorded per pulse. :func:`scan` places every echo of a pulse
        along the direction of its last one, so with more than one echo the
        first echoes at a stem's silhouette are displaced sideways and stem
        circles come out a few millimetres wide; one echo keeps every point
        on its surface.
    ground_density
        Terrain points per m² before scanning.
    min_spacing
        Least distance between stems (m).
    seed
        Random seed; the same seed gives the same epochs.

    Returns
    -------
    ForestEpochs

    Raises
    ------
    ValueError
        If the counts are inconsistent (more deaths than trees, more replaced
        than deaths or recruits) or the trees cannot be placed.

    Examples
    --------
    >>> ep = synthetic.forest_epochs(seed=1)
    >>> ref, new = ep.clouds
    >>> [c["tree_id"] for c in ep.of_kind("death")]
    """
    offset = _rpy(0.1, -0.15, 1.5, (0.8, -0.5, 0.15)) if offset is None else np.asarray(offset, dtype=float)
    if offset.shape != (4, 4):
        raise ValueError(f"offset must be a (4, 4) matrix, got shape {offset.shape}")
    if scan_positions is None:
        scan_positions = [(0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    pos = [tuple(float(v) for v in p) for p in np.asarray(scan_positions, dtype=float).reshape(-1, 2)]
    box = None if foliage_box is None else [float(v) for v in np.asarray(foliage_box, dtype=float).reshape(6)]
    for name, v in (("n_trees", n_trees), ("deaths", deaths), ("recruits", recruits), ("replaced", replaced),
                    ("small_increments", small_increments), ("branch_removals", branch_removals)):
        if int(v) < 0:
            raise ValueError(f"{name} must be >= 0, got {v}")
    noise = tuple(float(v) for v in range_noise)
    d = _core.change_forest_epochs(int(n_trees), float(size), float(min_spacing), int(deaths), int(recruits),
                                   int(replaced), int(small_increments),
                                   tuple(float(v) for v in dbh_increment),
                                   tuple(float(v) for v in height_increment), int(branch_removals), box,
                                   float(foliage_fraction), float(tree_shift), np.ascontiguousarray(offset),
                                   noise, pos, float(scan_jitter), float(resolution_deg), int(max_echoes),
                                   float(ground_density), int(seed))
    return ForestEpochs(
        clouds=[PointCloud(xyz, attrs) for xyz, attrs in d["clouds"]],
        shots=[Shots._from_core(s) for s in d["shots"]],
        trees=list(d["trees"]),
        changes=list(d["changes"]),
        transform=d["transform"],
        origins=list(d["origins"]),
        range_noise=noise,
    )


def waveforms(shots: Shots, gps_time=None, pulse_width: float = 1.5, interval: float = 1.0,
              n_samples: int = 120, margin: float = 3.0, start_range: float | None = None,
              background: float = 10.0, noise: float = 1.0, digitise: bool = True, bits: int = 16,
              amplitude: float = 100.0, metres_per_ns: float = 0.299792458 / 2.0, seed: int = 0):
    """Full waveforms of pulses whose targets are known.

    The received waveform is the system pulse, a Gaussian of standard
    deviation ``pulse_width``, convolved with the targets along the beam
    (Wagner et al. 2006): each echo of ``shots`` is a target at its range,
    with the peak amplitude of the echo attribute ``amplitude`` (for a point
    target) and the depth of the echo attribute ``extent`` (standard
    deviation along the beam, m; 0 if absent). A target of depth ``e``
    gives a Gaussian echo of width ``s = sqrt(pulse_width^2 +
    (e / metres_per_ns)^2)`` and, its energy being kept, peak
    ``amplitude * pulse_width / s``. The echoes are added to a constant
    ``background`` with Gaussian ``noise`` and, with ``digitise``, rounded
    and clipped to ``0 .. 2^bits - 1`` as a digitiser would.

    Parameters
    ----------
    shots
        Pulses and their targets; a shot without an echo gives a waveform
        of background and noise only.
    gps_time
        Time of each shot; the shot index by default.
    pulse_width
        Standard deviation of the system pulse (ns). A 3.5 ns full width at
        half maximum is 1.5 ns.
    interval
        Sampling interval (ns).
    n_samples
        Samples per waveform.
    margin
        Range (m) before a shot's first echo at which its record starts
        (a shot without an echo starts at the origin).
    start_range
        Range (m) at which every record starts, instead of ``margin``.
    background, noise
        Background level and noise standard deviation (sample units).
    digitise
        Round and clip the samples.
    bits
        Digitiser resolution.
    amplitude
        Peak of echoes without an ``amplitude`` attribute.
    metres_per_ns
        Range per nanosecond of round-trip time (m/ns).
    seed
        Seed of the noise.

    Returns
    -------
    waveforms : sylva.waveform.Waveforms
        One returning waveform per shot, anchored at its origin.
    truth : sylva.waveform.Echoes
        The echoes they contain: time, peak amplitude and width after the
        convolution, position and range.
def stand(n_trees: int, size: float = 100.0, min_spacing: float = 4.0, heights=(10.0, 25.0),
          seed: int = 0) -> list[tuple[float, float, float, float]]:
    """Random trees for :func:`forest`: a stand of a given density.

    Stems are drawn uniformly in the ``size`` m square and rejected when
    closer than ``min_spacing`` to one already placed; heights are uniform
    in ``heights`` and ``dbh = 0.1 + 0.015 * height``.

    Parameters
    ----------
    n_trees
        Trees to place.
    size
        Side of the square (m).
    min_spacing
        Smallest distance between stems (m).
    heights
        ``(low, high)`` tree heights (m), ``1 < low <= high``.
    seed
        Random seed (NumPy's ``default_rng`` stream).

    Returns
    -------
    list of (x, y, dbh, height)

    Raises
    ------
    ValueError
        For non-positive widths, intervals or sample counts, negative noise,
        or ``gps_time`` of the wrong length.
    """
    from .waveform import Echoes, Waveforms

    t = None if gps_time is None else np.ascontiguousarray(gps_time, dtype=np.float64).ravel()
    w, e = _core.synthetic_waveforms(shots._to_core(), t, float(pulse_width), float(interval),
                                     int(n_samples), float(margin),
                                     None if start_range is None else float(start_range),
                                     float(background), float(noise), bool(digitise), int(bits),
                                     float(amplitude), float(metres_per_ns), int(seed))
    return (Waveforms._from_core(w),
            Echoes(e["waveform"], e["time"], e["amplitude"], e["width"], e["xyz"], e["range"]))
        For bad settings, or if the trees cannot be placed that far apart.
    """
    lo, hi = (float(v) for v in heights)
    return [tuple(t) for t in _core.synthetic_stand(int(n_trees), float(size), float(min_spacing),
                                                     (lo, hi), int(seed))]


def crown_forest(trees, size: float = 100.0, shape: str = "ellipsoid",
                 crown_radius: float = 0.25, crown_length: float = 0.5, density: float = 40.0,
                 ground_points: int = 40000, margin: float = 4.0, seed: int = 0) -> PointCloud:
    """A scene of trees with solid crowns, for airborne lidar.

    Each tree ``(x, y, dbh, height)`` (e.g. from :func:`stand`) has a stem
    (points on a cylinder of diameter ``dbh``, ``classification`` 5) from
    the terrain up to its crown, and a crown filled uniformly with
    ``density`` leaf points per m³ (``classification`` 4): an ellipsoid of
    revolution or a cone of radius ``crown_radius * height`` and length
    ``crown_length * height``, its top ``height`` above the terrain at the
    stem. The trees of :func:`forest` carry their leaves in clusters at the
    ends of a few limbs, which suits terrestrial scanning; these have the
    closed outline that airborne tree detection assumes, and a projected
    crown area of ``pi * (crown_radius * height) ** 2``.

    Parameters
    ----------
    trees
        ``(x, y, dbh, height)`` per tree.
    size, ground_points, margin
        Terrain as in :func:`forest` (slope 0.05).
    shape : {"ellipsoid", "cone"}
        Crown shape.
    crown_radius, crown_length
        Crown radius and length as fractions of the tree height
        (``crown_length`` at most 1).
    density
        Leaf points per m³ of crown. With the default ``target_radius`` of
        :func:`als_flight` (3 cm), 40 stops about 10 % of a beam's energy
        per metre of crown.
    seed
        Random seed; the crown of tree ``i`` uses ``seed + i + 1``.

    Returns
    -------
    PointCloud
        With ``classification`` (2 ground, 4 leaf, 5 wood) and ``tree_id``
        (0 for ground, then 1.. in list order); :func:`forest_trees` gives
        the truth.

    Raises
    ------
    ValueError
        For a non-positive radius or density, a crown length outside
        (0, 1], or trees without finite positions and positive heights.
    """
    rows = [tuple(float(v) for v in t) for t in trees]
    xyz, attrs = _core.synthetic_crown_forest(rows, str(shape), float(crown_radius),
                                              float(crown_length), float(density), float(size),
                                              int(ground_points), float(margin), int(seed))
    return PointCloud(xyz, attrs)


def forest_trees(forest: PointCloud, terrain_slope: float = 0.05) -> dict:
    """The true trees of a scene from :func:`forest`, to check tree
    detection and crown delineation against.

    Parameters
    ----------
    forest
        The scene, from :func:`forest` or :func:`crown_forest`, with its
        ``tree_id`` (0 for ground).
    terrain_slope
        Slope of the terrain under it (0.05 for :func:`forest`, and the
        ``terrain_slope`` of :func:`als_flight`).

    Returns
    -------
    dict
        Columns in order of ``tree_id``: ``tree_id``; ``stem_x``,
        ``stem_y`` (mean of the wood points within 1 m of the tree's lowest
        point); ``top_x``, ``top_y``, ``top_z`` (the highest point);
        ``height`` (of the top above the terrain beneath it);
        ``crown_area`` (m², convex hull of all the tree's points seen from
        above) and ``crowns`` (those hulls, ``(k, 2)`` each).

    Raises
    ------
    ValueError
        If the scene has no ``tree_id``.
    """
    return dict(_core.synthetic_scene_trees(forest.xyz, forest.attrs, float(terrain_slope)))
