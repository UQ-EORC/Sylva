# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The flight trajectory of an airborne survey, and the pulses recovered from it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..shots import Shots

_SBET_SUFFIXES = (".out", ".sbet", ".sbt", ".bin")


def _opt(a, n: int, name: str) -> np.ndarray | None:
    if a is None:
        return None
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64).ravel())
    if len(a) != n:
        raise ValueError(f"trajectory {name} has {len(a)} values for {n} times")
    return a


@dataclass
class Trajectory:
    """The path of the scanner: positions, and optionally attitude, in time.

    Build one with :func:`read_trajectory` or :func:`estimate_trajectory`,
    or from arrays. Samples are sorted by time and a sample repeating an
    earlier time is dropped. It is plain data and pickles.

    Parameters
    ----------
    time
        GPS time of each sample (s), on the clock of the returns'
        ``gps_time`` (see ``time_offset`` of :func:`pulses` otherwise).
    x, y, z
        Sensor position (m), in the frame of the points.
    roll, pitch, heading
        Attitude (degrees; heading clockwise from north), if known.
    rms, n_pulses, line
        For an estimated trajectory: the root mean square distance (m) of
        the pulse lines from each sample, the pulses used, and the flight
        line.

    Raises
    ------
    ValueError
        Fewer than two samples at different times, arrays of different
        lengths, or non-finite values.
    """

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    roll: np.ndarray | None = None
    pitch: np.ndarray | None = None
    heading: np.ndarray | None = None
    rms: np.ndarray | None = None
    n_pulses: np.ndarray | None = None
    line: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = np.ascontiguousarray(np.asarray(self.time, dtype=np.float64).ravel())
        n = len(t)
        names = ("x", "y", "z", "roll", "pitch", "heading", "rms")
        cols = {k: _opt(getattr(self, k), n, k) for k in names}
        for k in ("x", "y", "z"):
            if cols[k] is None:
                raise ValueError(f"trajectory {k} is missing")
        if not (np.isfinite(t).all() and all(np.isfinite(cols[k]).all() for k in ("x", "y", "z"))):
            raise ValueError("trajectory times and positions must be finite")
        extra = {k: None if getattr(self, k) is None else np.asarray(getattr(self, k)).ravel()
                 for k in ("n_pulses", "line")}
        for k, v in extra.items():
            if v is not None and len(v) != n:
                raise ValueError(f"trajectory {k} has {len(v)} values for {n} times")
        order = np.argsort(t, kind="stable")
        keep = np.ones(n, dtype=bool)
        keep[1:] = np.diff(t[order]) > 0
        order = order[keep]
        if len(order) < 2:
            raise ValueError("a trajectory needs at least two samples at different times, "
                             f"got {len(order)}")
        self.time = t[order]
        for k, v in cols.items():
            setattr(self, k, None if v is None else v[order])
        for k, v in extra.items():
            setattr(self, k, None if v is None else v[order])

    def __len__(self) -> int:
        return len(self.time)

    def __repr__(self) -> str:
        kind = "estimated " if self.rms is not None else ""
        att = ", attitude" if self.heading is not None else ""
        return (f"Trajectory({len(self):,} {kind}samples, {self.time[0]:.3f} to "
                f"{self.time[-1]:.3f} s{att})")

    @property
    def xyz(self) -> np.ndarray:
        """``(N, 3)`` positions."""
        return np.column_stack([self.x, self.y, self.z])

    @property
    def has_attitude(self) -> bool:
        """Whether roll, pitch and heading are known."""
        return self.roll is not None and self.pitch is not None and self.heading is not None

    @classmethod
    def from_dict(cls, d: Mapping) -> Trajectory:
        """From a mapping of arrays, such as :attr:`sylva.synthetic.ALSFlight.trajectory`.

        Parameters
        ----------
        d
            ``time``, ``x``, ``y``, ``z`` and optionally ``roll``,
            ``pitch``, ``heading``, ``rms``, ``n_pulses``, ``line``.

        Returns
        -------
        Trajectory
        """
        keys = ("roll", "pitch", "heading", "rms", "n_pulses", "line")
        return cls(d["time"], d["x"], d["y"], d["z"], **{k: d.get(k) for k in keys})

    def _core(self) -> dict:
        d = {"time": self.time, "x": self.x, "y": self.y, "z": self.z}
        if self.has_attitude:
            d.update(roll=self.roll, pitch=self.pitch, heading=self.heading)
        return d

    def positions(self, gps_time, max_gap: float | None = None,
                  time_offset: float = 0.0) -> np.ndarray:
        """Sensor position at each time, linearly interpolated.

        Parameters
        ----------
        gps_time
            Times, e.g. ``cloud.attrs["gps_time"]``.
        max_gap
            Longest gap between samples (s) to interpolate across; ten
            median sample intervals if None.
        time_offset
            Seconds added to ``gps_time`` first.

        Returns
        -------
        numpy.ndarray
            ``(N, 3)``; NaN outside the trajectory or within a longer gap.
        """
        t = _times(gps_time, time_offset)
        return _core.als_trajectory_positions(self._core(), t, _gap(max_gap))

    def attitude(self, gps_time, max_gap: float | None = None,
                 time_offset: float = 0.0) -> np.ndarray:
        """Roll, pitch and heading at each time, linearly interpolated.

        Parameters
        ----------
        gps_time
            Times.
        max_gap, time_offset
            As for :meth:`positions`.

        Returns
        -------
        numpy.ndarray
            ``(N, 3)`` degrees; the heading is interpolated along the shorter
            arc and wrapped to ``[0, 360)``.

        Raises
        ------
        ValueError
            If the trajectory has no attitude.
        """
        if not self.has_attitude:
            raise ValueError("this trajectory has no roll, pitch and heading")
        t = _times(gps_time, time_offset)
        return _core.als_trajectory_attitude(self._core(), t, _gap(max_gap))


def _times(gps_time, time_offset) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(gps_time, dtype=np.float64).ravel() + float(time_offset))


def _gap(max_gap) -> float | None:
    if max_gap is None:
        return None
    if not float(max_gap) >= 0:
        raise ValueError(f"max_gap must be a non-negative number of seconds, got {max_gap}")
    return float(max_gap)


def _as_trajectory(t) -> Trajectory:
    if isinstance(t, Trajectory):
        return t
    if isinstance(t, Mapping):
        return Trajectory.from_dict(t)
    if isinstance(t, (str, Path)):
        return read_trajectory(t)
    raise ValueError("expected a Trajectory, a mapping of arrays or a trajectory file, "
                     f"got {type(t).__name__}")


def week_seconds(gps_time):
    """GPS seconds of the week of adjusted standard GPS times.

    LAS files flagged for it store adjusted standard GPS time (GPS time
    minus 10⁹ s); SBET trajectories count seconds from the start of the GPS
    week. Within one week the two differ by a constant, which
    ``week_seconds(t) - t`` gives, for the ``time_offset`` arguments.

    Parameters
    ----------
    gps_time
        Adjusted standard GPS times (scalar or array).

    Returns
    -------
    float or numpy.ndarray
        ``(t + 1e9) mod 604800``.
    """
    t = np.asarray(gps_time, dtype=np.float64)
    out = np.vectorize(_core.als_week_seconds, otypes=[np.float64])(t)
    return float(out) if out.ndim == 0 else out


def read_trajectory(path: str | Path, format: str | None = None, crs: str | None = None,
                    columns: Sequence[str] | None = None, z_offset: float = 0.0) -> Trajectory:
    """Read a trajectory file: an SBET or a delimited text table.

    **SBET** (``format="sbet"``, or a ``.out``, ``.sbet``, ``.sbt`` or
    ``.bin`` file): the binary "smoothed best estimate of trajectory" of
    Applanix POSPac and most GNSS/INS software, records of 17 little-endian
    doubles (time in GPS seconds of the week, latitude, longitude and
    platform heading in radians, ellipsoidal height, ...). Latitude and
    longitude are projected to ``crs``, which is required; the heading is
    the platform heading minus the wander angle.

    **Text** (any other file): columns separated by commas, semicolons or
    white space, with a header line naming them or with ``columns`` given.
    Names are matched case-insensitively: ``time`` (or ``gps_time``, ``t``,
    ``timestamp``), ``x`` (``easting``), ``y`` (``northing``), ``z``
    (``height``, ``altitude``, ``elevation``), ``roll``, ``pitch`` and
    ``heading`` (``yaw``); other columns are ignored. With ``crs``, ``x``
    and ``y`` are longitude and latitude in degrees and are projected.

    Parameters
    ----------
    path
        The file.
    format : {None, "sbet", "text"}
        None to decide from the suffix.
    crs
        The CRS of the points (e.g. ``"EPSG:32755"``), for geographic
        trajectories.
    columns
        Names of the text columns, in order, instead of a header line.
    z_offset
        Added to every height. SBET heights are ellipsoidal: for points in
        orthometric heights, give minus the geoid separation. A wrong
        height shows as a large ``line_offset_median`` in the report of
        :func:`pulses`.

    Returns
    -------
    Trajectory

    Raises
    ------
    OSError
        If the file cannot be read or is not a whole number of SBET records.
    ValueError
        Without ``crs`` for an SBET, for a projection that is not in metres,
        or for a text table without time, x, y and z columns.
    """
    p = Path(path)
    fmt = format or ("sbet" if p.suffix.lower() in _SBET_SUFFIXES else "text")
    if fmt not in ("sbet", "text"):
        raise ValueError(f"format must be 'sbet' or 'text', got {format!r}")
    if fmt == "sbet":
        if crs is None:
            raise ValueError("SBET positions are latitude and longitude: give crs, the CRS of "
                             "the points (e.g. 'EPSG:32755'), to project them")
        s = _core.als_read_sbet(str(p))
        xyz = _core.als_project_geographic(s["longitude"], s["latitude"], s["height"], str(crs))
        return Trajectory(s["time"], xyz[:, 0], xyz[:, 1], xyz[:, 2] + float(z_offset), s["roll"],
                          s["pitch"], s["heading"])
    names, cols = _core.als_read_table(str(p))
    if columns is not None:
        columns = [str(c) for c in columns]
        if len(columns) != len(cols):
            raise ValueError(f"columns names {len(columns)} columns, the file has {len(cols)}")
        names = columns
    roles: dict[str, np.ndarray] = {}
    for name, col in zip(names, cols, strict=True):
        role = _core.als_column_role(name)
        if role is not None and role not in roles:
            roles[role] = col
    missing = [k for k in ("time", "x", "y", "z") if k not in roles]
    if missing:
        raise ValueError("the trajectory table needs time, x, y and z columns; "
                         f"{', '.join(missing)} not found among {names} (name them with columns=)")
    x, y, z = roles["x"], roles["y"], roles["z"] + float(z_offset)
    if crs is not None:
        xyz = _core.als_project_geographic(np.ascontiguousarray(x), np.ascontiguousarray(y),
                                           np.ascontiguousarray(z), str(crs))
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    att = [roles.get(k) for k in ("roll", "pitch", "heading")]
    if any(a is None for a in att):
        att = [None, None, None]
    return Trajectory(roles["time"], x, y, z, *att)


def _is_cloud(source) -> bool:
    return isinstance(source, PointCloud)


def _catalog(source):
    from .catalogue import _as_catalog
    return _as_catalog(source)


def _workers(workers):
    from .catalogue import _workers as w
    return w(workers)


def estimate_trajectory(source, interval: float = 0.5, min_pulses: int = 30,
                        min_separation: float = 2.0, max_pulses: int = 4000, extend: float = 2.0,
                        workers: int | None = None) -> Trajectory:
    """An approximate trajectory from the returns themselves.

    Every pulse with two or more returns defines a line through the
    scanner: from its last return through its first. Within a short time
    window the platform flies a nearly straight line at nearly constant
    speed, so its position and velocity there follow from a linear least
    squares fit of a moving point ``a + b (t - t_c)`` to those lines. This
    is the idea of lidR's ``track_sensor()`` (Roussel et al. 2020, after
    Gatziolis & McGaughey 2019), which fits a fixed point per window; the
    motion within the window is modelled here, so longer windows can be used
    without the position lagging.

    Windows of ``interval`` seconds are anchored at multiples of
    ``interval`` in GPS time, per flight line (``point_source_id``). Each
    window with ``min_pulses`` usable pulses (first and last returns at
    least ``min_separation`` apart, the ``max_pulses`` widest used) gives a
    sample at the mean time of its pulses; the first and last windows of a
    line also give samples towards the line's first and last returns,
    extrapolated with the fitted velocity by at most ``extend`` seconds (over
    open ground a line has no multiple returns to fit). Lines more than three median distances (and
    5 cm) from the first fit are left out of a second one, and a window
    whose sensor comes out below its returns is rejected.

    The result is approximate: it depends on the multiple returns (few over
    open ground or sparse canopy), carries no attitude, and cannot follow
    the platform's wobble within a window. Check ``rms`` and use a measured
    trajectory where one exists.

    Parameters
    ----------
    source
        A :class:`~sylva.PointCloud` with ``gps_time`` (and
        ``return_number``, ``point_source_id`` if present), or a catalogue,
        whose tiles are read one at a time (each tile's own points only).
    interval
        Window length (s).
    min_pulses
        Fewest usable pulses in a window.
    min_separation
        Least first-to-last distance (m) of a usable pulse.
    max_pulses
        Most pulses fitted per window.
    extend
        Longest extrapolation (s) at each end of a line.
    workers
        Tiles read at once for a catalogue.

    Returns
    -------
    Trajectory
        With ``rms`` (m), ``n_pulses`` and ``line`` per sample.

    Raises
    ------
    ValueError
        For bad settings, points without ``gps_time``, or too few multiple
        returns for any window.
    """
    args = (float(interval), int(min_pulses), float(min_separation), int(max_pulses), float(extend))
    if _is_cloud(source):
        d = _core.als_estimate_trajectory(source.xyz, source.attrs, *args)
    else:
        d = _core.als_estimate_trajectory_catalog(_catalog(source)._core(), *args,
                                                  _workers(workers))
    return Trajectory.from_dict(d)


def pulses(cloud: PointCloud, trajectory, max_gap: float | None = None, time_offset: float = 0.0,
           fill_missing: bool = False, max_fill: int = 8, drop_incomplete: bool = False,
           report: bool = False):
    """Group discrete returns into pulses fired from the sensor.

    Returns sharing a ``gps_time`` (and ``point_source_id``, when present)
    are one pulse; a group that repeats a ``return_number`` (two channels
    firing at the same instant) is split where it repeats. Each pulse starts
    at the sensor position interpolated from ``trajectory`` at its time, and
    points at its farthest return; each return's range is its distance from
    the sensor. The result traces with :func:`sylva.voxels.ray_voxelize`
    like a terrestrial scan.

    What discrete returns cannot give back:

    - **Returns missing from a pulse** (a return below the detection
      threshold, or clipped away at a tile or chunk edge) are counted from
      ``number_of_returns``; the pulse keeps the returns it has
      (``drop_incomplete`` leaves it out). If its last return is missing,
      the tracer ends the pulse early.
    - **Pulses with no return** (into water, onto a dark roof, lost to
      absorption) leave no record. With ``fill_missing``, a hole of at most
      ``max_fill`` pulses in a line's regular firing is filled with pulses
      without returns, their directions interpolated between the pulses on
      either side, provided the mirror swept steadily across the hole.
      Larger holes, and pulses at the edge of the data, are not recovered.
    - **Returns closer than the range resolution** were merged by the
      receiver, and the energy each return carried is not recorded, so every
      return stands for an equal share of its pulse.

    Parameters
    ----------
    cloud
        Returns with ``gps_time`` (``return_number``, ``number_of_returns``
        and ``point_source_id`` are used when present). All attributes
        become echo attributes.
    trajectory
        A :class:`Trajectory`, a mapping of arrays (such as
        :attr:`sylva.synthetic.ALSFlight.trajectory`) or a trajectory file.
    max_gap
        Longest gap between trajectory samples (s) to interpolate across;
        ten median sample intervals if None. Returns in longer gaps, or
        outside the trajectory, are left out.
    time_offset
        Seconds added to ``gps_time`` to reach the trajectory's clock (see
        :func:`week_seconds`).
    fill_missing, max_fill
        Infer pulses without a return in holes of at most ``max_fill``
        pulses.
    drop_incomplete
        Leave out pulses with fewer returns than ``number_of_returns``.
    report
        Also return what was found.

    Returns
    -------
    Shots or (Shots, dict)
        Pulses ordered by flight line and time. The report has
        ``n_returns``, ``n_pulses``, ``n_incomplete``,
        ``n_missing_returns``, ``n_dropped``, ``n_split``,
        ``n_unpositioned`` (returns left out), ``n_filled``,
        ``pulse_interval`` (s) and ``line_offset_median`` /
        ``line_offset_p95``: the distance (m) from the sensor to the line
        through each pulse's first and last returns. Centimetres to
        decimetres mean the trajectory fits the points; metres mean a time
        or height offset.

    Raises
    ------
    ValueError
        Without ``gps_time``, or when no return falls within the trajectory
        (the message gives the offset if the clocks look like adjusted
        standard time against seconds of the week).
    """
    if not isinstance(cloud, PointCloud):
        raise ValueError(f"pulses takes a PointCloud, got {type(cloud).__name__}")
    if int(max_fill) < 0:
        raise ValueError(f"max_fill must be non-negative, got {max_fill}")
    traj = _as_trajectory(trajectory)
    d, rep = _core.als_pulses(cloud.xyz, cloud.attrs, traj._core(), _gap(max_gap),
                              float(time_offset), bool(fill_missing), int(max_fill),
                              bool(drop_incomplete))
    shots = Shots._from_core(d)
    return (shots, rep) if report else shots
