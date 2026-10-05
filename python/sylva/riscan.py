# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""RiSCAN PRO project parsing.

A ``.RiSCAN`` directory holds ``project.rsp`` (XML with the POP and every
scan position's SOP), ``SCANS/ScanPosNNN/SINGLESCANS/*.rxp`` and often a
``DAT/ScanPosNNN.DAT`` copy of each SOP matrix. Older exports use
``all_sop.csv`` / ``project.pop`` instead, as do the scanner's own ``.PROJ``
projects (``ScanPosNNN.SCNPOS/scans/*.rxp``), which also record each
position's attitude (roll, pitch and yaw from the inclination sensors and
compass), a GNSS fix, and the reflective targets the scanner found; all
layouts are handled.

The POP of geo-referenced projects is usually a geocentric (ECEF) transform
with translations of thousands of km, so it is returned but only applied
when asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core
from .io import read_rxp, read_rxp_shots
from .pointcloud import PointCloud
from .shots import Shots

__all__ = [
    "RISCAN_FILTER_MODES",
    "RiscanProject",
    "ScanPosition",
    "angular_steps",
    "export_settings_mask",
    "gnss_to_local",
    "read_export_settings",
    "read_riscan_project",
    "riscan_like_mask",
]


@dataclass
class ScanPosition:
    """One scan position of a RiSCAN project.

    Attributes
    ----------
    name
        Position name, e.g. ``ScanPos001``.
    scans
        Every ``.rxp`` of the position, monitoring and residual files
        excluded.
    instrument
        Scanner model from ``project.rsp`` (None for legacy projects).
    """

    name: str
    rxp: Path | None
    """First non-monitoring ``.rxp`` of the position (``None`` if no scan)."""
    sop: np.ndarray | None
    """4x4 scanner -> project transform."""
    scans: list[Path] = field(default_factory=list)
    instrument: str | None = None
    pattern: dict | None = None
    """Angular scan pattern of the first scan: ``theta_start``, ``theta_delta``,
    ``theta_count`` (zenith, degrees from up) and ``phi_start``, ``phi_delta``,
    ``phi_count`` (azimuth). Lets missing (no-return) pulses be reconstructed."""
    tiepoints: Path | None = None
    """RIEGL ``.tpl`` list of the reflective targets the scanner found, if any."""
    gnss: tuple[float, float, float] | None = None
    """``(latitude, longitude, altitude)`` the scanner recorded. Metres off at
    best under canopy: good for deciding which scans could overlap, not for
    registering them."""
    attitude: np.ndarray | None = None
    """``(3, 3)`` rotation from the scanner's frame to a level, north-referenced
    one (the ``.pose`` roll, pitch and yaw, or ``pose_estimation.sop``). A
    scanner on a tilt mount has its z-axis sideways; this levels it. Roll and
    pitch are good to about 0.1 degree, yaw to about a degree."""

    @property
    def levelling(self) -> np.ndarray | None:
        """``(4, 4)`` rotation that levels this scan (from :attr:`attitude`), or None."""
        if self.attitude is None:
            return None
        T = np.eye(4)
        T[:3, :3] = self.attitude
        return T

    @property
    def origin(self) -> np.ndarray | None:
        """Scanner position in project coordinates (the SOP translation), or None."""
        return None if self.sop is None else self.sop[:3, 3].copy()

    def reflectors(self) -> list:
        """The targets the scanner found here (:attr:`tiepoints`), in its own frame.

        Returns
        -------
        list of sylva.coreg.Reflector
        """
        if self.tiepoints is None:
            return []
        from .coreg.reflectors import read_tiepoint_list

        return read_tiepoint_list(self.tiepoints)

    def transform(self, pop: np.ndarray | None = None) -> np.ndarray:
        """Scanner-to-project (or scanner-to-global) transform.

        Parameters
        ----------
        pop
            Project-to-global matrix (:attr:`RiscanProject.pop`). If given,
            the result is ``pop @ sop``.

        Returns
        -------
        numpy.ndarray
            ``(4, 4)`` matrix; the identity if the position has no SOP.
        """
        m = np.eye(4) if self.sop is None else self.sop
        return m if pop is None else pop @ m

    def read(self, pop: np.ndarray | None = None, **options) -> PointCloud:
        """Read the scan as points in project (or global) coordinates.

        Parameters
        ----------
        pop
            Also apply this project-to-global matrix.
        **options
            Passed to :func:`sylva.io.read_rxp` (``shot_stride``,
            ``min_range``, ``echoes``, ...).

        Returns
        -------
        PointCloud

        Raises
        ------
        FileNotFoundError
            If the position has no ``.rxp``.
        """
        if self.rxp is None:
            raise FileNotFoundError(f"scan position {self.name} has no .rxp")
        return read_rxp(self.rxp, **options).transform(self.transform(pop))

    def read_shots(self, pop: np.ndarray | None = None, fill_missing: bool = False,
                   **options) -> Shots:
        """Read the scan as pulses in project (or global) coordinates.

        Parameters
        ----------
        pop
            Also apply this project-to-global matrix.
        fill_missing
            Reconstruct the no-return pulses from the scan pattern (see
            :meth:`sylva.Shots.fill_missing`) before applying the SOP. Needed
            for gap fraction and ray tracing.
        **options
            Passed to :func:`sylva.io.read_rxp_shots`.

        Returns
        -------
        Shots

        Raises
        ------
        FileNotFoundError
            If the position has no ``.rxp``.
        ValueError
            If ``fill_missing`` is set but ``project.rsp`` has no scan pattern.
        """
        if self.rxp is None:
            raise FileNotFoundError(f"scan position {self.name} has no .rxp")
        shots = read_rxp_shots(self.rxp, **options)
        if fill_missing:
            if self.pattern is None:
                raise ValueError(f"scan position {self.name} has no scan pattern in project.rsp")
            shots = shots.fill_missing(self.pattern, shot_stride=options.get("shot_stride", 1))
        return shots.transform(self.transform(pop))


@dataclass
class RiscanProject:
    """A parsed RiSCAN PRO project; build with :func:`read_riscan_project`.

    Index by position number or name: ``project[0]``,
    ``project["ScanPos003"]``. ``len(project)`` is the number of positions.

    Attributes
    ----------
    path
        The ``.RiSCAN`` directory.
    positions
        Every scan position, in project order.
    pop
        Project-to-global matrix, or None. Often geocentric (ECEF), so it is
        never applied unless passed explicitly.
    name
        Project name.
    """

    path: Path
    positions: list[ScanPosition]
    pop: np.ndarray | None = None
    name: str = ""

    def __len__(self) -> int:
        return len(self.positions)

    def __iter__(self):
        return iter(self.positions)

    def __getitem__(self, key: int | str) -> ScanPosition:
        if isinstance(key, int):
            return self.positions[key]
        for p in self.positions:
            if p.name == key:
                return p
        raise KeyError(key)

    @property
    def names(self) -> list[str]:
        """Position names, in project order."""
        return [p.name for p in self.positions]

    def with_scans(self, require_sop: bool = True) -> list[ScanPosition]:
        """Positions that have a scan.

        Parameters
        ----------
        require_sop
            Only positions that also have a SOP, so can be read into project
            coordinates. False for registration, which does not need one.

        Returns
        -------
        list of ScanPosition
        """
        return [p for p in self.positions
                if p.rxp is not None and (p.sop is not None or not require_sop)]

    def gnss_positions(self) -> np.ndarray:
        """Scanner GNSS fixes in local metres (see :func:`gnss_to_local`).

        Returns
        -------
        numpy.ndarray
            ``(n, 3)``, one row per position, NaN where there was no fix.
        """
        return gnss_to_local([p.gnss for p in self.positions])

    def origins(self) -> np.ndarray:
        """Scanner positions in project coordinates.

        Returns
        -------
        numpy.ndarray
            ``(n, 3)`` SOP translations, one row per position that has a SOP
            (positions without one are skipped, so rows may not line up
            with :attr:`positions`).
        """
        return np.array([p.sop[:3, 3] for p in self.positions if p.sop is not None])


def _matrix(text: str | None) -> np.ndarray | None:
    return _core.riscan_parse_matrix(text or None)


def _rotation_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _core.riscan_rotation_zyx(float(roll), float(pitch), float(yaw))


def _position(d: dict) -> ScanPosition:
    scans = [Path(f) for f in d["scans"]]
    return ScanPosition(
        d["name"], scans[0] if scans else None, d["sop"], scans, d["instrument"], d["pattern"],
        None if d["tiepoints"] is None else Path(d["tiepoints"]), d["gnss"], d["attitude"])


def gnss_to_local(coordinates: list[tuple[float, float, float] | None]) -> np.ndarray:
    """GNSS fixes as local metres about the survey's own centre.

    An equirectangular projection, accurate to well under a metre over a
    plot, far finer than the fixes themselves.

    Parameters
    ----------
    coordinates
        ``(latitude, longitude, altitude)`` per position, or None.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` east, north, altitude; NaN where there was no fix.
    """
    return _core.riscan_gnss_to_local(
        [None if c is None else (float(c[0]), float(c[1]), float(c[2])) for c in coordinates])


def read_riscan_project(path: str | Path) -> RiscanProject:
    """Parse a RiSCAN PRO project directory.

    Reads no scan data, only the project structure and matrices.

    Parameters
    ----------
    path
        The ``.RiSCAN`` directory. With ``project.rsp``, SOPs and scan
        patterns come from it (falling back to ``DAT/<pos>.DAT``). Without
        it, the legacy layout is read: ``all_sop.csv`` (roll/pitch/yaw in
        degrees and x, y, z), ``project.pop`` and ``SCANS/ScanPos*`` or
        ``*.SCNPOS`` folders. For a scanner ``.PROJ`` (``*.SCNPOS``) each
        position also gets the scanner's attitude (:attr:`ScanPosition.levelling`),
        its GNSS fix and its ``.tpl`` target list.

    Returns
    -------
    RiscanProject

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a directory.

    Examples
    --------
    >>> project = sylva.read_riscan_project("plot.RiSCAN")
    >>> shots = [p.read_shots(fill_missing=True, shot_stride=4) for p in project.with_scans()]
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"not a project directory: {root}")
    d = _core.riscan_read_project(root)
    return RiscanProject(root, [_position(p) for p in d["positions"]], d["pop"], d["name"])


# --------------------------------------------------------------------------- #
# RiSCAN PRO's filters
# --------------------------------------------------------------------------- #

RISCAN_FILTER_MODES = ("none", "current", "legacy")


def read_export_settings(path: str | Path) -> dict[str, tuple[float, float]]:
    """Read a RiSCAN PRO export filter settings file.

    One line per attribute, ``name, minimum, maximum`` (``;`` also separates,
    ``#`` starts a comment), for example::

        deviation, 0, 12
        range, 2, 100
        reflectance, -20, 5

    Names are lower-cased and a leading ``riegl.`` is dropped.

    Parameters
    ----------
    path
        The settings file.

    Returns
    -------
    dict
        ``{attribute: (minimum, maximum)}`` for ``range``, ``deviation``,
        ``reflectance`` and ``amplitude``.

    Raises
    ------
    ValueError
        On a malformed line, an unknown attribute (so a typo cannot pass
        silently) or a minimum above its maximum.
    """
    return {name: (lo, hi) for name, lo, hi in _core.riscan_read_export_settings(Path(path))}


def export_settings_mask(settings: dict[str, tuple[float, float]], xyz: np.ndarray,
                         attributes: dict[str, np.ndarray]) -> np.ndarray:
    """Keep-mask of points within every interval of ``settings``, as RiSCAN's export filter.

    Parameters
    ----------
    settings
        From :func:`read_export_settings`.
    xyz
        ``(n, 3)`` points in the scanner's frame (``range`` is measured from
        its origin).
    attributes
        Per-point ``deviation``, ``reflectance``, ``amplitude`` as needed.
        RIEGL's "not measured" deviation (65535) counts as -1.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` bool; intervals are closed.

    Raises
    ------
    KeyError
        If an attribute the settings bound is missing (silently skipping it
        would keep more than asked).
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    values = {}
    for name in settings:
        if name == "range":
            continue
        if name not in attributes:
            raise KeyError(f"export settings need attribute {name!r}, which this source lacks")
        values[name] = np.asarray(attributes[name], dtype=np.float64).ravel()
    bounds = [(name, float(lo), float(hi)) for name, (lo, hi) in settings.items()]
    return _core.riscan_export_settings_mask(bounds, xyz, values)


def angular_steps(xyz: np.ndarray, sample: int = 2_000_000) -> tuple[float, float]:
    """The scan pattern's angular increments (degrees), estimated from the points.

    The polar step is the median step between consecutive records; the
    azimuth step is the spacing of the azimuths within one polar row (one
    shot per line), the median over a dozen rows. Falls back to 0.03 degrees,
    the usual VZ-series setting, for a scan too small to tell.

    Parameters
    ----------
    xyz
        ``(n, 3)`` points in the scanner's frame, in recording order.
    sample
        Records used.

    Returns
    -------
    theta_step, phi_step : float
    """
    n = min(len(xyz), sample)
    if n < 1000:
        return 0.03, 0.03
    p = np.asarray(xyz[:n], dtype=np.float32).reshape(-1, 3)
    return _core.riscan_angular_steps(p, int(n))


def riscan_like_mask(xyz: np.ndarray, amplitude: np.ndarray, mode: str = "current", *,
                     min_range: float = 0.5, window_steps: float = 1.5, window_range: float = 1.0,
                     min_neighbours: int = 6, weak_db: float = 12.0,
                     steps: tuple[float, float] | None = None) -> np.ndarray:
    """What RiSCAN PRO's RXP import keeps, approximately, as a keep-mask.

    Measured against RiSCAN's own databases on TERN VZ-2000i scans:

    ``"current"``
        The import a current RiSCAN performs (rimta5 with window analysis).
        It drops 0.02-0.3 % of the echoes RiVLib delivers, nearly all window
        echoes within half a metre of the scanner, so this drops
        ``range < min_range``. Agreement 99.7 %.
    ``"legacy"``
        The older conversion, which discards echoes the MTA resolution cannot
        support by their neighbours, 19-30 % of a scan. Reproduced as: drop an
        echo weaker than ``weak_db`` with fewer than ``min_neighbours`` other
        echoes within ``window_steps`` scan increments and ``window_range`` m
        of range, plus the near-range drop. Agreement 85-86 %; the rest of
        RiSCAN's decision depends on pulse timing the point stream lacks.
    ``"none"``
        Keep everything.

    Parameters
    ----------
    xyz
        ``(n, 3)`` points in the scanner's frame, in recording order.
    amplitude
        ``(n,)`` amplitude (dB).
    mode
        ``"none"``, ``"current"`` or ``"legacy"``.
    steps
        ``(theta_step, phi_step)`` of the scan pattern (degrees); estimated
        by :func:`angular_steps` if None.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` bool.
    """
    if mode not in RISCAN_FILTER_MODES:
        raise ValueError(f"mode must be one of {RISCAN_FILTER_MODES}, not {mode!r}")
    n = len(xyz)
    if mode == "none" or n == 0:
        return np.ones(n, dtype=bool)
    p = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if steps:
        theta_step, phi_step = steps
        steps = (float(theta_step), float(phi_step))
    else:
        steps = None
    amplitude = np.asarray(amplitude, dtype=np.float32).ravel()
    return _core.riscan_like_mask(p, amplitude, mode, float(min_range), float(window_steps),
                                  float(window_range), int(min_neighbours), float(weak_db), steps)
