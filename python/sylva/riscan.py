# Sylva: terrestrial laser scanning processing for forest ecology.
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

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import read_matrix_file, read_rxp, read_rxp_shots
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
    if not text:
        return None
    vals = [float(t) for t in text.split()]
    return np.array(vals).reshape(4, 4) if len(vals) == 16 else None


def _rotation_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = np.radians([roll, pitch, yaw])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    return m


def _scan_files(pos_dir: Path, recursive: bool = True) -> list[Path]:
    return sorted(
        p for p in (pos_dir.rglob("*.rxp") if recursive else pos_dir.glob("*.rxp"))
        if ".mon." not in p.name and "residual" not in p.name and not p.name.endswith(".part")
    )


def _from_rsp(root: Path, rsp: Path) -> RiscanProject:
    tree = ET.parse(rsp).getroot()
    pop = None
    pop_el = tree.find("./pop")
    if pop_el is not None:
        pop = _matrix(pop_el.findtext("matrix"))
    positions = []
    for sp in tree.iter("scanposition"):
        name = sp.get("name") or sp.findtext("name") or ""
        sop = _matrix(sp.findtext("./sop/matrix"))
        pos_dir = root / "SCANS" / name
        files = _scan_files(pos_dir) if pos_dir.is_dir() else []
        instrument = None
        pattern = None
        for scan in sp.iter("scan"):
            instrument = instrument or scan.findtext("instrument")
            if pattern is None:
                keys = ("theta_start", "theta_delta", "theta_count", "phi_start", "phi_delta",
                        "phi_count")
                vals = {k: scan.findtext(k) for k in keys}
                if all(vals.values()):
                    pattern = {k: (int(v) if k.endswith("count") else float(v))
                               for k, v in vals.items()}
            fname = scan.findtext("file")
            if fname:
                candidate = pos_dir / "SINGLESCANS" / fname
                if candidate.exists() and candidate not in files:
                    files.append(candidate)
        if sop is None:
            dat = root / "DAT" / f"{name}.DAT"
            if dat.exists():
                sop = read_matrix_file(dat)
        positions.append(
            ScanPosition(name, files[0] if files else None, sop, files, instrument, pattern)
        )
    return RiscanProject(root, positions, pop, tree.findtext("name") or root.stem)


def _from_legacy(root: Path) -> RiscanProject:
    pop = None
    pop_file = root / "project.pop"
    if pop_file.exists():
        pop = _read_pop(pop_file)
    sops = _read_all_sop(root / "all_sop.csv")
    positions = []
    for pos_dir in sorted(root.glob("SCANS/ScanPos*")):
        name = pos_dir.name
        files = _scan_files(pos_dir)
        sop = sops.get(name)
        dat = root / "DAT" / f"{name}.DAT"
        if sop is None and dat.exists():
            sop = read_matrix_file(dat)
        positions.append(ScanPosition(name, files[0] if files else None, sop, files))
    for pos_dir in sorted(root.glob("*.SCNPOS")):
        # The scanner's own project: the survey scan is in scans/, beside
        # monitoring and tie-point scans that are not survey data.
        name = pos_dir.stem
        scans = pos_dir / "scans"
        files = _scan_files(scans, recursive=False) if scans.is_dir() else []
        sop = sops.get(name)
        dat = root / "DAT" / f"{name}.DAT"
        if sop is None and dat.exists():
            sop = read_matrix_file(dat)
        rxp = files[0] if files else None
        positions.append(ScanPosition(
            name, rxp, sop, files, tiepoints=next(iter(sorted(pos_dir.glob("*.tpl"))), None),
            gnss=_read_pose_gnss(pos_dir / "final.pose"), attitude=_read_attitude(pos_dir, rxp)))
    return RiscanProject(root, positions, pop, root.stem)


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _read_pop(path: Path) -> np.ndarray | None:
    """A ``project.pop`` matrix: the first 16 numbers in the file."""
    try:
        values = [float(v) for v in _NUMBER.findall(path.read_text())]
    except (OSError, UnicodeDecodeError):
        return None
    return np.array(values[:16]).reshape(4, 4) if len(values) >= 16 else None


def _read_all_sop(path: Path) -> dict[str, np.ndarray]:
    """``all_sop.csv`` (roll, pitch, yaw in degrees and x, y, z) as 4x4 matrices.

    Bad rows are skipped.
    """
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("scanPosName") or "").strip()
            try:
                t = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
                angles = (float(row[k]) for k in ("rollDeg", "pitchDeg", "yawDeg"))
                m = _rotation_zyx(*angles)
            except (KeyError, TypeError, ValueError):
                continue
            if name and np.all(np.isfinite(t)):
                m[:3, 3] = t
                out[name] = m
    return out


def _read_attitude(directory: Path, rxp: Path | None) -> np.ndarray | None:
    """The scanner's ``level_from_scanner`` rotation for one position.

    Tries the ``.pose`` written beside the scan (named after its timestamp),
    ``pose_estimation.sop``, then ``final.pose``. Roll, pitch and yaw compose
    as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``, as in ``all_sop.csv``;
    ``pose_estimation.sop`` stores the matrix itself.
    """
    candidates = [directory / (rxp.name.split(".")[0] + ".pose")] if rxp is not None else []
    for path in candidates + [directory / "pose_estimation.sop", directory / "final.pose"]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        matrix = data.get("matrix3x3") if isinstance(data, dict) else None
        if matrix is not None:
            R = np.asarray(matrix, dtype=float)
            if R.shape == (3, 3) and np.all(np.isfinite(R)):
                return R
        try:
            roll, pitch, yaw = (float(data[k]) for k in ("roll", "pitch", "yaw"))
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite([roll, pitch, yaw])):
            return _rotation_zyx(roll, pitch, yaw)[:3, :3]
    return None


def _read_pose_gnss(path: Path) -> tuple[float, float, float] | None:
    """The GNSS fix of a scanner ``.pose`` file, if it has one."""
    if not path.exists():
        return None
    try:
        gnss = json.loads(path.read_text()).get("gnss") or {}
        latitude, longitude = gnss["latitude"], gnss["longitude"]
        if latitude is None or longitude is None:
            return None
        return float(latitude), float(longitude), float(gnss.get("altitude", 0.0) or 0.0)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


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
    out = np.full((len(coordinates), 3), np.nan)
    known = [c for c in coordinates if c is not None]
    if not known:
        return out
    lat0 = float(np.mean([c[0] for c in known]))
    lon0 = float(np.mean([c[1] for c in known]))
    scale = np.cos(np.radians(lat0))
    for k, c in enumerate(coordinates):
        if c is not None:
            out[k] = ((c[1] - lon0) * 111_320.0 * scale, (c[0] - lat0) * 111_320.0, c[2])
    return out


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
    rsp = root / "project.rsp"
    if rsp.exists():
        return _from_rsp(root, rsp)
    return _from_legacy(root)


# --------------------------------------------------------------------------- #
# RiSCAN PRO's filters
# --------------------------------------------------------------------------- #

RISCAN_FILTER_MODES = ("none", "current", "legacy")
_EXPORT_ATTRIBUTES = ("range", "deviation", "reflectance", "amplitude")


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
    settings: dict[str, tuple[float, float]] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) != 3:
            raise ValueError(f"{path}: expected 'attribute, min, max', got {raw!r}")
        name = parts[0].lower().removeprefix("riegl.")
        if name not in _EXPORT_ATTRIBUTES:
            raise ValueError(f"{path}: unknown attribute {parts[0]!r}; "
                             f"known: {sorted(_EXPORT_ATTRIBUTES)}")
        lo, hi = float(parts[1]), float(parts[2])
        if lo > hi:
            raise ValueError(f"{path}: minimum above maximum for {name}")
        settings[name] = (lo, hi)
    return settings


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
    keep = np.ones(len(xyz), dtype=bool)
    for name, (lo, hi) in settings.items():
        if name == "range":
            values = np.sqrt(np.einsum("ij,ij->i", xyz, xyz))
        elif name in attributes:
            values = np.asarray(attributes[name], dtype=np.float64)
            if name == "deviation":
                values = np.where(values == 65535, -1.0, values)
        else:
            raise KeyError(f"export settings need attribute {name!r}, which this source lacks")
        keep &= (values >= lo) & (values <= hi)
    return keep


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
    p = np.asarray(xyz[:n], dtype=np.float32)
    r = np.linalg.norm(p, axis=1)
    ok = r > 0.05
    p, r = p[ok], r[ok]
    theta = np.degrees(np.arccos(np.clip(p[:, 2] / r, -1.0, 1.0)))
    phi = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0
    step = np.abs(np.diff(theta))
    step = step[(step > 1e-3) & (step < 0.5)]
    theta_step = float(np.median(step)) if len(step) > 100 else 0.03
    lo, hi = np.percentile(theta, [10, 90])
    row_steps = []
    for centre in np.linspace(lo, hi, 12):
        dphi = np.diff(np.sort(phi[np.abs(theta - centre) < 0.5 * theta_step]))
        dphi = dphi[(dphi > 0.5 * theta_step) & (dphi < 1.0)]
        if len(dphi) >= 50:
            row_steps.append(np.median(dphi))
    phi_step = float(np.median(row_steps)) if len(row_steps) >= 3 else 0.03
    return theta_step, phi_step


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
    from . import _core

    if mode not in RISCAN_FILTER_MODES:
        raise ValueError(f"mode must be one of {RISCAN_FILTER_MODES}, not {mode!r}")
    n = len(xyz)
    keep = np.ones(n, dtype=bool)
    if mode == "none" or n == 0:
        return keep
    p = np.asarray(xyz, dtype=np.float32)
    r = np.linalg.norm(p, axis=1)
    keep &= r >= min_range
    if mode == "current":
        return keep
    theta_step, phi_step = steps or angular_steps(xyz)
    safe = np.maximum(r, np.float32(1e-6))
    # The unit ball of this space is the window: +-window_steps increments in
    # either angle and +-window_range metres of range.
    q = np.empty((n, 3), dtype=np.float64)
    q[:, 0] = (np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0) / (phi_step * window_steps)
    zenith = np.degrees(np.arccos(np.clip(p[:, 2] / safe, -1.0, 1.0)))
    q[:, 1] = zenith / (theta_step * window_steps)
    q[:, 2] = r / window_range
    del safe
    neighbours = np.asarray(_core.count_within(q, 1.0)) - 1
    del q
    weak = np.asarray(amplitude, dtype=np.float32) < weak_db
    keep &= ~(weak & (neighbours < min_neighbours))
    return keep
