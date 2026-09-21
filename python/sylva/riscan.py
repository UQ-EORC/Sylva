"""RiSCAN PRO project parsing.

A ``.RiSCAN`` directory holds ``project.rsp`` (XML with the POP and every
scan position's SOP), ``SCANS/ScanPosNNN/SINGLESCANS/*.rxp`` and often a
``DAT/ScanPosNNN.DAT`` copy of each SOP matrix. Older exports use
``all_sop.csv`` / ``project.pop`` instead; both layouts are handled.

The POP of geo-referenced projects is usually a geocentric (ECEF) transform
with translations of thousands of km, so it is returned but only applied
when asked for.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import read_matrix_file, read_rxp, read_rxp_shots
from .pointcloud import PointCloud
from .shots import Shots

__all__ = ["ScanPosition", "RiscanProject", "read_riscan_project"]


@dataclass
class ScanPosition:
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

    def transform(self, pop: np.ndarray | None = None) -> np.ndarray:
        """SOP, optionally composed with the POP (project -> global)."""
        m = np.eye(4) if self.sop is None else self.sop
        return m if pop is None else pop @ m

    def read(self, pop: np.ndarray | None = None, **options) -> PointCloud:
        """Read the scan in project (or global, with ``pop``) coordinates."""
        if self.rxp is None:
            raise FileNotFoundError(f"scan position {self.name} has no .rxp")
        return read_rxp(self.rxp, **options).transform(self.transform(pop))

    def read_shots(self, pop: np.ndarray | None = None, fill_missing: bool = False,
                   **options) -> Shots:
        """Read the scan as pulses in project (or global) coordinates.

        ``fill_missing=True`` reconstructs the no-return pulses from the scan
        pattern (see :meth:`Shots.fill_missing`) before applying the SOP.
        """
        if self.rxp is None:
            raise FileNotFoundError(f"scan position {self.name} has no .rxp")
        shots = read_rxp_shots(self.rxp, **options)
        if fill_missing:
            if self.pattern is None:
                raise ValueError(f"scan position {self.name} has no scan pattern in project.rsp")
            shots = shots.fill_missing(self.pattern)
        return shots.transform(self.transform(pop))


@dataclass
class RiscanProject:
    path: Path
    positions: list[ScanPosition]
    pop: np.ndarray | None = None
    name: str = ""

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, key: int | str) -> ScanPosition:
        if isinstance(key, int):
            return self.positions[key]
        for p in self.positions:
            if p.name == key:
                return p
        raise KeyError(key)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.positions]

    def with_scans(self) -> list[ScanPosition]:
        """Positions that have both an ``.rxp`` and a SOP."""
        return [p for p in self.positions if p.rxp is not None and p.sop is not None]

    def origins(self) -> np.ndarray:
        """Scanner origins in project coordinates, ``(n, 3)``."""
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


def _scan_files(pos_dir: Path) -> list[Path]:
    return sorted(
        p for p in pos_dir.rglob("*.rxp")
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
        pop = _matrix(pop_file.read_text())
    sops: dict[str, np.ndarray] = {}
    sop_csv = root / "all_sop.csv"
    if sop_csv.exists():
        with open(sop_csv, newline="") as f:
            for row in csv.DictReader(f):
                m = _rotation_zyx(*(float(row[k]) for k in ("rollDeg", "pitchDeg", "yawDeg")))
                m[:3, 3] = [float(row["x"]), float(row["y"]), float(row["z"])]
                sops[row["scanPosName"]] = m
    positions = []
    dirs = sorted(root.glob("SCANS/ScanPos*")) + sorted(root.glob("*.SCNPOS"))
    for pos_dir in dirs:
        name = pos_dir.name.replace(".SCNPOS", "")
        files = _scan_files(pos_dir)
        sop = sops.get(name)
        dat = root / "DAT" / f"{name}.DAT"
        if sop is None and dat.exists():
            sop = read_matrix_file(dat)
        positions.append(ScanPosition(name, files[0] if files else None, sop, files))
    return RiscanProject(root, positions, pop, root.stem)


def read_riscan_project(path: str | Path) -> RiscanProject:
    """Parse a RiSCAN PRO project directory (``project.rsp`` or legacy layout)."""
    root = Path(path)
    rsp = root / "project.rsp"
    if rsp.exists():
        return _from_rsp(root, rsp)
    return _from_legacy(root)
