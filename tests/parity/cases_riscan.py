"""Parity cases for sylva.riscan.

The projects are small synthetic directories written into a temporary
folder: a RiSCAN PRO project (``project.rsp``), the legacy layout
(``all_sop.csv``, ``project.pop``, ``SCANS/ScanPos*``) and a scanner
``.PROJ`` (``*.SCNPOS`` with ``.pose`` files, GNSS fixes and target lists).
Paths are recorded relative to the project folder.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from sylva import riscan

KEYS = ("theta_start", "theta_delta", "theta_count", "phi_start", "phi_delta", "phi_count")


def _numbers(values):
    return " ".join(repr(float(v)) for v in np.ravel(values))


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _transform(rng, scale=100.0):
    m = np.eye(4)
    m[:3, :3] = _rotation(rng)
    m[:3, 3] = rng.uniform(-scale, scale, 3)
    return m


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _describe(project, root):
    """A project's parsed content as flat arrays; paths relative to ``root``."""
    rel = lambda p: "" if p is None else Path(p).relative_to(root).as_posix()  # noqa: E731
    ps = project.positions
    nan = np.full((4, 4), np.nan)
    out = {
        "name": np.asarray(project.name),
        "path": np.asarray(rel(project.path)),
        "has_pop": project.pop is not None,
        "pop": nan if project.pop is None else project.pop,
        "names": np.asarray(project.names + [""]),
        "rxp": np.asarray([rel(p.rxp) for p in ps] + [""]),
        "scans": np.asarray(["|".join(rel(s) for s in p.scans) for p in ps] + [""]),
        "has_sop": np.asarray([p.sop is not None for p in ps] + [False]),
        "sop": np.array([nan if p.sop is None else p.sop for p in ps] + [nan]),
        "instrument": np.asarray(
            ["<None>" if p.instrument is None else p.instrument for p in ps] + [""]
        ),
        "has_pattern": np.asarray([p.pattern is not None for p in ps] + [False]),
        "pattern": np.array(
            [[np.nan] * 6 if p.pattern is None else [p.pattern[k] for k in KEYS] for p in ps]
            + [[np.nan] * 6]
        ),
        "pattern_keys": np.asarray(["|".join(p.pattern) if p.pattern else "" for p in ps] + [""]),
        "pattern_int": np.asarray(
            [p.pattern is not None and isinstance(p.pattern["theta_count"], int) for p in ps]
            + [False]
        ),
        "tiepoints": np.asarray([rel(p.tiepoints) for p in ps] + [""]),
        "has_gnss": np.asarray([p.gnss is not None for p in ps] + [False]),
        "gnss": np.array(
            [[np.nan] * 3 if p.gnss is None else list(p.gnss) for p in ps] + [[np.nan] * 3]
        ),
        "has_attitude": np.asarray([p.attitude is not None for p in ps] + [False]),
        "attitude": np.array(
            [np.full((3, 3), np.nan) if p.attitude is None else p.attitude for p in ps]
            + [np.full((3, 3), np.nan)]
        ),
        "levelling": np.array([nan if p.levelling is None else p.levelling for p in ps] + [nan]),
        "origin": np.array(
            [[np.nan] * 3 if p.origin is None else p.origin for p in ps] + [[np.nan] * 3]
        ),
        "transform_pop": np.array([p.transform(project.pop) for p in ps] + [nan]),
        "transform": np.array([p.transform() for p in ps] + [nan]),
        "with_scans": np.asarray([p.name for p in project.with_scans()] + [""]),
        "with_scans_any": np.asarray(
            [p.name for p in project.with_scans(require_sop=False)] + [""]
        ),
        "origins": project.origins().reshape(-1, 3),
        "gnss_positions": project.gnss_positions().reshape(-1, 3),
        "len": len(project),
    }
    return out


def _in_temp(build):
    tmp = Path(tempfile.mkdtemp(prefix="sylva-parity-riscan-"))
    try:
        root = build(tmp)
        return _describe(riscan.read_riscan_project(root), root)
    finally:
        shutil.rmtree(tmp)


# ------------------------------------------------------------------ projects


def _rsp_project(tmp):
    rng = np.random.default_rng(11)
    root = tmp / "Parity.RiSCAN"
    pop = _transform(rng, 5e6)
    sop1, sop4 = _transform(rng), _transform(rng)
    dat2 = _transform(rng)
    pattern = {
        "theta_start": 30.0,
        "theta_delta": rng.uniform(0.02, 0.06),
        "theta_count": 2512,
        "phi_start": rng.uniform(0, 1),
        "phi_delta": rng.uniform(0.02, 0.06),
        "phi_count": 9001,
    }
    tags = "".join(
        f"<{k}> {v!r}</{k}>" if k.endswith("count") else f"<{k}>{v!r}</{k}>"
        for k, v in pattern.items()
    )
    s1 = root / "SCANS/ScanPos001"
    for f in (
        "SINGLESCANS/b.rxp",
        "SINGLESCANS/a.rxp",
        "SINGLESCANS/a.mon.rxp",
        "SINGLESCANS/residual_a.rxp",
        "SINGLESCANS/c.rxp.part",
        "SINGLESCANS/x.mon.rxp",
        "other/deep/z.rxp",
        "SINGLESCANS/notes.txt",
    ):
        _touch(s1 / f)
    _touch(root / "SCANS/ScanPos002/SINGLESCANS/only.rxp")
    _touch(root / "SCANS/ScanPos004/SINGLESCANS/four.rxp")
    (root / "DAT").mkdir(parents=True)
    np.savetxt(root / "DAT/ScanPos002.DAT", dat2)
    np.savetxt(root / "DAT/ScanPos003.DAT", _transform(rng))
    m1 = _numbers(sop1[:2]) + " <!-- split --> " + _numbers(sop1[2:])
    xml = f"""<?xml version="1.0" standalone="no"?>
<!DOCTYPE project SYSTEM "./project.dtd" [
  <!-- PUT INTERNAL DOCUMENT TYPE DEFINITION SUBSET HERE -->
]>
<project name="Parity" kind="ProjectX"><name>ParityProject</name>
<pop name="POP" kind="POP"><matrix rows="4" cols="4">{_numbers(pop)}</matrix></pop>
<scanpositions name="SCANS" kind="SCANS">
 <scanposition name="ScanPos001" kind="PositionX">
  <singlescans>
   <scan name="s0" kind="ScanAcquiredX"><file>a.rxp</file><instrument></instrument>
    <theta_start>30</theta_start><theta_delta>0.04</theta_delta></scan>
   <scan name="s1"><file>x.mon.rxp</file><instrument>VZ-2000i</instrument>{tags}</scan>
   <scan name="s2"><file>missing.rxp</file><instrument>VZ-400i</instrument></scan>
  </singlescans>
  <sop name="SOP" kind="SOP"><matrix rows="4" cols="4">{m1}</matrix></sop>
 </scanposition>
 <scanposition kind="PositionX"><name>ScanPos002</name>
  <singlescans><scan name="s"><file>only.rxp</file></scan></singlescans>
 </scanposition>
 <scanposition name="ScanPos003" kind="PositionX"><singlescans/>
  <sop name="SOP"><matrix>{_numbers(sop4[:3])} 1 2</matrix></sop></scanposition>
 <scanposition name="ScanPos004" kind="PositionX"><sop name="SOP"/><sop><matrix>
  {_numbers(sop4)}</matrix></sop></scanposition>
</scanpositions></project>
"""
    (root / "project.rsp").write_bytes(xml.replace("\n", "\r\n").encode())
    return root


def _legacy_project(tmp):
    rng = np.random.default_rng(12)
    root = tmp / "legacy.RiSCAN"
    for k in (1, 2, 3, 5):
        _touch(root / f"SCANS/ScanPos00{k}/SINGLESCANS/{k:03d}.rxp")
    _touch(root / "SCANS/ScanPos002/SINGLESCANS/000.mon.rxp")
    (root / "SCANS/ScanPos004").mkdir(parents=True)
    angles = rng.uniform(-180, 180, (6, 3))
    xyz = rng.uniform(-50, 50, (6, 3))
    rows = ["scanPosName,x,y,z,rollDeg,pitchDeg,yawDeg,comment"]
    for k in range(6):
        name = f"ScanPos00{k + 1}"
        if k == 0:
            name = f'"{name}"'
        vals = list(xyz[k]) + list(angles[k])
        if k == 2:
            vals[0] = "not a number"
        if k == 4:
            vals[1] = float("inf")
        rows.append(
            ",".join(
                [name]
                + [v if isinstance(v, str) else repr(float(v)) for v in vals]
                + ['"a, quoted note"']
            )
        )
        if k == 1:
            rows.append("")
    rows.append("ScanPos006,1,2")
    (root / "all_sop.csv").write_text("\r\n".join(rows) + "\r\n")
    (root / "DAT").mkdir()
    np.savetxt(root / "DAT/ScanPos003.DAT", _transform(rng))
    np.savetxt(root / "DAT/ScanPos004.DAT", _transform(rng))
    pop = _transform(rng, 1e6)
    (root / "project.pop").write_text(
        "# POP\nmatrix:\n"
        + "\n".join(" ".join(f"{v:.9e}" for v in row) for row in pop)
        + "\nextra 1 2 3\n"
    )
    return root


def _proj_project(tmp):
    rng = np.random.default_rng(13)
    root = tmp / "survey.PROJ"
    rows = ["scanPosName,x,y,z,rollDeg,pitchDeg,yawDeg"]
    for k in range(6):
        name = f"ScanPos{k + 1:03d}"
        pos = root / f"{name}.SCNPOS"
        stem = f"2608{k:02d}_1200{k:02d}"
        pos.mkdir(parents=True)
        if k != 5:
            _touch(pos / "scans" / f"{stem}.rxp")
            _touch(pos / "scans" / f"{stem}.mon.rxp")
            _touch(pos / "scans" / "deeper" / "ignored.rxp")
            _touch(pos / "tiepointscans" / f"{stem}.rxp")
        if k in (0, 3):
            _touch(pos / f"{stem}.tpl")
            _touch(pos / "0_earlier.tpl")
        roll, pitch, yaw = rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-180, 180)
        lat, lon = -12.5 + rng.uniform(0, 1e-3), 130.8 + rng.uniform(0, 1e-3)
        if k == 0:
            (pos / f"{stem}.pose").write_text(
                json.dumps({"roll": roll, "pitch": pitch, "yaw": yaw})
            )
            (pos / "final.pose").write_text(
                json.dumps({"gnss": {"latitude": lat, "longitude": lon, "altitude": 101.5}})
            )
        elif k == 1:
            (pos / f"{stem}.pose").write_text('{"roll": NaN, "pitch": 1, "yaw": 2}')
            (pos / "pose_estimation.sop").write_text(
                json.dumps({"matrix3x3": _rotation(rng).tolist()})
            )
            (pos / "final.pose").write_text(
                json.dumps({"gnss": {"latitude": str(lat), "longitude": lon, "altitude": None}})
            )
        elif k == 2:
            (pos / "pose_estimation.sop").write_text('{"matrix3x3": [[1, 0], [0, 1]]}')
            (pos / "final.pose").write_text(
                json.dumps(
                    {
                        "roll": str(roll),
                        "pitch": pitch,
                        "yaw": yaw,
                        "gnss": {"latitude": None, "longitude": lon},
                    }
                )
            )
        elif k == 3:
            (pos / f"{stem}.pose").write_text("not json")
            (pos / "final.pose").write_text(
                json.dumps(
                    {
                        "yaw": yaw,
                        "roll": True,
                        "pitch": 0,
                        "gnss": {"latitude": lat, "longitude": lon},
                    }
                )
            )
        elif k == 4:
            (pos / "final.pose").write_text(json.dumps({"gnss": {}}))
        rows.append(
            f"{name},{20.0 * k:.3f},{5.0 * k:.3f},{rng.uniform():.6f},{roll!r},{pitch!r},{yaw!r}"
        )
    (root / "all_sop.csv").write_text("\n".join(rows) + "\n")
    _touch(root / "SCANS/ScanPos900/SINGLESCANS/legacy.rxp")
    return root


def rsp_project():
    return _in_temp(_rsp_project)


def legacy_project():
    return _in_temp(_legacy_project)


def proj_project():
    return _in_temp(_proj_project)


def empty_project():
    return _in_temp(lambda tmp: (tmp / "empty.riproject").mkdir() or tmp / "empty.riproject")


# ----------------------------------------------------------------- numerics


def matrices():
    rng = np.random.default_rng(21)
    angles = np.vstack(
        [rng.uniform(-360, 360, (40, 3)), [[0, 0, 0], [90, -90, 180], [1e-9, 45, -720]]]
    )
    out = {"rotation_zyx": np.array([riscan._rotation_zyx(*a) for a in angles])}
    vals = rng.normal(size=16) * 1e3
    out["matrix"] = riscan._matrix(" \n".join(repr(float(v)) for v in vals))
    out["matrix_short"] = riscan._matrix(" ".join(repr(float(v)) for v in vals[:15])) is None
    out["matrix_empty"] = riscan._matrix("") is None
    return out


def gnss():
    rng = np.random.default_rng(22)
    fixes = [
        (-27.5 + rng.uniform(0, 1e-3), 153.0 + rng.uniform(0, 1e-3), rng.uniform(0, 50))
        for _ in range(9)
    ]
    fixes[3] = None
    fixes[7] = None
    return {
        "local": riscan.gnss_to_local(fixes),
        "none": riscan.gnss_to_local([None, None]),
        "empty": riscan.gnss_to_local([]).reshape(-1, 3),
    }


def export_settings():
    rng = np.random.default_rng(23)
    tmp = Path(tempfile.mkdtemp(prefix="sylva-parity-riscan-"))
    try:
        f = tmp / "settings.txt"
        f.write_text(
            "# RiSCAN export\nriegl.Deviation; 0; 12  # closed\n\nrange, 2.5, 60\n"
            "REFLECTANCE , -20, 5\namplitude,0,1e3\n"
        )
        settings = riscan.read_export_settings(f)
    finally:
        shutil.rmtree(tmp)
    n = 5000
    xyz = rng.uniform(-80, 80, (n, 3))
    attrs = {
        "deviation": rng.choice([0, 3, 11, 12, 13, 65535], n).astype(np.uint16),
        "reflectance": rng.uniform(-25, 10, n).astype(np.float32),
        "amplitude": rng.uniform(-1, 1100, n),
    }
    out = {f"settings_{k}": np.asarray(v) for k, v in settings.items()}
    out["settings_order"] = np.asarray("|".join(settings))
    out["mask"] = riscan.export_settings_mask(settings, xyz, attrs)
    out["mask_range_only"] = riscan.export_settings_mask(
        {"range": (10.0, 40.0)}, xyz.astype(np.float32), {}
    )
    return out


def _stream(seed, step=0.04, n_lines=60, n_shots=400, jitter=True):
    """A scan in recording order: vertical lines of shots at increasing azimuth."""
    rng = np.random.default_rng(seed)
    theta = np.radians(40.0 + step * np.arange(n_shots))
    phi = np.radians(10.0 + step * np.arange(n_lines))
    tt, pp = np.meshgrid(theta, phi)
    tt, pp = tt.ravel(), pp.ravel()
    if jitter:
        tt = tt + np.radians(rng.normal(0, step * 0.02, tt.size))
    r = rng.uniform(2.0, 40.0, tt.size) * np.where(rng.uniform(size=tt.size) < 0.8, 1.0, 0.3)
    xyz = r[:, None] * np.column_stack(
        [np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)]
    )
    xyz[rng.choice(len(xyz), 30, replace=False)] = rng.uniform(-0.03, 0.03, (30, 3))
    amplitude = rng.uniform(0, 30, len(xyz)).astype(np.float32)
    return xyz, amplitude


def angular():
    out = {}
    for k, (step, lines, shots) in enumerate([(0.04, 60, 400), (0.03, 40, 200), (0.06, 30, 900)]):
        xyz, _ = _stream(30 + k, step, lines, shots)
        out[f"steps_{k}"] = np.asarray(riscan.angular_steps(xyz))
        out[f"steps_f32_{k}"] = np.asarray(riscan.angular_steps(xyz.astype(np.float32)))
    xyz, _ = _stream(40)
    out["steps_sample"] = np.asarray(riscan.angular_steps(xyz, sample=5000))
    out["steps_small"] = np.asarray(riscan.angular_steps(xyz[:999]))
    out["steps_no_jitter"] = np.asarray(riscan.angular_steps(_stream(41, jitter=False)[0]))
    return out


def riscan_mask():
    xyz, amplitude = _stream(50)
    out = {m: riscan.riscan_like_mask(xyz, amplitude, m) for m in riscan.RISCAN_FILTER_MODES}
    out["legacy_steps"] = riscan.riscan_like_mask(xyz, amplitude, "legacy", steps=(0.04, 0.04))
    out["legacy_wide"] = riscan.riscan_like_mask(
        xyz.astype(np.float32),
        amplitude,
        "legacy",
        min_range=1.0,
        window_steps=3.0,
        window_range=2.5,
        min_neighbours=9,
        weak_db=20.0,
    )
    out["empty"] = riscan.riscan_like_mask(np.zeros((0, 3)), np.zeros(0), "legacy")
    return out


CASES = {
    "rsp_project": rsp_project,
    "legacy_project": legacy_project,
    "proj_project": proj_project,
    "empty_project": empty_project,
    "matrices": matrices,
    "gnss": gnss,
    "export_settings": export_settings,
    "angular": angular,
    "riscan_mask": riscan_mask,
}
