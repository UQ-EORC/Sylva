# Tests of reading a scanner .PROJ.
import json
from pathlib import Path

import numpy as np
import pytest

from sylva.riscan import gnss_to_local, read_riscan_project


def make_project(root: Path, names=("ScanPos001", "ScanPos002", "ScanPos003"), gnss=True) -> Path:
    """Build a minimal scanner .PROJ layout: SCNPOS dirs, all_sop.csv, final.pose."""
    project = root / "survey.PROJ"
    rows = ["scanPosName,x,y,z,rollDeg,pitchDeg,yawDeg"]
    for k, name in enumerate(names):
        pos = project / f"{name}.SCNPOS"
        (pos / "scans").mkdir(parents=True)
        (pos / "scans" / f"2408{k:02d}_120000.rxp").write_bytes(b"")
        (pos / "scans" / f"2408{k:02d}_120000.mon.rxp").write_bytes(b"")
        if gnss:
            (pos / "final.pose").write_text(
                json.dumps({"gnss": {"latitude": round(-12.5 + k * 0.0002, 6),
                                     "longitude": round(130.8 + k * 0.0002, 6), "altitude": 100.0}})
            )
        rows.append(f"{name},{20.0 * k:.3f},{5.0 * k:.3f},0.100,0.5,-0.25,{30.0 * k:.1f}")
    (project / "all_sop.csv").write_text("\n".join(rows) + "\n")
    return project


def test_missing_project_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_riscan_project(tmp_path / "nothing.PROJ")


def test_reads_a_scanner_project(tmp_path):
    project = read_riscan_project(make_project(tmp_path))
    assert len(project) == 3
    assert project.names == ["ScanPos001", "ScanPos002", "ScanPos003"]
    for position in project.with_scans(require_sop=False):
        assert position.rxp is not None and position.rxp.exists()
        assert ".mon." not in position.rxp.name, "monitoring streams are not survey scans"
        assert position.sop.shape == (4, 4)
        assert np.allclose(position.sop[3], [0, 0, 0, 1])
    assert project["ScanPos002"].origin is not None
    assert np.allclose(project["ScanPos002"].origin, [20.0, 5.0, 0.1])


def test_attitude_is_read_from_the_scan_pose_file(tmp_path):
    """A tilted scanner reports roll and pitch; the reader turns them into a levelling rotation."""
    project_dir = make_project(tmp_path)
    pos = project_dir / "ScanPos002.SCNPOS"
    stem = next(pos.glob("scans/*[0-9].rxp")).name.split(".")[0]
    (pos / f"{stem}.pose").write_text('{"roll": -70.0, "pitch": -80.0, "yaw": 150.0}')
    project = read_riscan_project(project_dir)
    tilted = project["ScanPos002"]
    assert tilted.attitude is not None and tilted.attitude.shape == (3, 3)
    assert np.isclose(np.linalg.det(tilted.attitude), 1.0, atol=1e-9)
    assert (tilted.attitude @ [0, 0, 1])[2] < 0.5, "a tilted scanner's z-axis is not up"
    assert tilted.levelling.shape == (4, 4) and np.allclose(tilted.levelling[:3, 3], 0)
    # positions without a pose file fall back to final.pose, which here carries no angles
    assert project["ScanPos001"].attitude is None


def test_sop_rotations_are_proper(tmp_path):
    for position in read_riscan_project(make_project(tmp_path)).with_scans():
        R = position.sop[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_sop_yaw_is_about_z(tmp_path):
    project = read_riscan_project(make_project(tmp_path))
    yaw = np.degrees(np.arctan2(project["ScanPos003"].sop[1, 0], project["ScanPos003"].sop[0, 0]))
    assert np.isclose(yaw, 60.0, atol=0.5)


def test_gnss_fixes_become_local_metres(tmp_path):
    project = read_riscan_project(make_project(tmp_path))
    fixes = [p.gnss for p in project]
    assert all(f is not None for f in fixes)
    local = gnss_to_local(fixes)
    assert local.shape == (3, 3) and np.all(np.isfinite(local))
    # 0.0002 degrees of latitude is about 22 m
    assert 15 < np.linalg.norm(local[1, :2] - local[0, :2]) < 40


def test_positions_without_a_fix_are_nan(tmp_path):
    project = read_riscan_project(make_project(tmp_path, gnss=False))
    local = gnss_to_local([p.gnss for p in project])
    assert np.all(np.isnan(local))


def test_position_without_a_scan_is_kept_but_has_no_rxp(tmp_path):
    project_dir = make_project(tmp_path)
    (project_dir / "ScanPos004.SCNPOS").mkdir()
    project = read_riscan_project(project_dir)
    assert len(project) == 4
    assert project["ScanPos004"].rxp is None
    assert len(project.with_scans(require_sop=False)) == 3
