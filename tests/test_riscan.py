import os
from pathlib import Path

import numpy as np
import pytest

from sylva import PointCloud, Shots, canopy, io, riscan

# Folder holding the TERN RiSCAN projects; the tests that need them skip without it.
TERN = Path(os.environ.get("SYLVA_TERN_DIR", "TERN_TLS_RAW"))
ROBSON = TERN / "RobsonCreek.RiSCAN"

RSP = """<?xml version="1.0"?>
<project><name>Demo</name>
<pop name="POP" kind="POP"><matrix rows="4" cols="4">
 1 0 0 100  0 1 0 200  0 0 1 300  0 0 0 1 </matrix></pop>
<scanpositions name="SCANS" kind="SCANS">
 <scanposition name="ScanPos001" kind="PositionX">
  <singlescans><scan name="s1" kind="ScanAcquiredX"><file>s1.rxp</file>
   <instrument>VZ-400i</instrument>
   <phi_count>9001</phi_count><phi_delta>0.04</phi_delta><phi_start>0</phi_start>
   <theta_count>2512</theta_count><theta_delta>0.04</theta_delta><theta_start>30</theta_start>
  </scan></singlescans>
  <sop name="SOP" kind="SOP"><matrix rows="4" cols="4">
   0 -1 0 1  1 0 0 2  0 0 1 3  0 0 0 1 </matrix></sop>
 </scanposition>
 <scanposition name="ScanPos002" kind="PositionX"><singlescans/></scanposition>
</scanpositions></project>
"""


def test_parse_rsp(tmp_path):
    root = tmp_path / "Demo.RiSCAN"
    (root / "SCANS/ScanPos001/SINGLESCANS").mkdir(parents=True)
    (root / "SCANS/ScanPos001/SINGLESCANS/s1.rxp").write_bytes(b"")
    (root / "DAT").mkdir()
    np.savetxt(root / "DAT/ScanPos002.DAT", np.eye(4) * [1, 1, 1, 1] + 0)
    (root / "project.rsp").write_text(RSP)
    proj = riscan.read_riscan_project(root)
    assert proj.name == "Demo" and proj.names == ["ScanPos001", "ScanPos002"]
    np.testing.assert_allclose(proj.pop[:3, 3], [100, 200, 300])
    p1 = proj["ScanPos001"]
    assert p1.rxp.name == "s1.rxp" and p1.instrument == "VZ-400i"
    assert p1.pattern["theta_count"] == 2512 and p1.pattern["phi_delta"] == 0.04
    np.testing.assert_allclose(p1.sop[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(p1.transform(proj.pop)[:3, 3], [101, 202, 303])
    assert proj["ScanPos002"].rxp is None
    np.testing.assert_allclose(proj["ScanPos002"].sop, np.eye(4))  # DAT fallback
    assert [p.name for p in proj.with_scans()] == ["ScanPos001"]


def test_fill_missing_and_pattern_gap(rng):
    pattern = {"theta_start": 30.0, "theta_delta": 1.0, "theta_count": 100,
               "phi_start": 0.0, "phi_delta": 1.0, "phi_count": 360}
    # Half of the pulses on each line returned a canopy echo at 10 m; the rest went to the sky.
    theta = np.radians(np.repeat(30 + np.arange(100), 180))
    az = np.radians(rng.uniform(0, 360, theta.size))
    d = np.column_stack([np.sin(theta) * np.sin(az), np.sin(theta) * np.cos(az), np.cos(theta)])
    shots = Shots.from_pointcloud(PointCloud(d * 10))
    full = shots.fill_missing(pattern)
    assert full.n_shots == 100 * 360 and full.n_echoes == shots.n_echoes
    heights = np.full(shots.n_echoes, 10.0)
    # Edges placed between zenith lines: a ring edge on a line is ambiguous.
    edges = np.array([0, 29.5, 60.5, 90.5, 130])
    cen, gap = canopy.gap_fraction_pattern(shots, heights, pattern, zenith_edges=edges)
    assert np.isnan(gap[0])
    np.testing.assert_allclose(gap[1:], 0.5, atol=1e-9)
    # The density grid now sees free space above the echoes.
    g = canopy.density_grid(full, 2.0, origin=(-12, -12, -12), shape=(12, 12, 12))
    assert (g.n_rays > g.n_hits).any()


needs_data = pytest.mark.skipif(not ROBSON.is_dir(), reason="set SYLVA_TERN_DIR to the TERN data")


@needs_data
def test_robson_creek_project():
    proj = riscan.read_riscan_project(ROBSON)
    assert len(proj) > 100 and proj.pop is not None
    pos = proj.with_scans()[0]
    assert pos.pattern["theta_count"] == 2512
    raw = io.read_rxp_shots(pos.rxp, max_points=500_000)  # scanner frame
    assert raw.n_shots > 100_000
    zen, _ = raw.zenith_azimuth()
    assert zen.min() > 29.9 and zen.max() < 131  # VZ-2000i theta 30..130
    expected = raw.expected_per_zenith(pos.pattern, np.array([0, 29.5, 130.5]))
    assert expected[0] == 0 and expected[1] == 2512 * 9001
    shots = pos.read_shots(max_points=500_000)  # project frame
    np.testing.assert_allclose(shots.origin[0], pos.sop[:3, 3])
    # Echoes land above the local ground: origin z minus ~2 m at most.
    pc = shots.to_pointcloud()
    assert pc.z.min() > pos.sop[2, 3] - 5
    assert pc.z.max() > pos.sop[2, 3] + 5
