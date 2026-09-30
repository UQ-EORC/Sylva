# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""RiSCAN projects: indexing positions, their reflectors, and what reading a
position without a scan reports."""

import json

import numpy as np
import pytest
from test_riscan import RSP

from sylva import riscan


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "Demo.RiSCAN"
    (root / "SCANS/ScanPos001/SINGLESCANS").mkdir(parents=True)
    (root / "SCANS/ScanPos001/SINGLESCANS/s1.rxp").write_bytes(b"")
    (root / "project.rsp").write_text(RSP)
    return riscan.read_riscan_project(root)


def test_positions_by_number_and_name(project):
    assert len(project) == 2 and [p.name for p in project] == project.names
    assert project[0] is project["ScanPos001"]
    assert project[-1] is project["ScanPos002"]
    with pytest.raises(KeyError, match="ScanPos009"):
        project["ScanPos009"]
    with pytest.raises(IndexError):
        project[2]


def test_reading_a_position_without_a_scan_says_which(project):
    empty = project["ScanPos002"]
    with pytest.raises(FileNotFoundError, match="scan position ScanPos002 has no .rxp"):
        empty.read()
    with pytest.raises(FileNotFoundError, match="scan position ScanPos002 has no .rxp"):
        empty.read_shots(fill_missing=True)


def test_reflectors_come_from_the_tiepoint_list(project, tmp_path):
    pos = project["ScanPos001"]
    assert pos.tiepoints is None and pos.reflectors() == []
    tpl = tmp_path / "targets.tpl"
    tpl.write_text(json.dumps([
        {"name": "TP01", "reflectance": 20.5, "diameter": 0.1, "pointcount": 300,
         "positionCartesian": {"x": 1.0, "y": 2.0, "z": -0.5}},
        {"name": "TP02", "reflectance": 18.0, "diameter": 0.1, "pointcount": 120,
         "positionCartesian": {"x": -3.0, "y": 0.25, "z": 0.75}},
    ]))
    pos.tiepoints = tpl
    got = pos.reflectors()
    assert [r.name for r in got] == ["TP01", "TP02"]
    np.testing.assert_allclose([r.position for r in got], [[1.0, 2.0, -0.5], [-3.0, 0.25, 0.75]])
    # In the scanner's own frame: the SOP is not applied.
    assert pos.sop is not None and not np.allclose(pos.sop, np.eye(4))


def test_transform_composes_pop_after_sop(project):
    pos = project["ScanPos001"]
    np.testing.assert_allclose(pos.transform(), pos.sop)
    np.testing.assert_allclose(pos.transform(project.pop), project.pop @ pos.sop)
    np.testing.assert_allclose(pos.origin, pos.sop[:3, 3])
    empty = riscan.ScanPosition("x", None, None)
    np.testing.assert_allclose(empty.transform(), np.eye(4))
    assert empty.origin is None and empty.levelling is None
