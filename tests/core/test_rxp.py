# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""RIEGL ``.rxp`` reading checked against RiVLib itself.

The reference reads the same file through RiVLib's C interface
(``scanifc_point3dstream_*``) with ctypes, and applies the documented
options with NumPy. The tests need RiVLib and a scan, which cannot be
distributed: give the scan with ``SYLVA_TEST_RXP``, or the folder of the TERN
RiSCAN projects with ``SYLVA_TERN_DIR`` (the smallest full scan found there is
used). Without either they are skipped.
"""

import ctypes
import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from sylva import io, riscan

XYZ = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
ATTR = np.dtype([("amplitude", "<f4"), ("reflectance", "<f4"), ("deviation", "<u2"),
                 ("flags", "<u2"), ("background", "<f4")])     # scanifc_attributes, 16 bytes
CHUNK = 1 << 18


def _find_scan() -> Path | None:
    if os.environ.get("SYLVA_TEST_RXP"):
        return Path(os.environ["SYLVA_TEST_RXP"])
    tern = Path(os.environ.get("SYLVA_TERN_DIR", "TERN_TLS_RAW"))
    scans = [p for p in tern.glob("*.RiSCAN/SCANS/*/SINGLESCANS/*.rxp")
             if ".residual." not in p.name and ".mon." not in p.name]
    return min(scans, key=lambda p: p.stat().st_size) if scans else None


def _find_library() -> Path | None:
    try:
        return io.find_rivlib()
    except ValueError:
        return None


SCAN = _find_scan()
LIBRARY = _find_library()
pytestmark = [
    pytest.mark.skipif(LIBRARY is None,
                       reason="RiVLib (libscanifc) is not installed; set RIVLIB_PATH"),
    pytest.mark.skipif(SCAN is None or not SCAN.is_file(),
                       reason="no .rxp scan: set SYLVA_TEST_RXP or SYLVA_TERN_DIR"),
]


class RiVLib:
    """RiVLib's point stream, read with ctypes."""

    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))

    def chunks(self, path: Path):
        """Yield ``(xyz, attributes, time_ns)`` record arrays in file order."""
        handle = ctypes.c_void_p()
        uri = f"file:{Path(path).resolve()}".encode()
        assert self.lib.scanifc_point3dstream_open(uri, 0, ctypes.byref(handle)) == 0
        try:
            while True:
                xyz, attr, t = np.empty(CHUNK, XYZ), np.empty(CHUNK, ATTR), np.empty(CHUNK, "<u8")
                got, eof = ctypes.c_uint32(), ctypes.c_int32()
                ptr = [a.ctypes.data_as(ctypes.c_void_p) for a in (xyz, attr, t)]
                rc = self.lib.scanifc_point3dstream_read(handle, CHUNK, *ptr, ctypes.byref(got),
                                                         ctypes.byref(eof))
                assert rc == 0
                n = got.value
                if n:
                    yield xyz[:n], attr[:n], t[:n]
                if n == 0 and eof.value == 0:
                    return
        finally:
            self.lib.scanifc_point3dstream_close(handle)

    def records(self, path: Path, at_least: int):
        """The first records (at least ``at_least``), with 1-based record and pulse numbers."""
        parts, n = [], 0
        for part in self.chunks(path):
            parts.append(part)
            n += len(part[0])
            if n >= at_least:
                break
        xyz = np.concatenate([p[0] for p in parts])
        attr = np.concatenate([p[1] for p in parts])
        t = np.concatenate([p[2] for p in parts])
        xyz = np.column_stack([xyz["x"], xyz["y"], xyz["z"]]).astype(np.float64)
        pulse = np.cumsum(np.concatenate([[True], t[1:] != t[:-1]]))
        return {"xyz": xyz, "attr": attr, "time": t, "record": np.arange(1, len(t) + 1),
                "pulse": pulse}


@pytest.fixture(scope="module")
def rivlib():
    return RiVLib(LIBRARY)


@pytest.fixture(scope="module")
def head(rivlib):
    """The first two million records of the scan."""
    return rivlib.records(SCAN, 2_000_000)


def _keep(ref, *, min_range=0.5, max_range=np.inf, drop_pseudo=True, stride=1, shot_stride=1,
          echoes="all"):
    """The documented reading options, applied to RiVLib's records."""
    r = np.linalg.norm(ref["xyz"], axis=1)
    flags = ref["attr"]["flags"]
    kind = flags & 3                                   # 0 single, 1 first, 2 interior, 3 last
    keep = (r >= min_range) & (r <= max_range)
    if drop_pseudo:
        keep &= (flags & (1 << 4)) == 0
    if stride > 1:
        keep &= ref["record"] % stride == 0
    if shot_stride > 1:
        keep &= ref["pulse"] % shot_stride == 0
    keep &= {"all": np.ones_like(keep), "first": (kind == 0) | (kind == 1),
             "last": (kind == 0) | (kind == 3), "single": kind == 0}[echoes]
    return np.flatnonzero(keep)


def _assert_same_points(cloud, ref, idx):
    np.testing.assert_array_equal(cloud.xyz, ref["xyz"][idx])
    a = ref["attr"][idx]
    np.testing.assert_array_equal(cloud.attrs["amplitude"], a["amplitude"])
    np.testing.assert_array_equal(cloud.attrs["reflectance"], a["reflectance"])
    np.testing.assert_array_equal(cloud.attrs["deviation"], a["deviation"])
    np.testing.assert_array_equal(cloud.attrs["echo_type"], a["flags"] & 3)
    np.testing.assert_array_equal(cloud.attrs["gps_time"], ref["time"][idx] * 1e-9)


def test_every_record_and_attribute_as_rivlib_reads_it(head):
    n = 1_000_000
    cloud = io.read_rxp(SCAN, library=LIBRARY, drop_pseudo_echoes=False, min_range=0.0,
                        max_points=n)
    assert len(cloud) == n
    assert cloud.attrs["amplitude"].dtype == np.float32
    assert cloud.attrs["deviation"].dtype == np.uint16
    _assert_same_points(cloud, head, np.arange(n))


@pytest.mark.parametrize("options", [
    {},                                              # min_range 0.5, pseudo echoes dropped
    {"min_range": 5.0, "max_range": 20.0},
    {"echoes": "first"}, {"echoes": "last"}, {"echoes": "single"},
    {"stride": 7},
    {"shot_stride": 5},
])
def test_reading_options_select_the_documented_records(head, options):
    n = 100_000
    cloud = io.read_rxp(SCAN, max_points=n, **options)
    rename = {"drop_pseudo_echoes": "drop_pseudo"}
    idx = _keep(head, **{rename.get(k, k): v for k, v in options.items()})[:n]
    assert len(idx) == n, "the reference read too few records"
    _assert_same_points(cloud, head, idx)


def test_pulses_are_runs_of_echoes_sharing_a_time(head):
    n = 200_000
    shots = io.read_rxp_shots(SCAN, max_points=n)
    idx = _keep(head)[:n]
    t = head["time"][idx]
    run = np.concatenate([[True], t[1:] != t[:-1]])
    counts = np.diff(np.append(np.flatnonzero(run), len(t)))
    assert shots.n_shots == run.sum() and shots.n_echoes == n
    np.testing.assert_array_equal(shots.echo_count, counts)
    assert np.all(shots.origin == 0.0)
    # Echoes of a pulse are ordered by range; the direction points at the farthest.
    xyz = head["xyz"][idx]
    r = np.linalg.norm(xyz, axis=1)
    pulse = np.cumsum(run) - 1
    order = np.lexsort((r, pulse))
    np.testing.assert_allclose(shots.echo_range, r[order], rtol=1e-12)
    last = np.cumsum(counts) - 1
    np.testing.assert_allclose(shots.direction, xyz[order][last] / r[order][last, None], atol=1e-12)
    # Every pulse of RiVLib's stream is one shot, however many echoes it has.
    multi = counts > 1
    assert multi.any() and np.all(head["attr"]["flags"][idx][run][multi] & 3 == 1)


def _position() -> riscan.ScanPosition:
    """The scan's position in its RiSCAN project (skips if it has none)."""
    try:
        pos = riscan.read_riscan_project(SCAN.parents[3])[SCAN.parents[1].name]
    except (OSError, ValueError, KeyError, IndexError):
        pos = None
    if pos is None or pos.pattern is None or pos.sop is None:
        pytest.skip(f"{SCAN} is not in a RiSCAN project with a SOP and a scan pattern")
    return pos


def test_scan_position_reads_in_project_coordinates():
    pos = _position()
    n = 50_000
    raw = io.read_rxp(SCAN, max_points=n)
    np.testing.assert_allclose(pos.read(max_points=n).xyz,
                               raw.xyz @ pos.sop[:3, :3].T + pos.sop[:3, 3], atol=1e-9)
    shots = pos.read_shots(max_points=n)
    np.testing.assert_allclose(shots.origin, np.tile(pos.sop[:3, 3], (shots.n_shots, 1)))
    # Echoes are placed along their pulse's one direction (to its farthest echo); RIEGL's
    # echoes of one pulse are collinear with the scanner origin to a few millimetres.
    single = np.repeat(shots.echo_count == 1, shots.echo_count)
    np.testing.assert_allclose(shots.echo_xyz()[single], pos.read(max_points=n).xyz[single],
                               atol=1e-6)
    np.testing.assert_allclose(shots.echo_xyz(), pos.read(max_points=n).xyz, atol=5e-3)
    # The misses are added in the scanner frame, where the pattern's lines are, then moved.
    filled = pos.read_shots(fill_missing=True, max_points=n, shot_stride=8)
    own = io.read_rxp_shots(SCAN, max_points=n, shot_stride=8).fill_missing(pos.pattern,
                                                                           shot_stride=8)
    assert filled.n_shots == own.n_shots > shots.n_shots // 8
    np.testing.assert_allclose(filled.direction, own.direction @ pos.sop[:3, :3].T, atol=1e-12)
    with pytest.raises(ValueError, match=f"scan position {pos.name} has no scan pattern"):
        dataclasses.replace(pos, pattern=None).read_shots(fill_missing=True, max_points=10)


def test_misses_follow_from_the_scan_pattern(rivlib):
    """A whole scan read every 16th pulse, then its no-return pulses added.

    The reference counts RiVLib's pulses per zenith line of the pattern and
    tops each line up to the pulses fired: the larger of the nominal
    ``phi_count / 16`` and the 0.98 quantile of the line counts.
    """
    pattern = _position().pattern
    stride = 16
    # Reference: RiVLib's records, the farthest kept echo of every 16th pulse.
    last_t, pulse0, kept = None, 0, []
    for xyz, attr, t in rivlib.chunks(SCAN):
        new = np.concatenate([[t[0] != last_t], t[1:] != t[:-1]])
        pulse = pulse0 + np.cumsum(new)
        pulse0, last_t = pulse[-1], t[-1]
        p = np.column_stack([xyz["x"], xyz["y"], xyz["z"]]).astype(np.float64)
        r = np.linalg.norm(p, axis=1)
        keep = (pulse % stride == 0) & (r >= 0.5) & ((attr["flags"] & (1 << 4)) == 0)
        kept.append((p[keep], r[keep], pulse[keep]))
    p, r, pulse = (np.concatenate([k[i] for k in kept]) for i in range(3))
    order = np.lexsort((r, pulse))
    far = np.append(pulse[order][1:] != pulse[order][:-1], True)
    zenith = np.degrees(np.arccos(np.clip(p[order][far][:, 2] / r[order][far], -1, 1)))
    lines = pattern["theta_start"] + pattern["theta_delta"] * np.arange(pattern["theta_count"])
    edges = np.append(lines - pattern["theta_delta"] / 2, lines[-1] + pattern["theta_delta"] / 2)
    observed = np.histogram(zenith, edges)[0]
    per_line = int(max(pattern["phi_count"] / stride, np.quantile(observed, 0.98)))
    missing = np.maximum(per_line - observed, 0)

    shots = io.read_rxp_shots(SCAN, shot_stride=stride)
    assert shots.n_shots == len(zenith)
    full = shots.fill_missing(pattern, shot_stride=stride)
    assert full.n_echoes == shots.n_echoes
    added = full.echo_count[shots.n_shots:]
    assert len(added) == missing.sum() and np.all(added == 0)
    got = np.histogram(np.degrees(np.arccos(full.direction[shots.n_shots:, 2])), edges)[0]
    np.testing.assert_array_equal(got, missing)
    # Every line that had fewer now holds exactly the pulses fired.
    after = np.histogram(np.degrees(np.arccos(np.clip(full.direction[:, 2], -1, 1))), edges)[0]
    np.testing.assert_array_equal(after, np.maximum(observed, per_line))
