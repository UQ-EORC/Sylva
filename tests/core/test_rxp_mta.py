"""RXP reading with MTA resolution against RiSCAN PRO's own conversion.

Needs RiVLib, the sylva_rivlib shim and a scan with both files:

    SYLVA_TEST_RXP=/path/ScanPosNNN.SCNPOS/scans/<ts>.rxp
    SYLVA_TEST_RDBX=/path/project.rdb/SCANS/ScanPosNNN/SINGLESCANS/<ts>/<ts>.rdbx

Skipped otherwise. The reference scan while developing was a VZ-600i scan
(Pilliga 2026-10-13, ScanPos001), where the first 300 lines agree with rimta5
on 99.1% of the point set and 99.5% of the zone choices.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from sylva import io

LINES = 300
pytestmark = pytest.mark.skipif(
    not (os.environ.get("SYLVA_TEST_RXP") and os.environ.get("SYLVA_TEST_RDBX")),
    reason="SYLVA_TEST_RXP / SYLVA_TEST_RDBX not set",
)


@pytest.fixture(scope="module")
def shim():
    try:
        return io.find_rivlib_shim()
    except ValueError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def ours(shim):
    return io.read_rxp(os.environ["SYLVA_TEST_RXP"], mta="on", max_lines=LINES)


@pytest.fixture(scope="module")
def riscan(ours):
    # the rdbx line index is negative: line L of the scan is -(L+1)
    pc = io.read_rdbx(
        os.environ["SYLVA_TEST_RDBX"],
        attributes=["riegl.timestamp", "riegl.mta_zone", "riegl.scan_line_index"],
        filter=f"riegl.scan_line_index >= -{LINES}",
    )
    return pc


def _keys(pc):
    t = np.asarray(pc.attrs["gps_time"], dtype=np.float64)
    r = np.linalg.norm(np.asarray(pc.xyz), axis=1)
    return t, r


def _match(ta, ra, tb, rb, dt=1e-7, dr=0.002):
    """For every point of a: is there a point of b with the same pulse time and range?"""
    order = np.argsort(tb, kind="stable")
    tb, rb = tb[order], rb[order]
    hit = np.zeros(len(ta), bool)
    lo = np.searchsorted(tb, ta - dt)
    hi = np.searchsorted(tb, ta + dt)
    for k in range(int((hi - lo).max())):
        j = lo + k
        ok = j < hi
        ok[ok] &= np.abs(rb[j[ok]] - ra[ok]) < dr
        hit |= ok
    return hit


def test_point_set_matches_riscan(ours, riscan):
    ta, ra = _keys(ours)
    tb = np.asarray(riscan.attrs["gps_time"], dtype=np.float64)
    rb = np.linalg.norm(np.asarray(riscan.xyz), axis=1)
    # scan-line subsets may differ by a line at the edge: compare over the common time span
    t0, t1 = max(ta.min(), tb.min()), min(ta.max(), tb.max())
    a = (ta >= t0) & (ta <= t1)
    b = (tb >= t0) & (tb <= t1)
    ours_in_riscan = _match(ta[a], ra[a], tb[b], rb[b]).mean()
    riscan_in_ours = _match(tb[b], rb[b], ta[a], ra[a]).mean()
    assert ours_in_riscan > 0.985, f"only {ours_in_riscan:.4f} of our points are in RiSCAN's cloud"
    assert riscan_in_ours > 0.985, f"only {riscan_in_ours:.4f} of RiSCAN's points are in ours"


def test_attributes_and_zones(ours):
    names = ("amplitude", "reflectance", "deviation", "gps_time", "mta_zone", "return_number",
             "number_of_returns", "scan_line_index", "shot_index_line", "mirror_facet")
    for name in names:
        assert name in ours.attrs, name
    z = np.asarray(ours.attrs["mta_zone"])
    assert z.min() >= 1 and (z == 1).mean() > 0.9
    line = np.asarray(ours.attrs["scan_line_index"])
    assert line.max() <= -1 and line.min() >= -LINES - 1
    rn = np.asarray(ours.attrs["return_number"])
    nr = np.asarray(ours.attrs["number_of_returns"])
    assert np.all(rn >= 1) and np.all(rn <= nr)


def test_keep_uncertain_adds_points(shim):
    strict = io.read_rxp(os.environ["SYLVA_TEST_RXP"], mta="on", max_lines=40)
    loose = io.read_rxp(os.environ["SYLVA_TEST_RXP"], mta="on", max_lines=40, keep_uncertain=True)
    assert len(loose) > len(strict)
    flag = np.asarray(loose.attrs["mta_uncertain_point"])
    assert flag.sum() == len(loose) - len(strict)


def test_rule_and_model_decisions_differ_but_agree_mostly(shim):
    rule = io.read_rxp(os.environ["SYLVA_TEST_RXP"], mta="on", max_lines=40, mta_model="rule")
    model = io.read_rxp(os.environ["SYLVA_TEST_RXP"], mta="on", max_lines=40)
    assert abs(len(rule) - len(model)) < 0.05 * len(rule)
    with pytest.raises(ValueError):
        io.read_rxp(os.environ["SYLVA_TEST_RXP"], mta="on", max_lines=40,
                    mta_model="/no/such/model.json")


def test_mta_off_is_the_plain_reader(shim):
    plain = io.read_rxp(os.environ["SYLVA_TEST_RXP"], mta="off", max_points=1000)
    assert "mta_zone" not in plain.attrs and len(plain) == 1000
