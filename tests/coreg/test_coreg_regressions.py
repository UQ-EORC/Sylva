"""Regressions in sylva.coreg: stem elevations from heights, and scans that cannot be converted."""

import numpy as np
import pytest

from sylva import coreg


def test_stems_from_heights_alone_stand_on_the_terrain():
    """With heights but no terrain model, a stem's z is still its elevation in the scan's frame."""
    plot = coreg.simulate_plot(seed=3, slope=0.15)
    cloud, _, _ = coreg.scan_plot(plot, np.array([2.0, 1.0]), seed=3)
    xyz = cloud.xyz + np.array([0.0, 0.0, 250.0])
    ground = coreg.fit_ground(xyz)
    heights = ground.normalise(xyz)
    with_ground = coreg.detect_stems(xyz, ground=ground)
    from_heights = coreg.detect_stems(xyz, heights=heights)
    assert len(with_ground) == len(from_heights) > 10
    np.testing.assert_allclose(from_heights.positions[:, :2], with_ground.positions[:, :2])
    np.testing.assert_allclose(from_heights.positions[:, 2], with_ground.positions[:, 2], atol=0.15)
    # The terrain slopes: the stems are not all at one height.
    assert np.ptp(from_heights.positions[:, 2]) > 0.5
    with pytest.raises(ValueError, match="heights for"):
        coreg.detect_stems(xyz, heights=heights[:-1])


def test_a_scan_that_cannot_be_converted_costs_that_scan_only():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 5, (2000, 3))
    cfg = coreg.CoregConfig(verbose=False)
    # A target that is not a Reflector, on the first scan.
    r = coreg.coregister([a, a + 1.0], cfg, reflectors=[[(1.0, 2.0, 3.0)], []])
    assert len(r.scans) == 2
    assert "AttributeError" in r.scans[0].error and r.scans[1].error == ""
    # A levelling that is not a matrix, on the second.
    r = coreg.coregister([a, a + 1.0], cfg, levelling=[np.eye(4), [[1.0, 2.0], [3.0]]])
    assert r.scans[0].error == "" and "ValueError" in r.scans[1].error
    r = coreg.coregister([a, a + 1.0], cfg, levelling=[np.eye(3), None])
    assert "(4, 4)" in r.scans[0].error and r.scans[1].error == ""


def test_coregister_wants_one_entry_per_scan():
    a = np.random.default_rng(0).uniform(0, 5, (2000, 3))
    cfg = coreg.CoregConfig(verbose=False)
    with pytest.raises(ValueError, match="levelling has 1 entries for 2 scans"):
        coreg.coregister([a, a + 1.0], cfg, levelling=[np.eye(4)])
    with pytest.raises(ValueError, match="reflectors has 3 entries for 2 scans"):
        coreg.coregister([a, a + 1.0], cfg, reflectors=[[], [], []])
