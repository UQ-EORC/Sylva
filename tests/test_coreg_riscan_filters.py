# RiSCAN filters ported from tlsalign's tests, and the trusted reflector matches.
import numpy as np
import pytest

from sylva import coreg
from sylva.riscan import angular_steps, export_settings_mask, read_export_settings, riscan_like_mask


def _synthetic_scan(step=0.03, n_lines=40, n_shots=200, distance=10.0):
    """A wall of points on a regular angular grid, in stream order (line by line)."""
    theta = np.radians(80.0 + step * np.arange(n_shots))
    phi = np.radians(step * np.arange(n_lines))
    tt, pp = np.meshgrid(theta, phi)
    tt, pp = tt.ravel(), pp.ravel()
    xyz = distance * np.column_stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)])
    return xyz.astype(np.float32)


def test_angular_steps_are_recovered_from_stream_order():
    theta_step, phi_step = angular_steps(_synthetic_scan(step=0.03))
    assert abs(theta_step - 0.03) < 0.003
    assert abs(phi_step - 0.03) < 0.003


def test_riscan_current_mode_drops_only_the_near_range():
    xyz = _synthetic_scan()
    near = np.array([[0.1, 0.0, 0.0], [0.3, 0.1, 0.0], [0.7, 0.0, 0.0]], dtype=np.float32)
    all_xyz = np.vstack([xyz, near])
    keep = riscan_like_mask(all_xyz, np.full(len(all_xyz), 15.0, np.float32), "current")
    assert keep[: len(xyz)].all()
    assert list(keep[len(xyz) :]) == [False, False, True]


def test_riscan_legacy_mode_drops_weak_unsupported_echoes_only():
    wall = _synthetic_scan()
    n = len(wall)
    idx = np.random.default_rng(1).choice(n, 6, replace=False)
    all_xyz = np.vstack([wall, wall[idx] * 0.7])  # isolated echoes 3 m in front of the wall
    amplitude = np.full(len(all_xyz), 15.0, dtype=np.float32)
    amplitude[n : n + 3] = 5.0  # three weak, three strong
    keep = riscan_like_mask(all_xyz, amplitude, "legacy", steps=(0.03, 0.03))
    assert keep[:n].all(), "supported wall points must survive"
    assert not keep[n : n + 3].any(), "weak echoes without neighbours are dropped"
    assert keep[n + 3 :].all(), "strong echoes survive even without neighbours"


def test_riscan_mask_rejects_unknown_mode():
    with pytest.raises(ValueError):
        riscan_like_mask(np.zeros((3, 3), np.float32), np.zeros(3, np.float32), "strict")


def test_riscan_export_settings_are_parsed_and_applied(tmp_path):
    f = tmp_path / "export_to_laz_filter_settings.txt"
    f.write_text("deviation, 0, 12\nrange, 2, 100\nreflectance, -20, 5\n")
    settings = read_export_settings(f)
    assert settings == {
        "deviation": (0.0, 12.0),
        "range": (2.0, 100.0),
        "reflectance": (-20.0, 5.0),
    }
    xyz = np.array([[1.0, 0, 0], [10.0, 0, 0], [10.0, 0, 0], [10.0, 0, 0], [200.0, 0, 0]])
    attrs = {
        "deviation": np.array([1, 1, 20, 1, 1]),
        "reflectance": np.array([-5.0, -5.0, -5.0, -30.0, -5.0]),
    }
    assert list(export_settings_mask(settings, xyz, attrs)) == [False, True, False, False, False]
    with pytest.raises(KeyError):
        export_settings_mask(settings, xyz, {"deviation": attrs["deviation"]})
    f.write_text("colour, 0, 1\n")
    with pytest.raises(ValueError):
        read_export_settings(f)


def test_explicit_bounds_override_the_settings_file(tmp_path):
    f = tmp_path / "settings.txt"
    f.write_text("range, 2, 100\ndeviation, 0, 12\n")
    options = coreg.reading_options(f, max_range=30.0, max_deviation=None)
    assert options == {
        "min_range": 2.0,
        "max_range": 30.0,
        "min_deviation": 0.0,
        "max_deviation": 12.0,
    }


# --------------------------------------------------------------------------- #
# Trusted reflector matches
# --------------------------------------------------------------------------- #


def _scan_with_targets(name, targets, surface_offset, rng):
    """A scan whose only shared content is its targets: its ICP points are unrelated."""
    pts = rng.uniform(-10, 10, (20_000, 3)) * [1, 1, 0.05] + surface_offset
    return coreg.ScanFeatures(
        name=name,
        n_points=len(pts),
        ground=None,
        stem_map=coreg.StemMap([], name=name),
        icp_points=pts.astype(np.float32),
        icp_heights=np.full(len(pts), 2.0, np.float32),
        reflectors=[coreg.Reflector(*p) for p in targets],
    )


@pytest.fixture
def far_pair():
    rng = np.random.default_rng(3)
    world = rng.uniform(-20, 20, (6, 3)) * [1, 1, 0.1]
    truth = coreg.yaw_transform(0.6, 30.0, 4.0, 0.3)  # world_from_scan of scan 1
    a = _scan_with_targets("a", world, [0, 0, 0], rng)
    b = _scan_with_targets(
        "b", coreg.transform_points(coreg.invert(truth), world), [200, 0, 0], rng
    )
    return a, b, truth


def test_trusted_reflectors_accept_what_icp_cannot_confirm(far_pair):
    a, b, truth = far_pair
    pair = coreg.register_pair(b, a, i=1, j=0)
    assert pair.success and pair.trusted, pair.reason
    # ICP finds nothing to fit here and stays on the targets' pose.
    rotation, translation = coreg.transform_difference(pair.transform, truth)
    assert translation < 1e-6 and rotation < 1e-8


def test_tlsalign_behaviour_without_trust(far_pair):
    a, b, _ = far_pair
    pair = coreg.register_pair(b, a, coreg.CoregConfig(trusted_reflector_matches=0), i=1, j=0)
    assert not pair.success and not pair.trusted


def test_too_few_targets_are_not_trusted(far_pair):
    a, b, _ = far_pair
    a.reflectors, b.reflectors = a.reflectors[:4], b.reflectors[:4]
    assert not coreg.register_pair(b, a, i=1, j=0).success


def test_pairs_without_stems_still_try_their_targets(far_pair):
    a, b, truth = far_pair
    result = coreg.coregister_prepared([a, b], coreg.CoregConfig(verbose=False))
    assert all(result.registered)
    assert coreg.transform_difference(result.poses[1], truth)[1] < 1e-4
    assert result.pairs[0].trusted


def test_trust_keeps_the_targets_pose_when_icp_drifts():
    from sylva.coreg.pipeline import _trust_reflectors

    targets = coreg.yaw_transform(0.2, 5.0, 1.0, 0.0)
    drifted = coreg.yaw_transform(0.2, 5.4, 1.0, 0.0)  # ICP slid 40 cm
    found = coreg.ReflectorMatch(targets, n_inliers=6, rmse=0.01, success=True)
    pair = coreg.PairResult(
        0, 1, transform=drifted, reflector_match=found, reason="low ICP fitness (0.050 < 0.1)"
    )
    _trust_reflectors(pair, targets, 0.4, coreg.CoregConfig())
    assert pair.success and pair.trusted and not pair.used_icp
    assert np.array_equal(pair.transform, targets)
    assert "low ICP fitness" in pair.reason
