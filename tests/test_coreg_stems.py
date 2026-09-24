# Ported from tlsalign's tests/test_stems.py.
# tlsalign's circle-fit tests (fit_circle_algebraic, fit_circle_ransac) are not
# ported: Sylva fits circles inside the Rust detector and exposes no such function.
import numpy as np

from sylva.coreg import (
    KdTree,
    StemDetectionConfig,
    StemMap,
    detect_stems,
    fit_ground,
    stem_map_from_arrays,
    transform_points,
    yaw_transform,
)


def test_detect_stems_recovers_the_simulated_plot(survey):
    """Detection must find most visible trees with no false positives."""
    cloud = survey.clouds[0]
    ground = fit_ground(cloud.xyz, 0.5)
    stem_map = detect_stems(cloud.xyz, ground, name="scan")
    assert len(stem_map) > 15

    truth_world = survey.plot.stem_positions[survey.visible[0]]
    detected_world = transform_points(survey.true_transforms[0], stem_map.positions)
    distance, index = KdTree(truth_world[:, :2]).query(detected_world[:, :2])

    matched = distance < 0.30
    assert matched.mean() > 0.9, "too many false positives"
    assert len(set(index[matched])) / len(truth_world) > 0.85, "recall too low"
    assert np.median(distance[matched]) < 0.05

    true_dbh = survey.plot.diameters[survey.visible[0]][index[matched]]
    assert np.abs(stem_map.diameters[matched] - true_dbh).mean() < 0.04


def test_detected_z_is_an_elevation_not_a_height(survey):
    """Stem z must live in the scan's vertical datum, or dZ is unrecoverable."""
    cloud = survey.clouds[0]
    ground = fit_ground(cloud.xyz, 0.5)
    stem_map = detect_stems(cloud.xyz, ground)
    terrain = ground.height_at(stem_map.xy)
    assert np.allclose(stem_map.positions[:, 2] - terrain, 1.3, atol=1e-6)
    assert stem_map.positions[:, 2].std() > 0.05, "z collapsed onto a single plane"


def test_detect_stems_on_an_empty_cloud():
    assert len(detect_stems(np.zeros((10, 3)))) == 0


def test_detect_stems_respects_the_diameter_range(survey):
    cloud = survey.clouds[0]
    config = StemDetectionConfig(min_radius=0.15, max_radius=0.25)
    stem_map = detect_stems(cloud.xyz, fit_ground(cloud.xyz, 0.5), config)
    assert np.all(stem_map.diameters >= 0.29)
    assert np.all(stem_map.diameters <= 0.51)


def test_stem_map_transform_moves_positions_and_axes():
    stem_map = stem_map_from_arrays([[0.0, 0.0], [3.0, 4.0]], dbh=[0.2, 0.3])
    T = yaw_transform(np.pi / 2, 1.0, 2.0, 3.0)
    moved = stem_map.transformed(T)
    assert np.allclose(moved.positions, transform_points(T, stem_map.positions))
    assert np.allclose(moved.axes, stem_map.axes)  # yaw leaves a vertical axis alone


def test_stem_map_top_keeps_the_best(survey):
    cloud = survey.clouds[0]
    stem_map = detect_stems(cloud.xyz, fit_ground(cloud.xyz, 0.5))
    best = stem_map.top(5)
    assert len(best) == 5
    assert best.qualities.min() >= stem_map.top(len(stem_map)).qualities[4] - 1e-12


def test_stem_map_roundtrip(tmp_path, survey):
    cloud = survey.clouds[0]
    stem_map = detect_stems(cloud.xyz, fit_ground(cloud.xyz, 0.5), name="scan_00")
    back = StemMap.load(stem_map.save(tmp_path / "stems.json"))
    assert len(back) == len(stem_map)
    assert np.allclose(back.positions, stem_map.positions)
    assert back.name == "scan_00"
