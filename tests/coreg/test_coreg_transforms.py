# Tests of the SE(3) transform helpers.
import numpy as np
import pytest

from sylva.coreg import transforms as tf


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def test_se3_exp_log_roundtrip(rng):
    for _ in range(200):
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * rng.uniform(0, np.pi * 0.99)
        xi = np.concatenate([w, rng.normal(size=3) * 5])
        assert np.allclose(tf.se3_log(tf.se3_exp(xi)), xi, atol=1e-9)


def test_so3_log_near_pi():
    for axis in np.eye(3):
        angle = np.pi - 1e-8
        assert np.allclose(tf.so3_log(tf.so3_exp(axis * angle)), axis * angle, atol=1e-4)
        assert np.allclose(tf.so3_log(tf.so3_exp(-axis * angle)), -axis * angle, atol=1e-4)


def test_so3_exp_is_a_rotation(rng):
    R = tf.so3_exp(rng.normal(size=3))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_invert_matches_matrix_inverse(rng):
    T = tf.se3_exp(rng.normal(size=6))
    assert np.allclose(tf.invert(T) @ T, np.eye(4), atol=1e-12)
    assert np.allclose(tf.invert(T), np.linalg.inv(T), atol=1e-10)


def test_kabsch_recovers_a_known_transform(rng):
    points = rng.normal(size=(60, 3)) * 4
    T = tf.se3_exp(np.concatenate([rng.normal(size=3) * 0.3, rng.normal(size=3) * 5]))
    assert np.allclose(tf.kabsch(points, tf.transform_points(T, points)), T, atol=1e-9)


def test_kabsch_rejects_reflections():
    """A mirrored point set must still yield a proper rotation, not a flip."""
    source = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    target = source * np.array([1.0, 1.0, -1.0])
    T = tf.kabsch(source, target)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def test_kabsch_2d_yaw_recovers_yaw_only(rng):
    points = rng.normal(size=(40, 3)) * 6
    T = tf.yaw_transform(0.85, 3.0, -2.0, 0.4)
    estimated = tf.kabsch_2d_yaw(points, tf.transform_points(T, points))
    assert np.allclose(estimated, T, atol=1e-9)


def test_kabsch_2d_yaw_ignores_tilt(rng):
    """Roll/pitch in the data must not leak into a 4-DoF estimate."""
    points = rng.normal(size=(40, 3)) * 6
    tilted = tf.se3_exp(np.array([0.05, -0.03, 0.0, 0, 0, 0]))
    T = tf.yaw_transform(0.4, 1.0, 2.0, 0.0) @ tilted
    estimated = tf.kabsch_2d_yaw(points, tf.transform_points(T, points))
    assert np.allclose(
        estimated[:3, :3],
        tf.yaw_transform(np.arctan2(estimated[1, 0], estimated[0, 0]))[:3, :3],
        atol=1e-9,
    )
    assert abs(estimated[2, 0]) < 1e-9 and abs(estimated[2, 1]) < 1e-9


def test_kabsch_requires_enough_points():
    with pytest.raises(ValueError):
        tf.kabsch(np.zeros((2, 3)), np.zeros((2, 3)))


def test_transform_points_handles_empty():
    assert tf.transform_points(np.eye(4), np.zeros((0, 3))).shape == (0, 3)


def test_transform_difference_is_zero_for_identical():
    T = tf.se3_exp(np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0]))
    rotation, translation = tf.transform_difference(T, T)
    assert rotation < 1e-12 and translation < 1e-12
