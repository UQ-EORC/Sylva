"""Parity cases for sylva.coreg.transforms."""

import numpy as np

from sylva.coreg import transforms as tf


def _rotations(rng):
    """Rotation vectors: general, tiny, zero and near pi (exactly pi has no
    defined sign, so rounding decides it)."""
    w = [rng.normal(0, 1, 3) for _ in range(6)]
    w += [rng.normal(0, 1e-9, 3), np.zeros(3)]
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    w += [axis * (np.pi - 1e-8), axis * (np.pi - 1e-7), np.array([0.0, 0.0, np.pi - 5e-7]), axis * 3.0]
    return w


def so3():
    rng = np.random.default_rng(1)
    out = {}
    for k, w in enumerate(_rotations(rng)):
        R = tf.so3_exp(w)
        out[f"exp_{k}"] = R
        out[f"log_{k}"] = tf.so3_log(R)
        out[f"skew_{k}"] = tf.skew(w)
        out[f"angle_{k}"] = tf.rotation_angle(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]))
    # A rotation matrix with rounding noise, and one from a 4x4.
    out["log_noisy"] = tf.so3_log(tf.so3_exp([0.3, -0.2, 0.1]) + 1e-12)
    out["log_of_4x4"] = tf.so3_log(tf.se3_exp([0.1, 0.2, 0.3, 1.0, 2.0, 3.0]))
    out["identity"] = tf.identity()
    return out


def se3():
    rng = np.random.default_rng(2)
    out = {}
    for k, w in enumerate(_rotations(rng)):
        xi = np.concatenate([w, rng.normal(0, 5, 3)])
        T = tf.se3_exp(xi)
        out[f"exp_{k}"] = T
        out[f"log_{k}"] = tf.se3_log(T)
        out[f"inv_{k}"] = tf.invert(T)
        U = tf.se3_exp(rng.normal(0, 0.3, 6))
        rot, tr = tf.transform_difference(T, U)
        out[f"diff_{k}"] = np.array([rot, tr])
    out["yaw"] = tf.yaw_transform(0.7, 1.0, -2.0, 0.5)
    out["yaw_default"] = tf.yaw_transform(-2.5)
    out["exp_list"] = tf.se3_exp([0.01, 0.02, -0.03, 1, 2, 3])
    return out


def points():
    rng = np.random.default_rng(3)
    T = tf.se3_exp(rng.normal(0, 1, 6))
    pts = rng.normal(0, 30, (1000, 3))
    return {
        "points": tf.transform_points(T, pts),
        "vectors": tf.transform_vectors(T, pts[:50]),
        "empty_points": tf.transform_points(T, np.zeros((0, 3))),
        "empty_vectors": tf.transform_vectors(T, np.zeros((0, 3))),
        "list_points": tf.transform_points(T, [[1.0, 2.0, 3.0]]),
    }


def kabsch():
    rng = np.random.default_rng(4)
    out = {}
    for k, n in enumerate([3, 4, 7, 30, 500]):
        src = rng.normal(0, 10, (n, 3))
        T = tf.se3_exp(rng.normal(0, 1, 6))
        dst = tf.transform_points(T, src) + rng.normal(0, 0.01, (n, 3))
        w = rng.uniform(0.1, 2.0, n)
        out[f"plain_{k}"] = tf.kabsch(src, dst)
        out[f"weighted_{k}"] = tf.kabsch(src, dst, w)
        out[f"yaw_plain_{k}"] = tf.kabsch_2d_yaw(src, dst)
        out[f"yaw_weighted_{k}"] = tf.kabsch_2d_yaw(src, dst, w)
    # A reflected target: the proper rotation closest to it.
    src = rng.normal(0, 5, (10, 3))
    out["reflection"] = tf.kabsch(src, src * np.array([1.0, 1.0, -1.0]))
    out["yaw_two"] = tf.kabsch_2d_yaw(src[:2], src[:2] + 1.0)
    errors = []
    for args in [
        (src[:2], src[:2]),
        (src, src[:5]),
        (src[:, :2], src[:, :2]),
    ]:
        try:
            tf.kabsch(*args)
            errors.append(0)
        except ValueError:
            errors.append(1)
    for args in [(src, src, np.zeros(10))]:
        for fn in (tf.kabsch, tf.kabsch_2d_yaw):
            try:
                fn(*args)
                errors.append(0)
            except ValueError:
                errors.append(1)
    try:
        tf.kabsch_2d_yaw(src[:1], src[:1])
        errors.append(0)
    except ValueError:
        errors.append(1)
    out["errors"] = np.array(errors)
    return out


CASES = {"so3": so3, "se3": se3, "points": points, "kabsch": kabsch}
