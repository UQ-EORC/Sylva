"""Parity cases for sylva.coreg.refine."""

import numpy as np

from sylva.coreg import refine
from sylva.coreg import transforms as tf

STEMS = np.array([[3.0, 2.0, 0.20], [-4.0, 5.0, 0.30], [6.0, -5.0, 0.25], [-6.0, -3.0, 0.15],
                  [0.5, 8.0, 0.22], [9.0, 4.0, 0.35], [-9.0, 7.0, 0.18], [1.0, -8.0, 0.28]])


def _terrain(xy):
    return 0.04 * xy[:, 0] - 0.02 * xy[:, 1] + 0.15 * np.sin(0.4 * xy[:, 0]) * np.cos(0.3 * xy[:, 1])


def _scene(rng, n_ground, n_stem, n_log):
    """Terrain, stems up to 4 m and a fallen log, as world points."""
    xy = rng.uniform(-12, 12, (n_ground, 2))
    parts = [np.column_stack([xy, _terrain(xy) + rng.normal(0, 0.003, n_ground)])]
    for x, y, r in STEMS:
        a = rng.uniform(0, 2 * np.pi, n_stem)
        h = rng.uniform(0.0, 4.0, n_stem)
        base = _terrain(np.array([[x, y]]))[0]
        parts.append(np.column_stack([x + r * np.cos(a), y + r * np.sin(a), base + h]))
    t = rng.uniform(0, 6, n_log)
    a = rng.uniform(0, 2 * np.pi, n_log)
    c = np.column_stack([-2.0 + 0.8 * t, -1.0 + 0.6 * t])
    parts.append(np.column_stack([c[:, 0] - 0.6 * 0.2 * np.cos(a), c[:, 1] + 0.8 * 0.2 * np.cos(a),
                                  _terrain(c) + 0.2 + 0.2 * np.sin(a)]))
    return np.vstack(parts)


def _survey(seed, n_scans=3, n_ground=45000, n_stem=2500, n_log=2500, error=(0.004, 0.06)):
    rng = np.random.default_rng(seed)
    world = _scene(rng, n_ground, n_stem, n_log)
    truth, start, points, stems = [], [], [], []
    for k in range(n_scans):
        origin = np.array([-4.0 + 4.0 * k, 1.5 * (k % 2), 0.0])
        T = tf.se3_exp(np.concatenate([[0.0, 0.0, rng.uniform(-3, 3)], origin]))
        seen = world[np.linalg.norm(world[:, :2] - origin[:2], axis=1) < 11.0]
        seen = seen[rng.uniform(size=len(seen)) < 0.8]
        local = tf.transform_points(tf.invert(T), seen) + rng.normal(0, 0.002, seen.shape)
        truth.append(T)
        points.append(local)
        near = STEMS[np.linalg.norm(STEMS[:, :2] - origin[:2], axis=1) < 11.0]
        s_world = np.column_stack([near[:, :2], _terrain(near[:, :2]) + 1.3])
        stems.append(tf.transform_points(tf.invert(T), s_world) + rng.normal(0, 0.01, s_world.shape))
        noise = np.concatenate([rng.normal(0, error[0], 3), rng.normal(0, error[1], 3)])
        start.append(tf.se3_exp(noise) @ T if k else T)
    return truth, start, points, stems


def _out(r, prefix, messages=None):
    out = {
        f"{prefix}_poses": np.array(r.poses),
        f"{prefix}_shifts": r.shifts,
        f"{prefix}_rotations": r.rotations,
        f"{prefix}_residual_before": r.residual_before,
        f"{prefix}_residual_after": r.residual_after,
        f"{prefix}_correspondences": r.correspondences,
    }
    if messages is not None:
        out[f"{prefix}_log"] = np.array(messages, dtype="U200")
    return out


def default():
    truth, start, points, stems = _survey(30)
    messages = []
    r = refine.refine_joint(points, start, [(0, 1), (1, 2), (0, 2)], stems, 0, log=messages.append)
    return _out(r, "default", messages)


def sampled():
    """Caps small enough that both of numpy's sampling paths are taken."""
    truth, start, points, stems = _survey(31, n_ground=30000)
    out = {}
    r = refine.refine_joint(points, start, [(0, 1), (1, 2)], stems, 1, points_per_scan=3000,
                            correspondences_per_pair=1000, seed=5, voxel_sizes=(0.3, 0.2),
                            max_distances=(0.6, 0.4))
    out.update(_out(r, "floyd"))
    r = refine.refine_joint(points, start, [(0, 1), (1, 2)], stems, 1, points_per_scan=12000,
                            correspondences_per_pair=150, seed=6, voxel_sizes=(0.05,),
                            max_distances=(0.2,))
    out.update(_out(r, "tail"))
    return out


def options():
    truth, start, points, stems = _survey(32, n_scans=4, n_ground=20000)
    points[3] = np.zeros((0, 3))
    stems[2] = None
    messages = []
    r = refine.refine_joint(points, start, [(0, 1), (1, 2), (2, 3), (0, 0), (1, 3)], stems[:3], 2,
                            rounds=2, iterations=4, min_voxel_points=2, stem_weight=0.2,
                            stem_radius=0.3, stem_scale=0.05, robust_scale=0.01,
                            prior_translation=0.1, prior_rotation_deg=1.0, max_step_translation=0.02,
                            max_step_rotation_deg=0.2, max_normal_angle_deg=30.0,
                            min_planarity=0.4, normal_neighbours=12, log=messages.append)
    out = _out(r, "options", messages)
    # Levels too fine for the gaps between scans: no correspondences.
    messages = []
    r = refine.refine_joint(points[:2], [start[0], start[1] @ tf.se3_exp(np.r_[0, 0, 0, 3.0, 0, 0])],
                            [(0, 1)], stems[:2], 0, voxel_sizes=(0.1,), max_distances=(0.05,),
                            log=messages.append)
    out.update(_out(r, "none", messages))
    return out


CASES = {"default": default, "sampled": sampled, "options": options}
