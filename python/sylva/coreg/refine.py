# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Joint multi-view refinement: every pose solved at once from raw correspondences.

The pose graph sees each accepted pair as one relative transform, which throws
away the geometry behind it. This keeps every point-to-plane correspondence of
every accepted pair, adds the matched stems as horizontal constraints, and
solves all poses together with a robust Gauss-Newton, the reference fixed, so
that errors are spread by the data rather than handed from scan to scan.
The solve runs in the Rust core.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .. import _core

__all__ = ["JointRefinement", "refine_joint"]


class JointRefinement:
    """Result of :func:`refine_joint`: the poses and how far each moved.

    Attributes
    ----------
    poses
        Refined ``world_from_scan``.
    shifts, rotations
        How far each scan moved (m, degrees).
    residual_before, residual_after
        Median absolute point-to-plane residual (m).
    correspondences
        Largest number of correspondences in one association.
    """

    def __init__(self, poses, shifts, rotations, residual_before, residual_after, correspondences):
        self.poses = poses
        self.shifts = shifts
        self.rotations = rotations
        self.residual_before = residual_before
        self.residual_after = residual_after
        self.correspondences = correspondences


def refine_joint(
    points: list[np.ndarray],
    poses: list[np.ndarray],
    edges: list[tuple[int, int]],
    stems: list[np.ndarray],
    reference: int,
    *,
    voxel_sizes: tuple[float, ...] = (0.10, 0.05, 0.03),
    max_distances: tuple[float, ...] = (0.30, 0.15, 0.08),
    rounds: int = 3,
    iterations: int = 3,
    points_per_scan: int = 400_000,
    correspondences_per_pair: int = 20_000,
    min_planarity: float = 0.25,
    normal_neighbours: int = 20,
    max_normal_angle_deg: float = 45.0,
    stem_weight: float = 0.05,
    stem_radius: float = 0.15,
    stem_scale: float = 0.03,
    min_voxel_points: int = 1,
    robust_scale: float = 0.02,
    prior_translation: float = 0.03,
    prior_rotation_deg: float = 0.3,
    max_step_translation: float = 0.05,
    max_step_rotation_deg: float = 0.5,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> JointRefinement:
    """Solve all poses together from point-to-plane and stem correspondences.

    Levels run coarse to fine. At each, correspondences are associated
    ``rounds`` times under the current poses and each association is followed
    by ``iterations`` Gauss-Newton steps with Huber (1964) reweighting.

    Stems are paired afresh at every association, mutually nearest within
    ``stem_radius`` horizontally, with a fixed Huber scale ``stem_scale``, and
    carry ``stem_weight`` of each pair's weight whatever their count, so a few
    trunks can steer the horizontal directions that a hundred thousand ground
    points leave open.

    Two things keep the solve honest: a weak prior ties every pose to its
    pose-graph solution (``prior_translation`` m, ``prior_rotation_deg`` at
    one sigma), so a scan whose correspondences are all ground cannot slide;
    and a step is kept only if it lowers the robust cost, with the damping
    raised and the step retried otherwise, capped at ``max_step_translation``
    and ``max_step_rotation_deg``.

    Parameters
    ----------
    points
        Per scan, its planar ICP points in its own frame.
    poses
        Current ``world_from_scan``.
    edges
        Accepted pairs.
    stems
        Per scan, its stem positions in its own frame.
    reference
        Scan held fixed.
    log
        Called with progress messages.

    Returns
    -------
    JointRefinement
    """
    n = len(poses)
    if len(points) < n:
        raise IndexError("list index out of range")
    d = _core.coreg_refine_joint(
        [np.ascontiguousarray(np.asarray(p, dtype=np.float64).reshape(-1, 3)) for p in points[:n]],
        np.array([np.asarray(p, dtype=float) for p in poses], dtype=float).reshape(-1, 4, 4),
        [(int(i), int(j)) for i, j in edges],
        [
            np.ascontiguousarray(np.asarray(s, dtype=np.float64).reshape(-1, 3))
            if s is not None
            else np.zeros((0, 3))
            for s in stems
        ],
        int(reference),
        [float(v) for v in voxel_sizes],
        [float(v) for v in max_distances],
        int(rounds),
        int(iterations),
        int(points_per_scan),
        int(correspondences_per_pair),
        float(min_planarity),
        int(normal_neighbours),
        float(max_normal_angle_deg),
        float(stem_weight),
        float(stem_radius),
        float(stem_scale),
        int(min_voxel_points),
        float(robust_scale),
        float(prior_translation),
        float(prior_rotation_deg),
        float(max_step_translation),
        float(max_step_rotation_deg),
        int(seed),
        log,
    )
    return JointRefinement(
        list(d["poses"]),
        np.asarray(d["shifts"]),
        np.asarray(d["rotations"]),
        float(d["residual_before"]),
        float(d["residual_after"]),
        int(d["correspondences"]),
    )
