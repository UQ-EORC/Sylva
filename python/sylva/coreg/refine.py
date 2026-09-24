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
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .geometry import KdTree, estimate_normals, voxel_downsample
from .transforms import (
    invert,
    se3_exp,
    se3_log,
    transform_difference,
    transform_points,
    transform_vectors,
)

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
    by ``iterations`` Gauss-Newton steps with Huber reweighting.

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
    say = log or (lambda _: None)
    n = len(poses)
    poses = [np.asarray(p, dtype=float).copy() for p in poses]
    start = [p.copy() for p in poses]
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i, j in edges if i != j and len(points[i]) and len(points[j])]
    stems = [
        np.asarray(s, float).reshape(-1, 3) if s is not None else np.zeros((0, 3)) for s in stems
    ]
    stems += [np.zeros((0, 3))] * (n - len(stems))
    total_corr = 0
    first_residual = last_residual = float("nan")

    for voxel, max_dist in zip(voxel_sizes, max_distances, strict=True):
        level_pts, level_nrm, level_ok = [], [], []
        for k in range(n):
            if len(points[k]):
                pts, counts = voxel_downsample(
                    np.asarray(points[k], np.float64), voxel, return_counts=True
                )
                if min_voxel_points > 1:
                    pts = pts[counts >= min_voxel_points]
            else:
                pts = np.zeros((0, 3))
            if len(pts) > points_per_scan:
                pts = pts[np.sort(rng.choice(len(pts), points_per_scan, replace=False))]
            if len(pts) >= normal_neighbours:
                nrm, planarity = estimate_normals(pts, k=normal_neighbours, radius=3.0 * voxel)
                ok = planarity >= min_planarity
            else:
                nrm, ok = np.zeros((len(pts), 3)), np.zeros(len(pts), dtype=bool)
            level_pts.append(pts)
            level_nrm.append(nrm)
            level_ok.append(ok)
        cos_limit = np.cos(np.radians(max_normal_angle_deg))

        for round_no in range(rounds):
            world = [transform_points(poses[k], level_pts[k]) for k in range(n)]
            wnrm = [transform_vectors(poses[k], level_nrm[k]) for k in range(n)]
            trees: dict[int, KdTree] = {}
            corr = []
            for i, j in edges:
                for a, b in ((i, j), (j, i)):
                    idx_b = np.flatnonzero(level_ok[b])
                    if len(idx_b) == 0 or level_ok[a].sum() == 0:
                        continue
                    if b not in trees:
                        trees[b] = KdTree(world[b][level_ok[b]])
                    src_idx = np.flatnonzero(level_ok[a])
                    if len(src_idx) > correspondences_per_pair:
                        src_idx = src_idx[
                            np.sort(
                                rng.choice(len(src_idx), correspondences_per_pair, replace=False)
                            )
                        ]
                    d, m = trees[b].query(world[a][src_idx], distance_upper_bound=max_dist)
                    hit = np.isfinite(d)
                    if not hit.any():
                        continue
                    src_idx, dst_idx = src_idx[hit], idx_b[m[hit]]
                    keep = (
                        np.abs(np.einsum("ij,ij->i", wnrm[a][src_idx], wnrm[b][dst_idx]))
                        >= cos_limit
                    )
                    if keep.sum() < 20:
                        continue
                    corr.append(
                        (
                            a,
                            b,
                            level_pts[a][src_idx[keep]],
                            level_pts[b][dst_idx[keep]],
                            level_nrm[b][dst_idx[keep]],
                        )
                    )
            stems_by_pair = _match_stems(poses, stems, edges, stem_radius)
            n_corr = sum(len(c[2]) for c in corr)
            total_corr = max(total_corr, n_corr)
            if n_corr == 0:
                say(f"  level {voxel:.2f} m: no correspondences")
                break

            damping = 1e-4
            prior_w = np.r_[
                np.full(3, 1.0 / np.radians(prior_rotation_deg) ** 2),
                np.full(3, 1.0 / prior_translation**2),
            ]
            for _ in range(iterations):
                H, g, cost, med, scales = _assemble(
                    poses, corr, stems_by_pair, n, robust_scale, stem_weight, stem_scale=stem_scale
                )
                for k in range(n):
                    if k == reference:
                        continue
                    xi_k = _twist_between(start[k], poses[k])
                    H[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] += np.diag(prior_w)
                    g[6 * k : 6 * k + 6] += prior_w * xi_k
                    cost += 0.5 * float(np.sum(prior_w * xi_k**2))
                if np.isnan(first_residual):
                    first_residual = med
                last_residual = med
                free = np.ones(6 * n, dtype=bool)
                free[6 * reference : 6 * reference + 6] = False
                for k in range(n):
                    if not len(level_pts[k]):
                        free[6 * k : 6 * k + 6] = False
                Hf, gf = H[np.ix_(free, free)], g[free]
                accepted = False
                scale = rot = tr = 0.0
                for _attempt in range(6):
                    try:
                        delta = np.linalg.solve(
                            Hf + damping * np.diag(np.maximum(Hf.diagonal(), 1e-9)), -gf
                        )
                    except np.linalg.LinAlgError:
                        damping *= 10.0
                        continue
                    full = np.zeros(6 * n)
                    full[free] = delta
                    rot = max(float(np.linalg.norm(full[6 * k : 6 * k + 3])) for k in range(n))
                    tr = max(float(np.linalg.norm(full[6 * k + 3 : 6 * k + 6])) for k in range(n))
                    scale = min(
                        1.0,
                        np.radians(max_step_rotation_deg) / max(rot, 1e-12),
                        max_step_translation / max(tr, 1e-12),
                    )
                    full *= scale
                    trial = [
                        se3_exp(full[6 * k : 6 * k + 6]) @ poses[k]
                        if np.any(full[6 * k : 6 * k + 6])
                        else poses[k]
                        for k in range(n)
                    ]
                    _, _, trial_cost, trial_med, _ = _assemble(
                        trial,
                        corr,
                        stems_by_pair,
                        n,
                        robust_scale,
                        stem_weight,
                        need_system=False,
                        scales=scales,
                        stem_scale=stem_scale,
                    )
                    for k in range(n):
                        if k != reference:
                            xi_k = _twist_between(start[k], trial[k])
                            trial_cost += 0.5 * float(np.sum(prior_w * xi_k**2))
                    if trial_cost <= cost:
                        poses, last_residual = trial, trial_med
                        damping = max(damping / 3.0, 1e-6)
                        accepted = True
                        break
                    damping *= 10.0
                if not accepted or scale * max(rot, tr) < 1e-5:
                    break
            say(
                f"  level {voxel:.2f} m, association {round_no + 1}/{rounds}: "
                f"{n_corr:,} correspondences, "
                f"median |residual| {last_residual * 100:.2f} cm"
            )

    shifts, rots = np.zeros(n), np.zeros(n)
    for k in range(n):
        rot, tr = transform_difference(start[k], poses[k])
        shifts[k], rots[k] = tr, np.degrees(rot)
    return JointRefinement(poses, shifts, rots, first_residual, last_residual, total_corr)


def _twist_between(T0: np.ndarray, T1: np.ndarray) -> np.ndarray:
    """Left-perturbation twist with ``T1 = exp(xi) T0``."""
    return se3_log(T1 @ invert(T0))


def _assemble(
    poses,
    corr,
    stems_by_pair,
    n,
    robust_scale,
    stem_weight,
    need_system=True,
    scales=None,
    stem_scale=0.03,
):
    """Robust cost, median residual and (optionally) the normal equations.

    ``scales`` fixes each block's Huber scale; when None they are estimated
    and returned, so a trial step is costed at the scales of the step it is
    compared with.
    """
    fixed = scales is not None
    scales = scales if fixed else {}
    H = np.zeros((6 * n, 6 * n)) if need_system else None
    g = np.zeros(6 * n) if need_system else None
    cost = 0.0
    res_all = []
    pair_weight: dict[tuple[int, int], float] = {}
    for a, b, p, q, nq in corr:
        xa, xb = transform_points(poses[a], p), transform_points(poses[b], q)
        nw = transform_vectors(poses[b], nq)
        dvec = xa - xb
        r = np.einsum("ij,ij->i", nw, dvec)
        key_s = ("p", a, b, len(r))
        s = scales[key_s] if fixed else max(robust_scale, float(np.median(np.abs(r))), 1e-9)
        scales.setdefault(key_s, s)
        absr = np.abs(r)
        w = np.where(absr <= s, 1.0, s / np.maximum(absr, 1e-12))
        cost += float(np.sum(np.where(absr <= s, 0.5 * r * r, s * (absr - 0.5 * s))))
        res_all.append(absr)
        key = (a, b) if (a, b) in stems_by_pair else (b, a)
        pair_weight[key] = pair_weight.get(key, 0.0) + float(w.sum())
        if need_system:
            J = np.empty((len(r), 12))
            J[:, 0:3] = np.cross(xa, nw)
            J[:, 3:6] = nw
            J[:, 6:9] = np.cross(nw, xb) - np.cross(dvec, nw)
            J[:, 9:12] = -nw
            _accumulate(H, g, J, r, w, a, b)
    for (i, j), (p, q) in stems_by_pair.items():
        if (i, j) not in pair_weight:
            continue
        xi, xj = transform_points(poses[i], p), transform_points(poses[j], q)
        r2 = (xi - xj)[:, :2].reshape(-1)
        s = stem_scale
        absr = np.abs(r2)
        w = np.where(absr <= s, 1.0, s / np.maximum(absr, 1e-12))
        weight = stem_weight * pair_weight[(i, j)] / max((1.0 - stem_weight) * w.sum(), 1e-9)
        w = w * weight
        cost += float(np.sum(weight * np.where(absr <= s, 0.5 * r2 * r2, s * (absr - 0.5 * s))))
        if need_system:
            J = np.zeros((len(r2), 12))
            Sxi, Sxj = -_skew_rows(xi), _skew_rows(xj)
            J[0::2, 0:3], J[1::2, 0:3] = Sxi[:, 0, :], Sxi[:, 1, :]
            J[0::2, 3], J[1::2, 4] = 1.0, 1.0
            J[0::2, 6:9], J[1::2, 6:9] = Sxj[:, 0, :], Sxj[:, 1, :]
            J[0::2, 9], J[1::2, 10] = -1.0, -1.0
            _accumulate(H, g, J, r2, w, i, j)
    med = float(np.median(np.concatenate(res_all))) if res_all else float("nan")
    return H, g, cost, med, scales


def _match_stems(
    poses, stems, edges, radius
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Mutually nearest stems within ``radius`` horizontally, per edge, under ``poses``."""
    out = {}
    world = [
        transform_points(poses[k], s)[:, :2] if len(s) else np.zeros((0, 2))
        for k, s in enumerate(stems)
    ]
    for i, j in edges:
        if len(world[i]) == 0 or len(world[j]) == 0:
            continue
        dij, mij = KdTree(world[j]).query(world[i], distance_upper_bound=radius)
        _, mji = KdTree(world[i]).query(world[j], distance_upper_bound=radius)
        a = np.flatnonzero(np.isfinite(dij))
        a = a[mji[mij[a]] == a]
        if len(a):
            out[(i, j)] = (stems[i][a], stems[j][mij[a]])
    return out


def _skew_rows(x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), 3, 3))
    out[:, 0, 1], out[:, 0, 2] = -x[:, 2], x[:, 1]
    out[:, 1, 0], out[:, 1, 2] = x[:, 2], -x[:, 0]
    out[:, 2, 0], out[:, 2, 1] = -x[:, 1], x[:, 0]
    return out


def _accumulate(H, g, J, r, w, a, b) -> None:
    Jw = J * w[:, None]
    Hp, gp = J.T @ Jw, Jw.T @ r
    sa, sb = slice(6 * a, 6 * a + 6), slice(6 * b, 6 * b + 6)
    H[sa, sa] += Hp[0:6, 0:6]
    H[sa, sb] += Hp[0:6, 6:12]
    H[sb, sa] += Hp[6:12, 0:6]
    H[sb, sb] += Hp[6:12, 6:12]
    g[sa] += gp[0:6]
    g[sb] += gp[6:12]
