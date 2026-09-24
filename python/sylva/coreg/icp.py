# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Fine registration by iterative closest point, tuned for forests.

Two departures from a textbook ICP matter:

* **Point-to-plane with a planarity gate.** Foliage is a view-dependent mess
  whose normals are noise; the cost is kept to planar neighbourhoods (stems,
  ground, logs), the parts stable between viewpoints.
* **Robust, trimmed correspondences.** Forest scans overlap only partly, so
  many source points have no true match. A Huber weight with an adaptive
  scale and a trim that is phased in over the first iterations of each level
  keep them from biasing the pose without discarding the far points that
  carry the rotation.

The solver is a damped Gauss-Newton on SE(3) with a capped step, over a
coarse-to-fine voxel pyramid. The loop runs in the Rust core.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import _core

__all__ = ["ICPConfig", "ICPResult", "ICPTarget", "evaluate_registration", "icp"]


@dataclass
class ICPConfig:
    """Settings of :func:`icp`.

    The pyramid should span only the error left by the coarse stem match, a
    few centimetres to decimetres: a 0.4 m level with a 2 m search blurs the
    stand into a blob and converges confidently to the wrong pose.
    """

    voxel_sizes: tuple[float, ...] = (0.30, 0.15, 0.07, 0.05)
    max_distances: tuple[float, ...] | None = (0.80, 0.40, 0.20, 0.12)
    """Correspondence cut-off per level (m); None for 5 x voxel."""
    max_iterations: int = 30
    method: str = "point_to_plane"
    """``"point_to_plane"`` or ``"point_to_point"``."""
    robust: str = "huber"
    """``"huber"``, ``"tukey"`` or ``"none"``."""
    robust_scale: float = 0.05
    """Lower bound (m) of the adaptive robust scale."""
    trim_fraction: float = 0.85
    """Keep this fraction of the closest correspondences once trimming is in."""
    trim_ramp: int = 3
    """Iterations over which trimming is phased in, per level."""
    min_planarity: float = 0.25
    """Planarity a target point needs (point-to-plane); 0 disables the gate."""
    normal_neighbours: int = 20
    translation_tolerance: float = 1e-4
    rotation_tolerance: float = 2e-5
    fitness_threshold: float = 0.10
    """Distance (m) within which a source point counts as fitted."""
    damping: float = 1e-6
    max_points: int = 120_000
    """Cap per level after thinning."""
    plateau_tolerance: float = 0.0
    plateau_patience: int = 3
    seed: int = 0

    def distances(self) -> tuple[float, ...]:
        """Correspondence cut-off of each level."""
        if self.max_distances is not None:
            if len(self.max_distances) != len(self.voxel_sizes):
                raise ValueError("max_distances must have one entry per voxel size")
            return tuple(self.max_distances)
        return tuple(5.0 * v for v in self.voxel_sizes)


@dataclass
class ICPResult:
    """Outcome of :func:`icp`; ``transform`` maps source into target, initial guess included.

    Attributes
    ----------
    fitness
        Fraction of (voxel-thinned) source points with a target point within
        ``fitness_threshold``.
    inlier_rmse
        RMSE over those points (m).
    """

    transform: np.ndarray
    fitness: float
    inlier_rmse: float
    n_correspondences: int
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ICPResult(fitness={self.fitness:.3f}, rmse={self.inlier_rmse * 1000:.1f} mm, "
            f"n={self.n_correspondences}, iters={self.iterations}, converged={self.converged})"
        )


def _xyz(points) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(points, dtype=np.float64).reshape(-1, 3))


class ICPTarget:
    """A target's ICP pyramid, built once and reused.

    Every ICP rebuilds the target's voxel pyramid, normals and search trees,
    over half its time on a large scan. The pyramid depends only on the
    target and the pyramid settings, so registering several scans against a
    prepared target gives exactly the results of passing the points.

    Parameters
    ----------
    points
        ``(n, 3)`` target points.
    config
        ICP settings; the pyramid is built for its ``voxel_sizes``,
        ``max_points``, ``method``, ``normal_neighbours`` and ``seed``, and
        can only be used with settings that agree on those.
    """

    def __init__(self, points: np.ndarray, config: ICPConfig | None = None) -> None:
        cfg = config or ICPConfig()
        self._core = _core.CoregIcpTarget(
            _xyz(points),
            list(map(float, cfg.voxel_sizes)),
            int(cfg.max_points),
            cfg.method,
            int(cfg.normal_neighbours),
            int(cfg.seed),
        )

    def __len__(self) -> int:
        return len(self._core)


def icp(
    source: np.ndarray,
    target: np.ndarray | ICPTarget,
    initial: np.ndarray | None = None,
    config: ICPConfig | None = None,
) -> ICPResult:
    """Align ``source`` onto ``target``.

    Parameters
    ----------
    source
        ``(n, 3)`` points in their own frame.
    target
        ``(m, 3)`` points in their own frame, or an :class:`ICPTarget`
        prepared with the same pyramid settings.
    initial
        Initial guess of the source-to-target transform. Forest scans need a
        good one, from :func:`sylva.coreg.match_stem_maps`.
    config
        Settings; the defaults are tlsalign's.

    Returns
    -------
    ICPResult
    """
    cfg = config or ICPConfig()
    T = np.eye(4) if initial is None else np.asarray(initial, dtype=np.float64)
    prepared = target._core if isinstance(target, ICPTarget) else None
    d = _core.coreg_icp(
        _xyz(source),
        None if prepared is not None else _xyz(target),
        np.ascontiguousarray(T),
        list(map(float, cfg.voxel_sizes)),
        list(map(float, cfg.distances())),
        int(cfg.max_iterations),
        cfg.method,
        cfg.robust,
        float(cfg.robust_scale),
        float(cfg.trim_fraction),
        int(cfg.trim_ramp),
        float(cfg.min_planarity),
        int(cfg.normal_neighbours),
        float(cfg.translation_tolerance),
        float(cfg.rotation_tolerance),
        float(cfg.fitness_threshold),
        float(cfg.damping),
        int(cfg.max_points),
        float(cfg.plateau_tolerance),
        int(cfg.plateau_patience),
        int(cfg.seed),
        prepared=prepared,
    )
    return ICPResult(
        np.asarray(d["transform"]),
        float(d["fitness"]),
        float(d["inlier_rmse"]),
        int(d["n_correspondences"]),
        int(d["iterations"]),
        bool(d["converged"]),
        list(d["history"]),
    )


def evaluate_registration(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    threshold: float = 0.10,
    *,
    max_points: int = 200_000,
    voxel: float | None = 0.05,
    seed: int = 0,
) -> tuple[float, float, int]:
    """Score a registration without changing it.

    Fitness, the fraction of source points with a target point within
    ``threshold``, says more than RMSE in a forest: a low RMSE over a handful
    of points usually means the clouds slid into a spurious minimum.

    Parameters
    ----------
    source, target
        ``(n, 3)`` points.
    transform
        Source-to-target transform.
    threshold
        Inlier distance (m).
    max_points
        Random cap on each cloud after thinning.
    voxel
        Voxel centroids at this size first (m); None to skip.
    seed
        Seed of the random cap.

    Returns
    -------
    fitness : float
    inlier_rmse : float
    n_inliers : int
    """
    f, r, n = _core.coreg_evaluate(
        _xyz(source),
        _xyz(target),
        np.ascontiguousarray(np.asarray(transform, dtype=np.float64)),
        float(threshold),
        int(max_points),
        voxel,
        int(seed),
    )
    return float(f), float(r), int(n)
