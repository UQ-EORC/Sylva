"""Marker-free coregistration of scan positions in forests.

Registers scans with no targets and no initial alignment, from the trees
themselves:

1. **Per scan** (:func:`prepare_scan`): a ground model gives height above
   ground; stems are detected and reduced to a stem map (position at breast
   height, diameter, quality); locally planar points (stems, ground, logs) are
   kept for ICP and foliage is dropped.
2. **Per pair** (:func:`register_pair`): stem distances do not depend on the
   unknown transform, so stem pairs at equal separation are candidate matches
   and a sorted pair table makes the search near-exhaustive
   (:func:`match_stem_maps`, yaw and translation). Point-to-plane ICP refines
   the pose, and the pair is accepted only if it fits over all points *and*
   over the points above ground, so the ground plane alone cannot confirm a
   wrong pair.
3. **Whole survey** (:func:`register_scans`): accepted pairs are edges of a
   pose graph solved with a Huber kernel and outlier rejection that never
   disconnects the graph (:class:`PoseGraph`). Scans left over are placed
   against the combined registered survey (:func:`place_scan`). Scans whose
   poses are already trusted can be held fixed, which registers new or badly
   registered positions into an existing project.

Scans are expected roughly levelled (z up), as they are after applying the
scanner's inclination or a RiSCAN SOP; ICP recovers residual tilt. A ported
and adapted version of the author's tlsalign.

Examples
--------
>>> scans = [coreg.prepare_scan(cloud, name) for name, cloud in levelled.items()]
>>> result = coreg.register_scans(scans, positions=approximate_origins)
>>> result.poses[3]            # world_from_scan of scan 3
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import _core, filters, ground, progress, trees
from .pointcloud import PointCloud

__all__ = [
    "StemMap",
    "StemMatch",
    "ScanFeatures",
    "PairResult",
    "CoregResult",
    "PoseGraph",
    "stem_map",
    "match_stem_maps",
    "prepare_scan",
    "icp_refine",
    "evaluate_registration",
    "register_pair",
    "place_scan",
    "place_from_prior",
    "register_scans",
    "se3_exp",
    "se3_log",
    "invert",
]


# --------------------------------------------------------------------------- #
# Rigid-transform helpers
# --------------------------------------------------------------------------- #


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform.

    Parameters
    ----------
    T
        ``(4, 4)`` rigid transform.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` inverse, computed from the rotation's transpose.
    """
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    K = _skew(w)
    if theta < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta**2 * (K @ K)


def _so3_log(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(c))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * w
    if np.pi - theta < 1e-6:
        # Near pi the antisymmetric part vanishes: take the axis from R + I.
        A = (R + np.eye(3)) * 0.5
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        axis /= max(np.linalg.norm(axis), 1e-12)
        if w @ axis < 0:
            axis = -axis
        return axis * theta
    return w * (theta / (2.0 * np.sin(theta)))


def _left_jacobian(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    K = _skew(w)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * K + (K @ K) / 6.0
    t2 = theta * theta
    return (
        np.eye(3) + (1 - np.cos(theta)) / t2 * K + (theta - np.sin(theta)) / (t2 * theta) * (K @ K)
    )


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Transform from a twist ``[rotation vector, translation]``.

    Parameters
    ----------
    xi
        Length-6 twist.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` rigid transform.
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = _so3_exp(xi[:3])
    T[:3, 3] = _left_jacobian(xi[:3]) @ xi[3:]
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Twist ``[rotation vector, translation]`` of a rigid transform.

    Parameters
    ----------
    T
        ``(4, 4)`` rigid transform.

    Returns
    -------
    numpy.ndarray
        Length-6 twist; the inverse of :func:`se3_exp`.
    """
    T = np.asarray(T, dtype=float)
    w = _so3_log(T[:3, :3])
    return np.concatenate([w, np.linalg.solve(_left_jacobian(w), T[:3, 3])])


def _transform(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return xyz @ T[:3, :3].T + T[:3, 3]


def _shift(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """Translation (m) and yaw (deg) separating two transforms."""
    d = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    yaw = np.degrees(abs(np.arctan2(A[1, 0], A[0, 0]) - np.arctan2(B[1, 0], B[0, 0])))
    return d, float(min(yaw, 360 - yaw))


# --------------------------------------------------------------------------- #
# Stem maps
# --------------------------------------------------------------------------- #


@dataclass
class StemMap:
    """The stems of one scan: the features a coarse alignment is found from.

    Attributes
    ----------
    positions
        ``(n, 3)`` stem axis at ``reference_height`` above the scan's ground,
        as absolute coordinates in the scan's frame.
    diameters
        Stem diameters (m).
    qualities
        Detection quality (0-1); the best stems take part in matching first.
    """

    positions: np.ndarray
    diameters: np.ndarray
    qualities: np.ndarray

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(
            np.asarray(self.positions, dtype=float).reshape(-1, 3)
        )
        self.diameters = np.ascontiguousarray(np.asarray(self.diameters, dtype=float).reshape(-1))
        self.qualities = np.ascontiguousarray(np.asarray(self.qualities, dtype=float).reshape(-1))

    def __len__(self) -> int:
        return len(self.positions)

    def transformed(self, T: np.ndarray) -> StemMap:
        """The same stems in another frame.

        Parameters
        ----------
        T
            ``(4, 4)`` transform applied to the positions.

        Returns
        -------
        StemMap
        """
        return StemMap(
            _transform(np.asarray(T, float), self.positions), self.diameters, self.qualities
        )

    @classmethod
    def concatenate(cls, maps: Sequence[StemMap], merge_distance: float = 0.3) -> StemMap:
        """Combine stem maps, keeping one entry per tree.

        The same trunk is seen from many scans; duplicates would wreck the
        matcher, which assumes one-to-one correspondences.

        Parameters
        ----------
        maps
            Stem maps in a common frame.
        merge_distance
            Stems closer than this horizontally (m) are one tree; the best
            quality one is kept.

        Returns
        -------
        StemMap
        """
        maps = [m for m in maps if len(m)]
        if not maps:
            return cls(np.zeros((0, 3)), np.zeros(0), np.zeros(0))
        pos = np.vstack([m.positions for m in maps])
        dia = np.concatenate([m.diameters for m in maps])
        qual = np.concatenate([m.qualities for m in maps])
        order = np.argsort(-qual)
        kept: list[int] = []
        for i in order:
            if kept:
                d = np.hypot(*(pos[kept, :2] - pos[i, :2]).T)
                if d.min() < merge_distance:
                    continue
            kept.append(int(i))
        return cls(pos[kept], dia[kept], qual[kept])


def stem_map(
    cloud: PointCloud,
    dtm=None,
    height_attr: str = "height",
    reference_height: float = 1.3,
    **detect,
) -> StemMap:
    """Stem map of one levelled scan.

    Parameters
    ----------
    cloud
        One scan (or any cloud) with z up.
    dtm
        Terrain of the cloud; classified and built from the cloud (CSF on a
        5 cm copy) if None.
    height_attr
        Attribute holding height above ground; computed from ``dtm`` if absent.
    reference_height
        Height above ground at which stem positions are taken (m).
    **detect
        Passed to :func:`sylva.trees.detect_stems`.

    Returns
    -------
    StemMap
        Positions at ``reference_height`` above the terrain, as absolute z.
    """
    if dtm is None:
        g = ground.classify_ground_csf(filters.voxel_downsample(cloud, 0.05))
        dtm = ground.make_dtm(g, resolution=0.5)
    if height_attr not in cloud.attrs:
        cloud = ground.normalize_height(cloud, dtm, attr=height_attr)
    found = trees.detect_stems(cloud, height_attr=height_attr, **detect)
    if not found:
        return StemMap(np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    xy = np.array([[t.x, t.y] for t in found])
    z = dtm.sample(xy[:, 0], xy[:, 1]) + reference_height
    return StemMap(np.column_stack([xy, z]), [t.dbh for t in found], [t.quality for t in found])


@dataclass
class StemMatch:
    """Outcome of :func:`match_stem_maps`; ``transform`` maps source into target.

    Attributes
    ----------
    success
        At least ``min_inliers`` stems matched consistently. Check it first.
    correspondences
        ``(n, 2)`` indices ``(source, target)`` of matched stems.
    ambiguity
        Inliers of the best distinctly different alignment over the best's:
        near 1 means the stem pattern matches itself elsewhere (a planted
        lattice) and the winner is a coin toss.
    rival
        That alternative alignment, refined, for ICP to decide between.
    """

    transform: np.ndarray
    n_inliers: int
    inlier_rmse: float
    correspondences: np.ndarray
    success: bool
    ambiguity: float
    rival: StemMatch | None = None

    @classmethod
    def _from_core(cls, d: dict) -> StemMatch:
        rival = cls._from_core(d["rival"]) if d["rival"] is not None else None
        return cls(
            np.asarray(d["transform"]),
            int(d["n_inliers"]),
            float(d["inlier_rmse"]),
            np.asarray(d["correspondences"]),
            bool(d["success"]),
            float(d["ambiguity"]),
            rival,
        )


def match_stem_maps(source: StemMap, target: StemMap, **params) -> StemMatch:
    """Align two stem maps with no initial guess (yaw and translation).

    Pairs of source stems are matched to target pairs at the same separation
    (a binary search in the target's sorted pair table); each hypothesis is
    scored by the one-to-one stem correspondences it brings, and the best is
    refined by alternating fit and correspondence search.

    Parameters
    ----------
    source, target
        Stem maps; ``transform`` maps source into target.
    **params
        ``min_pair_distance`` [2], ``max_pair_distance`` [35],
        ``pair_distance_tolerance`` [0.25], ``inlier_tolerance`` [0.40] (m),
        ``diameter_rel_tolerance`` [0.30], ``diameter_abs_tolerance`` [0.04],
        ``use_diameters`` [True], ``max_stems`` [70], ``max_hypotheses``
        [60000], ``min_inliers`` [4], ``early_exit_inliers`` [40],
        ``distinct_translation`` [1.0 m], ``distinct_yaw_deg`` [5],
        ``refine_iterations`` [6].

    Returns
    -------
    StemMatch
    """
    d = _core.match_stem_maps(
        source.positions,
        source.diameters,
        source.qualities,
        target.positions,
        target.diameters,
        target.qualities,
        **params,
    )
    return StemMatch._from_core(d)


# --------------------------------------------------------------------------- #
# Scan features and ICP
# --------------------------------------------------------------------------- #


@dataclass
class ScanFeatures:
    """What registration needs from one scan; build with :func:`prepare_scan`.

    Attributes
    ----------
    name
        Scan name, for reports.
    stems
        The scan's stem map.
    points
        ``(n, 3)`` locally planar points (stems, ground, logs) for ICP.
    heights
        Height above ground of each of ``points``.
    origin
        Scanner position in the scan's frame (where it stood); used to find a
        scan's neighbours. The centroid of ``points`` if unknown.
    """

    name: str
    stems: StemMap
    points: np.ndarray
    heights: np.ndarray
    origin: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.origin is None:
            self.origin = self.points.mean(axis=0) if len(self.points) else np.zeros(3)
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)

    def location(self, pose: np.ndarray) -> np.ndarray:
        """Where the scanner stood in the world, given the scan's ``world_from_scan``.

        Parameters
        ----------
        pose
            ``(4, 4)`` ``world_from_scan``.

        Returns
        -------
        numpy.ndarray
            ``(3,)`` position.
        """
        return _transform(np.asarray(pose, float), self.origin[None])[0]

    @property
    def usable(self) -> bool:
        """Enough stems and points to register."""
        return len(self.stems) >= 3 and len(self.points) >= 1000


def prepare_scan(
    cloud: PointCloud,
    name: str = "",
    voxel: float = 0.05,
    min_planarity: float = 0.35,
    max_height: float = 12.0,
    k: int = 20,
    origin=None,
    **detect,
) -> ScanFeatures:
    """Reduce one levelled scan to its stem map and ICP points.

    Parameters
    ----------
    cloud
        One scan with z up, in its own (or an approximate) frame. Crop it to
        the range that matters (30-40 m) first; far returns are sparse and slow.
    name
        Name used in reports.
    voxel
        Spacing (m) of the ICP points.
    min_planarity
        Local planarity a point needs to be kept for ICP. Crown returns have
        no stable geometry between viewpoints (occlusion, wind) and trap ICP;
        stems and ground are planar.
    max_height
        Points higher above ground than this (m) are left out of ICP.
    k
        Neighbours for planarity.
    origin
        Scanner position in the cloud's frame; ``(0, 0, 0)`` for a scan in its
        own scanner frame, the SOP translation after applying the SOP.
    **detect
        Passed to :func:`sylva.trees.detect_stems`.

    Returns
    -------
    ScanFeatures
    """
    g = ground.classify_ground_csf(filters.voxel_downsample(cloud, 0.05))
    dtm = ground.make_dtm(g, resolution=0.5)
    cloud = ground.normalize_height(cloud, dtm)
    stems = stem_map(cloud, dtm, **detect)
    thin = filters.voxel_downsample(cloud, voxel)
    h = thin.attrs["height"]
    planarity, _ = filters.planarity_linearity(thin, k=k)
    keep = (planarity >= min_planarity) & (h <= max_height)
    return ScanFeatures(name, stems, thin.xyz[keep], h[keep], origin)


def evaluate_registration(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    threshold: float = 0.10,
    max_points: int = 200_000,
) -> tuple[float, float]:
    """Score an alignment without changing it.

    Parameters
    ----------
    source, target
        ``(n, 3)`` points.
    transform
        ``(4, 4)`` transform applied to ``source``.
    threshold
        Distance (m) within which a source point counts as matched.
    max_points
        Source points used at most (a fixed random subset beyond).

    Returns
    -------
    fitness, rmse : float
        Share of source points with a target point within ``threshold``, and
        the RMS distance of those. Fitness is the more telling of the two: a
        low RMSE over a handful of points usually means a spurious minimum.
    """
    src = np.asarray(source, float)
    if len(src) == 0 or len(target) == 0:
        return 0.0, float("inf")
    if len(src) > max_points:
        src = src[np.random.default_rng(0).choice(len(src), max_points, replace=False)]
    d, _ = filters.knn(np.asarray(target, float), _transform(transform, src), 1)
    hit = d[:, 0] <= threshold
    rmse = float(np.sqrt(np.mean(d[hit, 0] ** 2))) if hit.any() else float("inf")
    return float(hit.mean()), rmse


def _thin(xyz: np.ndarray, voxel: float, max_points: int) -> np.ndarray:
    keep = _core.voxel_downsample_indices(np.ascontiguousarray(xyz, dtype=float), voxel)
    out = xyz[keep]
    if len(out) > max_points:
        out = out[np.random.default_rng(0).choice(len(out), max_points, replace=False)]
    return np.ascontiguousarray(out)


def icp_refine(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    levels: Sequence[tuple[float, float]] = (
        (0.30, 0.80),
        (0.15, 0.40),
        (0.07, 0.20),
        (0.05, 0.12),
    ),
    trim: float = 0.85,
    max_iterations: int = 30,
    max_points: int = 120_000,
) -> tuple[np.ndarray, dict]:
    """Coarse-to-fine point-to-plane ICP.

    Parameters
    ----------
    source, target
        ``(n, 3)`` points (planar points from :func:`prepare_scan`).
    initial
        ``(4, 4)`` starting transform of ``source`` into ``target``.
    levels
        ``(voxel, max correspondence distance)`` per level (m); each level
        starts from the last one's result.
    trim
        Share of the closest correspondences kept each iteration.
    max_iterations
        Per level.
    max_points
        Points per cloud and level at most.

    Returns
    -------
    transform : numpy.ndarray
        ``(4, 4)`` refined transform.
    info : dict
        ``fitness`` and ``rmse`` at 10 cm, and ``n_correspondences``.
    """
    T = np.asarray(initial, dtype=float)
    info = {"rmse": float("inf"), "n_correspondences": 0}
    for voxel, dist in levels:
        s, t = _thin(source, voxel, max_points), _thin(target, voxel, max_points)
        if len(s) < 50 or len(t) < 50:
            continue
        T, info = _core.icp(
            s, t, np.ascontiguousarray(T), dist, max_iterations, 1e-6, "plane", trim, 12
        )
    fitness, rmse = evaluate_registration(source, target, T)
    return T, {
        "fitness": fitness,
        "rmse": rmse,
        "n_correspondences": int(info["n_correspondences"]),
    }


# --------------------------------------------------------------------------- #
# Pairs
# --------------------------------------------------------------------------- #


@dataclass
class PairResult:
    """Registration of scan ``i`` onto scan ``j`` (``transform`` is ``j_from_i``).

    Attributes
    ----------
    success
        Accepted by every test.
    reason
        Why it was refused, or how it was accepted.
    fitness, fitness_above, rmse
        ICP fitness over all points and over points above ``fitness_min_height``,
        and inlier RMSE (m).
    stem_residual
        Median distance of the matched stems after alignment (m).
    """

    i: int
    j: int
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    success: bool = False
    reason: str = ""
    fitness: float = 0.0
    fitness_above: float = float("nan")
    rmse: float = float("inf")
    n_correspondences: int = 0
    n_stems: int = 0
    stem_residual: float = float("nan")
    seconds: float = 0.0


@dataclass
class _Accept:
    min_match_inliers: int = 5
    max_match_rmse: float = 0.30
    max_match_ambiguity: float = 0.8
    ambiguity_margin: float = 1.25
    min_fitness: float = 0.10
    min_fitness_above: float = 0.06
    fitness_min_height: float = 1.0
    max_rmse: float = 0.15
    max_shift: float = 2.0
    stem_agreement_tolerance: float = 0.25


def _fitness_above(src: ScanFeatures, target: np.ndarray, T: np.ndarray, acc: _Accept) -> float:
    m = src.heights > acc.fitness_min_height
    if m.sum() < 100:
        return float("nan")
    return evaluate_registration(src.points[m], target, T)[0]


def _stem_residual(T: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(_transform(T, a) - b, axis=1)))


def _judge(
    result: PairResult,
    source: ScanFeatures,
    target_points: np.ndarray,
    coarse: np.ndarray,
    match: StemMatch | None,
    target_stems: StemMap | None,
    acc: _Accept,
    icp_kw: dict,
) -> None:
    """Refine a coarse transform with ICP and decide whether to accept it."""
    T, info = icp_refine(source.points, target_points, coarse, **icp_kw)
    a = b = np.zeros((0, 3))
    if match is not None and target_stems is not None and len(match.correspondences):
        a = source.stems.positions[match.correspondences[:, 0]]
        b = target_stems.positions[match.correspondences[:, 1]]
    # Cross-check ICP against the stems it was meant to refine.
    if (
        len(a) >= 3
        and _stem_residual(T, a, b) > _stem_residual(coarse, a, b) + acc.stem_agreement_tolerance
    ):
        T = coarse
        info["fitness"], info["rmse"] = evaluate_registration(source.points, target_points, T)
        result.reason = "kept the stem alignment over ICP; "
    above = _fitness_above(source, target_points, T, acc)

    if match is not None and match.rival is not None and match.ambiguity > acc.max_match_ambiguity:
        # An ambiguous stem pattern: refine the rival too and keep whichever
        # fits the points above ground clearly better.
        T2, info2 = icp_refine(source.points, target_points, match.rival.transform, **icp_kw)
        above2 = _fitness_above(source, target_points, T2, acc)
        d, yaw = _shift(T, T2)
        if d > 1.0 or yaw > 5.0:
            mine, theirs = np.nan_to_num(above), np.nan_to_num(above2)
            if theirs > mine:
                T, info, above, mine, theirs = T2, info2, above2, theirs, mine
                coarse = match.rival.transform
            if mine < acc.ambiguity_margin * theirs:
                result.reason = f"ambiguous stem pattern ({mine:.3f} vs {theirs:.3f} above ground)"
                return
            result.reason += f"rival settled by ICP ({mine:.3f} vs {theirs:.3f}); "

    result.transform, result.fitness, result.rmse = T, info["fitness"], info["rmse"]
    result.fitness_above, result.n_correspondences = above, info["n_correspondences"]
    result.stem_residual = _stem_residual(T, a, b)
    shift = float(np.linalg.norm(coarse[:3, 3] - T[:3, 3]))
    if shift > acc.max_shift:
        result.reason = f"ICP moved {shift:.2f} m from the stem alignment"
    elif info["fitness"] < acc.min_fitness:
        result.reason = f"low fitness {info['fitness']:.3f}"
    elif np.isfinite(above) and above < acc.min_fitness_above:
        result.reason = f"low fitness above ground {above:.3f}: the ground alone matched"
    elif info["rmse"] > acc.max_rmse:
        result.reason = f"high rmse {info['rmse']:.3f} m"
    else:
        result.success = True
        result.reason += f"moved {shift * 100:.1f} cm by ICP"


def _coarse_ok(result: PairResult, m: StemMatch, acc: _Accept) -> bool:
    result.n_stems = m.n_inliers
    if not m.success or m.n_inliers < acc.min_match_inliers:
        result.reason = f"stem matching failed ({m.n_inliers} stems, need {acc.min_match_inliers})"
    elif m.inlier_rmse > acc.max_match_rmse:
        result.reason = f"stem match too loose ({m.inlier_rmse * 100:.0f} cm)"
    elif m.ambiguity > acc.max_match_ambiguity and (m.rival is None or not m.rival.success):
        result.reason = f"ambiguous stem pattern (rival has {m.ambiguity:.0%} of the stems)"
    else:
        return True
    return False


def register_pair(
    source: ScanFeatures,
    target: ScanFeatures,
    initial: np.ndarray | None = None,
    i: int = 0,
    j: int = 1,
    match: dict | None = None,
    icp: dict | None = None,
    **accept,
) -> PairResult:
    """Register one scan onto another.

    Parameters
    ----------
    source, target
        Prepared scans.
    initial
        Approximate transform of ``source`` into ``target``; skips stem
        matching when given.
    i, j
        Indices recorded in the result.
    match
        Keyword arguments for :func:`match_stem_maps`.
    icp
        Keyword arguments for :func:`icp_refine`.
    **accept
        Acceptance limits: ``min_match_inliers`` [5], ``max_match_rmse``
        [0.30 m], ``max_match_ambiguity`` [0.8], ``ambiguity_margin`` [1.25],
        ``min_fitness`` [0.10], ``min_fitness_above`` [0.06],
        ``fitness_min_height`` [1.0 m], ``max_rmse`` [0.15 m], ``max_shift``
        [2.0 m], ``stem_agreement_tolerance`` [0.25 m].

    Returns
    -------
    PairResult
        ``transform`` maps ``source`` into ``target``; check ``success``.
    """
    start = time.perf_counter()
    acc = _Accept(**accept)
    result = PairResult(i, j)
    if initial is not None:
        _judge(
            result, source, target.points, np.asarray(initial, float), None, None, acc, icp or {}
        )
    else:
        m = match_stem_maps(source.stems, target.stems, **(match or {}))
        if _coarse_ok(result, m, acc):
            _judge(result, source, target.points, m.transform, m, target.stems, acc, icp or {})
    result.seconds = time.perf_counter() - start
    return result


def place_scan(
    scan: ScanFeatures,
    survey: Sequence[ScanFeatures],
    poses: Sequence[np.ndarray],
    neighbours: int = 6,
    match: dict | None = None,
    icp: dict | None = None,
    **accept,
) -> tuple[PairResult, list[int]]:
    """Place one scan against an already registered survey.

    The scan's stems are matched against the combined stems of every
    registered scan at once, then refined by ICP against the merged points of
    the nearest ones. This is how scans that failed pairwise, or new scans, join
    an existing registration.

    Parameters
    ----------
    scan
        The scan to place.
    survey
        Registered scans.
    poses
        ``world_from_scan`` of each of ``survey``.
    neighbours
        Registered scans (nearest to the coarse position) ICP runs against.
    match, icp, **accept
        As for :func:`register_pair`.

    Returns
    -------
    result : PairResult
        ``transform`` is ``world_from_scan``; ``j`` is -1.
    used : list of int
        Indices into ``survey`` of the scans ICP ran against.
    """
    start = time.perf_counter()
    acc = _Accept(**accept)
    result = PairResult(-1, -1)
    moved = [s.stems.transformed(T) for s, T in zip(survey, poses, strict=True)]
    combined = StemMap.concatenate(moved)
    m = match_stem_maps(scan.stems, combined, **(match or {}))
    used: list[int] = []
    if _coarse_ok(result, m, acc):
        here = scan.location(m.transform)
        order = sorted(
            range(len(survey)),
            key=lambda n: float(np.linalg.norm(survey[n].location(poses[n]) - here)),
        )
        used = [n for n in order if len(survey[n].points)][:neighbours]
        target = np.vstack([_transform(poses[n], survey[n].points) for n in used])
        _judge(result, scan, target, m.transform, m, combined, acc, icp or {})
    result.seconds = time.perf_counter() - start
    return result, used


# --------------------------------------------------------------------------- #
# Pose graph
# --------------------------------------------------------------------------- #


def _information(rmse: float, fitness: float, n: int, extent: float = 15.0) -> np.ndarray:
    sigma_t = max(rmse, 1e-4) / np.sqrt(max(n, 1))
    sigma_r = sigma_t / extent
    scale = min(max(fitness, 1e-3), 1.0)
    return np.diag([scale / sigma_r**2] * 3 + [scale / sigma_t**2] * 3)


@dataclass(eq=False)
class _Edge:
    i: int
    j: int
    transform: np.ndarray
    information: np.ndarray
    weight: float


class PoseGraph:
    """Scan poses reconciled from all pairwise measurements at once.

    Going A->B->C->A pairwise does not return to the start; the graph spreads
    that closure error over the loop. Nodes are ``world_from_scan`` poses,
    edges measured ``j_from_i`` transforms weighted by their quality. Solved
    by Levenberg-Marquardt on SE(3) with a Huber kernel, then edges far
    worse than the rest are dropped (never disconnecting the graph) and the
    graph re-solved, because a pairwise match can be confidently wrong.

    Parameters
    ----------
    n
        Number of scans.
    fixed
        ``{index: world_from_scan}`` of scans held fixed; scan 0 at the
        identity if None.
    """

    def __init__(self, n: int, fixed: dict[int, np.ndarray] | None = None) -> None:
        self.n = n
        self.fixed = {int(k): np.asarray(v, float) for k, v in (fixed or {0: np.eye(4)}).items()}
        self.poses = [self.fixed.get(k, np.eye(4)).copy() for k in range(n)]
        self.edges: list[_Edge] = []

    def add_edge(
        self,
        i: int,
        j: int,
        transform: np.ndarray,
        rmse: float = 0.01,
        fitness: float = 1.0,
        n_correspondences: int = 1,
    ) -> None:
        """Add a measurement that scan ``i`` maps into scan ``j`` by ``transform``.

        Parameters
        ----------
        i, j
            Scan indices.
        transform
            ``(4, 4)`` ``j_from_i``.
        rmse, fitness, n_correspondences
            Registration quality; together they set the edge's weight.
        """
        info = _information(rmse, fitness, n_correspondences)
        self.edges.append(
            _Edge(i, j, np.asarray(transform, float), info, fitness * max(n_correspondences, 1))
        )

    def _residual(self, e: _Edge, poses: list[np.ndarray]) -> np.ndarray:
        return se3_log(invert(e.transform) @ invert(poses[e.j]) @ poses[e.i])

    def connected(self, edges: list[_Edge] | None = None) -> set[int]:
        """Scans connected to a fixed scan.

        Parameters
        ----------
        edges
            Edges to use; all if None.

        Returns
        -------
        set of int
        """
        adj: dict[int, list[int]] = {k: [] for k in range(self.n)}
        for e in self.edges if edges is None else edges:
            adj[e.i].append(e.j)
            adj[e.j].append(e.i)
        seen = set(self.fixed)
        stack = list(self.fixed)
        while stack:
            for nb in adj[stack.pop()]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen

    def initialise(self) -> None:
        """Chain the strongest edges outward from the fixed scans (a spanning tree)."""
        done = set(self.fixed)
        while True:
            frontier = [e for e in self.edges if (e.i in done) != (e.j in done)]
            if not frontier:
                break
            e = max(frontier, key=lambda e: e.weight)
            if e.i in done:
                self.poses[e.j] = self.poses[e.i] @ invert(e.transform)
                done.add(e.j)
            else:
                self.poses[e.i] = self.poses[e.j] @ e.transform
                done.add(e.i)

    def _cost(self, edges: list[_Edge], poses: list[np.ndarray], delta: float) -> float:
        total = 0.0
        for e in edges:
            r = self._residual(e, poses)
            chi2 = float(r @ e.information @ r)
            total += chi2 if chi2 <= delta**2 else 2 * delta * np.sqrt(chi2) - delta**2
        return total

    def _solve(self, edges: list[_Edge], iterations: int, delta: float, tol: float) -> int:
        free = [k for k in range(self.n) if k not in self.fixed and k in self.connected(edges)]
        slot = {k: s for s, k in enumerate(free)}
        dim = 6 * len(free)
        if dim == 0:
            return 0
        lam, cost = 1e-4, self._cost(edges, self.poses, delta)
        for it in range(iterations):
            H, b = np.zeros((dim, dim)), np.zeros(dim)
            for e in edges:
                r = self._residual(e, self.poses)
                chi2 = float(r @ e.information @ r)
                w = 1.0 if chi2 <= delta**2 else delta / np.sqrt(chi2)
                om = e.information * w
                blocks = []
                for node in (e.i, e.j):
                    if node in slot:
                        J = np.zeros((6, 6))
                        for k in range(6):
                            step = np.zeros(6)
                            step[k] = 1e-6
                            p = list(self.poses)
                            p[node] = self.poses[node] @ se3_exp(step)
                            J[:, k] = (self._residual(e, p) - r) / 1e-6
                        blocks.append((slot[node], J))
                for sa, Ja in blocks:
                    b[6 * sa : 6 * sa + 6] -= Ja.T @ om @ r
                    for sb, Jb in blocks:
                        H[6 * sa : 6 * sa + 6, 6 * sb : 6 * sb + 6] += Ja.T @ om @ Jb
            diag = np.maximum(np.diag(H), 1e-12)
            for _ in range(12):
                try:
                    step = np.linalg.solve(H + np.diag(lam * diag), b)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                trial = list(self.poses)
                for k, s in slot.items():
                    trial[k] = self.poses[k] @ se3_exp(step[6 * s : 6 * s + 6])
                new = self._cost(edges, trial, delta)
                if new <= cost:
                    improvement, cost, self.poses = cost - new, new, trial
                    lam = max(lam * 0.5, 1e-12)
                    break
                lam *= 10
            else:
                return it + 1
            if improvement < tol * max(cost, 1.0):
                return it + 1
        return iterations

    def optimise(
        self,
        iterations: int = 100,
        huber_delta: float = 3.0,
        outlier_sigma: float = 5.0,
        passes: int = 2,
        tolerance: float = 1e-6,
    ) -> list[int]:
        """Solve the graph.

        Parameters
        ----------
        iterations
            Levenberg-Marquardt iterations per pass.
        huber_delta
            Mahalanobis error beyond which an edge is down-weighted.
        outlier_sigma
            Edges with an error above median + this many MADs are dropped
            after a pass, unless that would disconnect a scan.
        passes
            Solve / reject rounds.
        tolerance
            Stop when an iteration improves the cost by less than this share.

        Returns
        -------
        list of int
            Indices of the edges rejected as outliers.
        """
        self.initialise()
        rejected: list[int] = []
        for _ in range(passes):
            active = [e for k, e in enumerate(self.edges) if k not in rejected]
            self._solve(active, iterations, huber_delta, tolerance)
            if len(active) < 4:
                break
            err = np.array(
                [
                    np.sqrt(max((r := self._residual(e, self.poses)) @ e.information @ r, 0))
                    for e in active
                ]
            )
            med = np.median(err)
            mad = np.median(np.abs(err - med)) * 1.4826
            if mad <= 1e-9:
                break
            reach = self.connected(active)
            kept = list(active)
            new = []
            for k in np.argsort(-err):
                if err[k] <= med + outlier_sigma * mad:
                    break
                trial = [e for e in kept if e is not active[k]]
                if self.connected(trial) >= reach:
                    kept = trial
                    new.append(self.edges.index(active[k]))
            if not new:
                break
            rejected += new
        return sorted(rejected)


# --------------------------------------------------------------------------- #
# Whole survey
# --------------------------------------------------------------------------- #


@dataclass
class CoregResult:
    """Outcome of :func:`register_scans`.

    Attributes
    ----------
    poses
        ``world_from_scan`` per scan (the identity for unregistered scans).
    registered
        Boolean per scan.
    pairs
        Every attempted pair and placement, with its verdict.
    rejected
        Pairs accepted but then rejected by the pose graph as outliers.
    """

    poses: list[np.ndarray]
    registered: np.ndarray
    pairs: list[PairResult]
    rejected: list[PairResult]

    def report(self, names: Sequence[str] | None = None) -> str:
        """Plain-text summary: one line per accepted pair and the unregistered scans.

        Parameters
        ----------
        names
            Scan names; indices if None.

        Returns
        -------
        str
        """
        nm = (
            (lambda k: names[k] if k >= 0 else "survey")
            if names
            else (lambda k: str(k) if k >= 0 else "survey")
        )
        ok = [p for p in self.pairs if p.success]
        lines = [
            f"{int(self.registered.sum())} of {len(self.registered)} scans registered; "
            f"{len(ok)} accepted pairs, {len(self.rejected)} rejected by the pose graph"
        ]
        for p in ok:
            lines.append(
                f"  {nm(p.i):>14s} -> {nm(p.j):<14s} fitness {p.fitness:.2f} (above ground "
                f"{p.fitness_above:.2f}), rmse {p.rmse * 100:.1f} cm, {p.n_stems} stems, "
                f"stems off by {p.stem_residual * 100:.1f} cm"
            )
        missing = [nm(k) for k in np.flatnonzero(~self.registered)]
        if missing:
            lines.append("  unregistered: " + ", ".join(missing))
        return "\n".join(lines)


def _prior_ok(
    pose: np.ndarray,
    prior: np.ndarray,
    origin: np.ndarray,
    max_shift: float,
    max_rotation: float | None,
) -> tuple[bool, str]:
    """Does a ``world_from_scan`` put the scanner where its prior says it stood?"""
    shift = float(np.linalg.norm(_transform(pose, origin[None]) - _transform(prior, origin[None])))
    if shift > max_shift:
        return False, f"scanner {shift:.1f} m from its prior position"
    if max_rotation is not None:
        rot = float(np.degrees(np.linalg.norm(se3_log(invert(prior) @ pose)[:3])))
        if rot > max_rotation:
            return False, f"{rot:.1f} deg from the prior orientation"
    return True, ""


def _lowest_per_cell(xyz: np.ndarray, cell: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    key = np.floor(xyz[:, :2] / cell).astype(np.int64)
    k = key[:, 0] * 10_000_000 + key[:, 1]
    order = np.lexsort((xyz[:, 2], k))
    first = np.r_[True, k[order][1:] != k[order][:-1]]
    return k[order][first], xyz[order][first, 2]


def _ground_offset(points: np.ndarray, target: np.ndarray) -> float:
    """Median height of ``target``'s ground over ``points``' ground, per cell (NaN if none)."""
    ka, za = _lowest_per_cell(points)
    kb, zb = _lowest_per_cell(target)
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    return float(np.median(zb[ib] - za[ia])) if len(common) >= 50 else float("nan")


def place_from_prior(
    scan: ScanFeatures,
    survey: Sequence[ScanFeatures],
    poses: Sequence[np.ndarray],
    prior: np.ndarray,
    neighbours: int = 6,
    icp: dict | None = None,
    **accept,
) -> tuple[PairResult, list[int]]:
    """Place one scan starting from an approximate pose instead of stems.

    For scans that see too few stems to match. The prior (GNSS and compass, a
    RiSCAN SOP) is first corrected in height by the median offset between the
    scan's ground and its registered neighbours' ground, since a vertical
    error of metres is common and beyond ICP's reach; ICP then starts with a
    1.5 m correspondence distance.

    Parameters
    ----------
    scan
        The scan to place.
    survey
        Registered scans.
    poses
        ``world_from_scan`` of each of ``survey``.
    prior
        Approximate ``world_from_scan`` of ``scan``.
    neighbours
        Registered scans nearest to the prior that ICP runs against.
    icp, **accept
        As for :func:`register_pair`; the ICP levels default to a wider
        coarse-to-fine schedule.

    Returns
    -------
    result : PairResult
        ``transform`` is ``world_from_scan``; check ``success``.
    used : list of int
        Indices into ``survey`` of the scans ICP ran against.
    """
    start = time.perf_counter()
    acc = _Accept(**accept)
    result = PairResult(-1, -1)
    here = scan.location(prior)
    order = sorted(
        range(len(survey)), key=lambda n: float(np.linalg.norm(survey[n].location(poses[n]) - here))
    )
    used = [n for n in order if len(survey[n].points)][:neighbours]
    target = np.vstack([_transform(poses[n], survey[n].points) for n in used])
    coarse = np.asarray(prior, float).copy()
    dz = _ground_offset(_transform(coarse, scan.points), target)
    if not np.isfinite(dz):
        result.reason = "no ground shared with the registered scans"
        return result, []
    coarse[2, 3] += dz
    kw = {
        "levels": ((0.30, 1.50), (0.30, 0.80), (0.15, 0.40), (0.07, 0.20), (0.05, 0.12)),
        **(icp or {}),
    }
    _judge(result, scan, target, coarse, None, None, acc, kw)
    result.reason = f"from the prior, height corrected by {dz:+.2f} m; " + result.reason
    result.seconds = time.perf_counter() - start
    return result, used


def register_scans(
    scans: Sequence[ScanFeatures],
    positions: np.ndarray | None = None,
    fixed: dict[int, np.ndarray] | None = None,
    max_pair_distance: float = 40.0,
    priors: Sequence[np.ndarray] | None = None,
    max_prior_shift: float = 5.0,
    max_prior_rotation: float | None = None,
    workers: int = 4,
    recover: bool = True,
    neighbours: int = 6,
    match: dict | None = None,
    icp: dict | None = None,
    log: Callable[[str], None] | None = print,
    **accept,
) -> CoregResult:
    """Register a set of scans into one frame.

    Parameters
    ----------
    scans
        Prepared scans (:func:`prepare_scan`).
    positions
        Approximate ``(n, 3)`` scanner positions (GNSS, a sketch map); pairs
        further apart than ``max_pair_distance`` are not attempted. All pairs
        if None.
    fixed
        ``{index: world_from_scan}`` of scans whose poses are trusted and held
        fixed. With it, pairs are only attempted that involve a free scan, and
        free scans are placed into the fixed scans' frame. Scan 0 fixed at the
        identity if None.
    max_pair_distance
        Largest separation (m) of a pair attempted.
    priors
        Approximate ``world_from_scan`` per scan (a RiSCAN SOP, GNSS and
        compass). Stem matching is global and can, rarely, confirm a wrong
        alignment in a repetitive stand; with priors, any pair or placement
        that puts a scanner more than ``max_prior_shift`` (m) from its prior
        position is refused. Only the position is checked by default: GNSS
        positions are good to a metre or two, but recorded headings can be
        wrong by any amount (on the TERN Litchfield core plot two by 123 and
        155 degrees), and the stems then give the right one.
    max_prior_shift
        How far (m) a scanner may end up from its prior position.
    max_prior_rotation
        If set, also refuse results rotated more than this (degrees) from the
        prior orientation.
    workers
        Pairs registered in parallel (threads; the heavy parts release the GIL).
    recover
        Place scans left unregistered against the registered survey; with
        ``priors``, scans whose stems do not match are placed from their
        prior (:func:`place_from_prior`).
    neighbours
        Registered scans a placement refines against.
    match, icp, **accept
        As for :func:`register_pair`.
    log
        Called with progress messages; None for silence.

    Returns
    -------
    CoregResult
    """
    log = log or (lambda s: None)
    n = len(scans)
    fixed = {0: np.eye(4)} if not fixed else fixed
    pos = None if positions is None else np.asarray(positions, float)
    todo = []
    for i in range(n):
        for j in range(i + 1, n):
            if i in fixed and j in fixed:
                continue
            if not (scans[i].usable and scans[j].usable):
                continue
            if pos is not None and np.linalg.norm(pos[i, :2] - pos[j, :2]) > max_pair_distance:
                continue
            todo.append((i, j))
    log(f"registering {len(todo)} pairs of {n} scans ...")
    t0 = time.perf_counter()
    with progress.task("registering scan pairs", len(todo)) as prog, ThreadPoolExecutor(max(workers, 1)) as ex:
        def one(ij):
            r = register_pair(scans[ij[0]], scans[ij[1]], i=ij[0], j=ij[1], match=match, icp=icp, **accept)
            prog.update()
            return r

        pairs = list(ex.map(one, todo))
    if priors is not None:
        for p in pairs:
            if p.success:
                good, why = _prior_ok(
                    np.asarray(priors[p.j]) @ p.transform,
                    priors[p.i],
                    scans[p.i].origin,
                    max_prior_shift,
                    max_prior_rotation,
                )
                if not good:
                    p.success, p.reason = False, "refused: " + why
    ok = [p for p in pairs if p.success]
    log(f"  {len(ok)} accepted in {time.perf_counter() - t0:.0f} s")

    graph = PoseGraph(n, fixed)
    for p in ok:
        graph.add_edge(p.i, p.j, p.transform, p.rmse, p.fitness, p.n_correspondences)
    rejected_idx = graph.optimise() if ok else []
    rejected = [ok[k] for k in rejected_idx]
    registered = np.zeros(n, bool)
    registered[
        list(graph.connected([e for k, e in enumerate(graph.edges) if k not in rejected_idx]))
    ] = True

    if recover:
        for _ in range(2):
            pending = [k for k in range(n) if not registered[k] and scans[k].usable]
            if not pending:
                break
            reg = list(np.flatnonzero(registered))
            gained = 0
            for k in pending:
                r, used = place_scan(
                    scans[k],
                    [scans[m] for m in reg],
                    [graph.poses[m] for m in reg],
                    neighbours=neighbours,
                    match=match,
                    icp=icp,
                    **accept,
                )
                r.i = k
                if r.success and priors is not None:
                    good, why = _prior_ok(
                        r.transform, priors[k], scans[k].origin, max_prior_shift, max_prior_rotation
                    )
                    if not good:
                        r.success, r.reason = False, "refused: " + why
                if not r.success and priors is not None:
                    # Too few stems: start from the prior instead.
                    r2, used2 = place_from_prior(
                        scans[k],
                        [scans[m] for m in reg],
                        [graph.poses[m] for m in reg],
                        priors[k],
                        neighbours=neighbours,
                        icp=icp,
                        **{**accept, "max_shift": max_prior_shift},
                    )
                    r2.i = k
                    if r2.success:
                        good, why = _prior_ok(
                            r2.transform,
                            priors[k],
                            scans[k].origin,
                            max_prior_shift,
                            max_prior_rotation,
                        )
                        if not good:
                            r2.success, r2.reason = False, "refused: " + why
                    pairs.append(r)
                    r, used = r2, used2
                pairs.append(r)
                if not r.success:
                    log(f"  {scans[k].name or k}: still unplaced ({r.reason})")
                    continue
                graph.poses[k] = r.transform
                for u in used:
                    m_ = reg[u]
                    graph.add_edge(
                        k,
                        m_,
                        invert(graph.poses[m_]) @ r.transform,
                        r.rmse,
                        r.fitness,
                        max(r.n_correspondences // max(len(used), 1), 1),
                    )
                registered[k] = True
                gained += 1
                log(f"  {scans[k].name or k}: placed against {len(used)} registered scans")
            if not gained:
                break
            rejected_idx = graph.optimise()
            rejected = [ok[k] for k in rejected_idx if k < len(ok)]

    poses = [graph.poses[k] if registered[k] else np.eye(4) for k in range(n)]
    return CoregResult(poses, registered, pairs, rejected)
