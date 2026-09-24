# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""End-to-end coregistration of a survey.

The pipeline is staged, and each stage can run on its own:

1. **Per scan** (:func:`prepare_scan`): a terrain model, the stem map, and a
   planarity-filtered subsample for ICP. The expensive part, done once per
   scan however many pairs are tried.
2. **Per pair** (:func:`register_pair`): reflective targets where both scans
   saw them, otherwise a global stem-map match, gives a coarse transform that
   ICP refines; the pair is accepted only if it fits over all points and
   above the ground.
3. **Whole survey** (:func:`coregister_prepared`): accepted pairs become the
   edges of a pose graph solved with outlier rejection; scans left over are
   retried against the combined registered survey, and optionally every pose
   is refined jointly.

Every stage records its own quality, because the useful question is not "did
it run" but "which scans can I trust". This is a port of the author's
tlsalign. Two things are added: scans can be held fixed at trusted poses, so
that new scans are registered into an existing project; and approximate poses
(a RiSCAN SOP, GNSS and compass) can serve as priors, refusing results that
move a scanner implausibly far and placing scans that see too few stems.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .geometry import KdTree, planar_filter, voxel_downsample
from .ground import GroundModel, fit_ground
from .icp import ICPConfig, ICPResult, ICPTarget, evaluate_registration, icp
from .matching import MatchConfig, MatchResult, match_stem_maps
from .posegraph import OptimisationResult, PoseGraph
from .reflectors import Reflector, match_reflectors
from .stems import StemDetectionConfig, StemMap, detect_stems
from .transforms import identity, invert, se3_log, transform_difference, transform_points

__all__ = [
    "CoregConfig",
    "PairResult",
    "ScanFeatures",
    "SurveyResult",
    "coregister",
    "coregister_prepared",
    "load_transforms",
    "merge_clouds",
    "place_from_prior",
    "prepare_scan",
    "reading_options",
    "register_pair",
]


@dataclass
class CoregConfig:
    """Settings of the whole pipeline; the defaults are tlsalign's."""

    ground_cell_size: float = 0.5
    stems: StemDetectionConfig = field(default_factory=StemDetectionConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    icp: ICPConfig = field(default_factory=ICPConfig)

    icp_voxel: float = 0.05
    icp_min_planarity: float = 0.35
    """Planarity gate on the ICP points of every scan; 0 keeps foliage, which
    costs accuracy and can trap ICP in a local minimum."""
    icp_max_height: float = 12.0
    """Returns higher above ground (m) are left out of ICP: the upper canopy is
    the least repeatable part of the scene and the most moved by wind."""

    use_reflectors: bool = True
    """Try reflective targets before stems where both scans saw at least
    :attr:`min_reflector_matches`. Far more precise, but they usually cover
    only part of a survey."""
    min_reflector_matches: int = 3
    reflector_tolerance: float = 0.05
    trusted_reflector_matches: int = 5
    """A reflector match with at least this many targets, agreeing within
    :attr:`trusted_reflector_rmse`, is accepted even when ICP then fails its
    tests. Scans tens of metres apart share targets long before they share
    enough surface for ICP's fitness test, and so many targets cannot form a
    congruent pattern by coincidence. The ICP pose is kept if it stayed
    within :attr:`reflector_tolerance` of the targets', otherwise the
    targets' own. A departure from tlsalign, which always requires ICP to
    agree: 0 restores that."""
    trusted_reflector_rmse: float = 0.03

    min_match_inliers: int = 5
    """Stem correspondences a coarse match needs to reach ICP. Screening only
    saves ICP time; ICP fitness decides, so this is permissive."""
    max_match_ambiguity: float = 0.8
    """A coarse match whose best distinctly different rival has more than this
    share of its inliers is ambiguous: both are refined by ICP and the pair is
    accepted only if one fits the above-ground points clearly better."""
    ambiguity_margin: float = 1.25
    max_match_rmse: float = 0.30
    """Largest scatter (m) of a coarse match's own correspondences."""
    max_coarse_stem_rmse: float = float("inf")
    """Optional cap (m) on the median disagreement of matched stems. Off: it
    mixes horizontal agreement with the vertical difference of two ground
    models, which on steep terrain alone reaches a metre."""

    max_pair_distance: float = 40.0
    """Pairs whose approximate positions are further apart (m) are not tried.
    Scale it with the range: with a 30 m gate no pair more than 35 m apart
    ever registered (0 of 1687)."""
    screen_pairs: bool = True
    """Match every pair first and run ICP only on the survivors."""
    max_pairs_per_scan: int | None = None
    """After screening keep only each scan's best N pairs; None keeps all."""

    min_icp_fitness: float = 0.10
    max_icp_rmse: float = 0.15
    min_icp_fitness_above_ground: float = 0.06
    """Fitness over source points above :attr:`fitness_min_height`. Ground is
    planar and fits under any horizontal shift, so on open sites it can carry
    a wrong pair past :attr:`min_icp_fitness` alone; stems, branches and logs
    are what fix the horizontal position."""
    fitness_min_height: float = 1.0
    stem_agreement_tolerance: float = 0.25
    """If ICP leaves the matched stems this much (m) further apart than the
    coarse match did, the coarse transform is kept: ICP slid somewhere
    unrelated. Loose on purpose; tightening it rejected good refinements."""
    max_coarse_to_fine_shift: float = 2.0
    """ICP moving further than this (m) from the coarse solution diverged."""

    recover_unregistered: bool = True
    """Retry each unregistered scan against the combined registered survey: a
    scan can overlap several registered scans partly without overlapping any
    one of them enough."""
    recovery_rounds: int = 2
    recovery_neighbours: int = 6

    refine_multiview: bool = False
    """Refine every registered pose in one joint Gauss-Newton over the
    point-to-plane correspondences of every accepted pair plus the matched
    stems (:func:`sylva.coreg.refine_joint`)."""
    refinement_rounds: int = 3
    refinement_voxel_sizes: tuple[float, ...] = (0.10, 0.05, 0.03)
    refinement_max_distances: tuple[float, ...] = (0.30, 0.15, 0.08)
    refinement_points_per_scan: int = 400_000
    refinement_stem_weight: float = 0.05
    refinement_stem_radius: float = 0.15
    refinement_min_voxel_points: int = 1
    refinement_max_shift: float = 0.30
    """A refinement moving any scan further (m) is discarded."""

    optimise_globally: bool = True
    reference_scan: int = 0
    reject_outlier_edges: bool = True

    max_prior_shift: float = 5.0
    """With priors: results putting a scanner further (m) from its prior
    position are refused."""
    max_prior_rotation: float | None = None
    """With priors: if set, also refuse results rotated more (degrees) from the
    prior. Off, since recorded headings can be wrong by any amount."""

    workers: int = 0
    """Scans prepared and pairs registered at once; 0 picks from the cores and
    free memory (:attr:`memory_per_worker_gb` per preparing scan)."""
    memory_per_worker_gb: float = 4.0
    riscan_filter: str = "none"
    """Apply RiSCAN PRO's RXP import filter when reading (:func:`sylva.riscan.riscan_like_mask`):
    ``"none"``, ``"current"`` (drops echoes within 0.5 m of the scanner, as a
    current RiSCAN import) or ``"legacy"`` (also the weak, isolated echoes
    the older conversion discarded, a fifth to a third of a scan)."""
    riegl_options: dict = field(default_factory=dict)
    """RXP reading (build it from a RiSCAN export settings file with
    :func:`reading_options`): ``min_range``, ``max_range`` (m, from the scanner),
    ``min_deviation``/``max_deviation``, ``min_reflectance``/``max_reflectance``,
    ``min_amplitude``/``max_amplitude``, ``echoes``, ``library``. None bound by
    default, as tlsalign."""
    min_points_per_scan: int = 1000
    """Scans with fewer points are set aside rather than processed."""
    max_points_per_scan: int | None = None
    verbose: bool = True

    def to_dict(self) -> dict:
        """The whole configuration as plain data."""
        return asdict(self)


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


@dataclass
class ScanFeatures:
    """Everything later stages need from one scan; build with :func:`prepare_scan`.

    Attributes
    ----------
    icp_points
        Planar subsample for ICP (float32: every scan's stays resident).
    icp_heights
        Height above ground of each ICP point.
    levelling
        ``level_from_scan`` rotation applied to the raw points first; every
        pose is in the levelled frame, and
        :meth:`SurveyResult.transform_for` composes it back out.
    origin
        Scanner position in the levelled frame (the origin for a raw scan).
    error
        Why the scan is unusable; empty if it is fine.
    """

    name: str
    n_points: int
    ground: GroundModel | None
    stem_map: StemMap
    icp_points: np.ndarray
    reflectors: list[Reflector] = field(default_factory=list)
    icp_heights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    levelling: np.ndarray = field(default_factory=identity)
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    source: Path | None = None
    seconds: float = 0.0
    error: str = ""

    @property
    def usable(self) -> bool:
        """Can the scan take part in registration at all?"""
        return not self.error and len(self.icp_points) > 0

    def location(self, pose: np.ndarray) -> np.ndarray:
        """Where the scanner stood in the world, given this scan's ``world_from_scan``."""
        return transform_points(pose, self.origin[None])[0]

    def __repr__(self) -> str:
        return (
            f"ScanFeatures(name={self.name!r}, points={self.n_points}, stems={len(self.stem_map)}, "
            f"icp_points={len(self.icp_points)})"
        )


@dataclass
class PairResult:
    """Registration of one pair; ``transform`` maps scan ``i`` into scan ``j``.

    Attributes
    ----------
    matched_source, matched_target
        Positions of the matched stems in each scan's own frame.
    coarse_stem_rmse, fine_stem_rmse
        Median matched-stem distance under the coarse and the ICP transform.
    fitness_above
        ICP fitness over source points above ground; NaN if unknown.
    rival
        A distinctly different coarse hypothesis nearly as well supported,
        resolved by ICP.
    used_icp
        False when the coarse transform was kept over ICP's.
    """

    i: int
    j: int
    name_i: str = ""
    name_j: str = ""
    transform: np.ndarray = field(default_factory=identity)
    coarse_transform: np.ndarray = field(default_factory=identity)
    match: MatchResult | None = None
    reflector_match: object | None = None
    icp: ICPResult | None = None
    success: bool = False
    reason: str = ""
    seconds: float = 0.0
    matched_source: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    matched_target: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    coarse_stem_rmse: float = float("nan")
    fine_stem_rmse: float = float("nan")
    fitness_above: float = float("nan")
    rival: MatchResult | None = None
    used_icp: bool = True
    trusted: bool = False
    """Accepted on the strength of its reflector match although ICP failed
    (:attr:`CoregConfig.trusted_reflector_matches`)."""

    @property
    def fitness(self) -> float:
        return self.icp.fitness if self.icp else 0.0

    @property
    def rmse(self) -> float:
        return self.icp.inlier_rmse if self.icp else float("inf")

    @property
    def n_stem_matches(self) -> int:
        return self.match.n_inliers if self.match else 0

    @property
    def stem_rmse(self) -> float:
        """Stem agreement of the transform this pair reports."""
        return self.fine_stem_rmse if self.used_icp else self.coarse_stem_rmse

    def summary(self) -> str:
        """One line for the report."""
        status = "ok  " if self.success else "FAIL"
        if self.reflector_match is not None:
            stem = f" targets={self.reflector_match.n_inliers}"
        else:
            stem = f" stem={self.stem_rmse * 100:.1f}cm" if np.isfinite(self.stem_rmse) else ""
        return (
            f"[{status}] {self.name_i} -> {self.name_j}: stems={self.n_stem_matches:2d} "
            + (
                f"amb={self.match.ambiguity:.2f} "
                if self.match is not None and self.match.ambiguity > 0
                else ""
            )
            + f"fitness={self.fitness:.3f} "
            + (f"above={self.fitness_above:.3f} " if np.isfinite(self.fitness_above) else "")
            + f"rmse={self.rmse * 1000:6.1f} mm{stem} ({self.reason})"
        )


@dataclass
class SurveyResult:
    """Outcome of coregistering a survey.

    Attributes
    ----------
    poses
        ``world_from_levelled_scan`` per scan; see :meth:`transform_for` for
        the transform of the raw scan.
    registered
        Per scan, connected to the reference by accepted pairs.
    edge_to_pair
        Index into :attr:`pairs` of each pose-graph edge.
    """

    scans: list[ScanFeatures]
    pairs: list[PairResult]
    poses: list[np.ndarray]
    reference: int
    optimisation: OptimisationResult | None = None
    registered: list[bool] = field(default_factory=list)
    seconds: float = 0.0
    edge_to_pair: list[int] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.scans]

    def transform_for(self, index: int) -> np.ndarray:
        """``world_from_scan`` of one scan, applying directly to its raw points."""
        return self.poses[index] @ self.scans[index].levelling

    def successful_pairs(self) -> list[PairResult]:
        return [p for p in self.pairs if p.success]

    def pair_for_edge(self, edge_index: int) -> PairResult | None:
        """The pair behind a pose-graph edge."""
        if not 0 <= edge_index < len(self.edge_to_pair):
            return None
        return self.pairs[self.edge_to_pair[edge_index]]

    def rejected_pairs(self) -> list[PairResult]:
        """Accepted pairs the global solve found inconsistent with the rest."""
        if self.optimisation is None:
            return []
        return [
            p
            for p in (self.pair_for_edge(k) for k in self.optimisation.rejected_edges)
            if p is not None
        ]

    def consistency(self, robust: bool = True) -> dict[tuple[int, int], float]:
        """Stem disagreement per accepted pair under the final poses (m).

        The field-usable quality check: no ground truth needed, it measures
        whether the same tree lands in the same place from two scans.

        Parameters
        ----------
        robust
            Median rather than RMSE: in a dense stand a few coarse
            correspondences pair neighbouring trees, and would dominate an RMSE.

        Returns
        -------
        dict
            ``{(i, j): distance}``.
        """
        out = {}
        for pair in self.pairs:
            if not pair.success or len(pair.matched_source) == 0:
                continue
            relative = invert(self.poses[pair.j]) @ self.poses[pair.i]
            d = np.linalg.norm(
                transform_points(relative, pair.matched_source) - pair.matched_target, axis=1
            )
            out[(pair.i, pair.j)] = float(np.median(d) if robust else np.sqrt(np.mean(d**2)))
        return out

    def report(self) -> str:
        """Plain-text summary for a log or a QC record."""
        lines = [
            f"Coregistration of {len(self.scans)} scans ({sum(self.registered)} registered) "
            f"in {self.seconds:.1f} s",
            f"reference scan: {self.names[self.reference]}",
            "",
            "Scans:",
        ]
        for k, scan in enumerate(self.scans):
            flag = (
                f"SET ASIDE ({scan.error})"
                if scan.error
                else "registered"
                if self.registered[k]
                else "NOT REGISTERED"
            )
            lines.append(
                f"  {k:2d} {scan.name:<24s} {scan.n_points:>10,d} pts  "
                f"{len(scan.stem_map):3d} stems  {flag}"
            )
        lines += ["", "Pairs:"] + ["  " + p.summary() for p in self.pairs]
        consistency = self.consistency()
        if consistency:
            lines += [
                "",
                "Stem agreement under the final poses (median tree-to-tree distance, "
                "no ground truth needed):",
            ]
            # Flag pairs against the survey itself: the absolute level depends
            # on the stand and the spacing, but an outlier is always worth a look.
            threshold = max(3.0 * float(np.median(list(consistency.values()))), 0.10)
            for (i, j), value in sorted(consistency.items()):
                flag = "" if value <= threshold else "   <-- check"
                lines.append(f"  {self.names[i]} <-> {self.names[j]}: {value * 100:6.2f} cm{flag}")
        if self.optimisation is not None:
            opt = self.optimisation
            lines += [
                "",
                f"Global optimisation: {opt.iterations} iterations, converged={opt.converged}, "
                f"error {opt.initial_error:.4g} -> {opt.final_error:.4g}",
            ]
            if opt.rejected_edges:
                lines.append(
                    "  rejected inconsistent pairs: "
                    + ", ".join(f"{p.name_i}->{p.name_j}" for p in self.rejected_pairs())
                )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Write the transforms and quality as JSON (``transforms.json``).

        Returns
        -------
        pathlib.Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reference": self.reference,
            "seconds": self.seconds,
            "scans": [
                {
                    "index": k,
                    "name": scan.name,
                    "source": str(scan.source) if scan.source else None,
                    "n_points": scan.n_points,
                    "n_stems": len(scan.stem_map),
                    "registered": bool(self.registered[k]),
                    "world_from_scan": self.transform_for(k).tolist(),
                    "levelling": scan.levelling.tolist(),
                }
                for k, scan in enumerate(self.scans)
            ],
            "pairs": [
                {
                    "i": p.i,
                    "j": p.j,
                    "name_i": p.name_i,
                    "name_j": p.name_j,
                    "success": p.success,
                    "reason": p.reason,
                    "stem_matches": p.n_stem_matches,
                    "fitness": _json_number(p.fitness),
                    "rmse": _json_number(p.rmse),
                    "n_correspondences": p.icp.n_correspondences if p.icp else 0,
                    "coarse_stem_rmse": _json_number(p.coarse_stem_rmse),
                    "fine_stem_rmse": _json_number(p.fine_stem_rmse),
                    "used_icp": p.used_icp,
                    "trusted": p.trusted,
                    "transform": p.transform.tolist(),
                }
                for p in self.pairs
            ],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path


def load_transforms(path: str | Path) -> dict[str, np.ndarray]:
    """Read the ``world_from_scan`` of every registered scan from :meth:`SurveyResult.save`.

    Parameters
    ----------
    path
        A ``transforms.json``.

    Returns
    -------
    dict
        ``{name: (4, 4) matrix}``.
    """
    payload = json.loads(Path(path).read_text())
    return {
        s["name"]: np.asarray(s["world_from_scan"]) for s in payload["scans"] if s.get("registered")
    }


# --------------------------------------------------------------------------- #
# Stage 1: per scan
# --------------------------------------------------------------------------- #


def _make_logger(verbose: bool) -> Callable[[str], None]:
    return functools.partial(print, flush=True) if verbose else (lambda _message: None)


_READ_KEYS = {"library", "drop_pseudo_echoes", "echoes", "stride", "shot_stride"}
_GATES = ("range", "deviation", "reflectance", "amplitude")


def reading_options(settings: str | Path | None = None, **bounds) -> dict:
    """RXP reading options from a RiSCAN PRO export settings file and explicit bounds.

    Parameters
    ----------
    settings
        A RiSCAN export filter settings file (:func:`sylva.riscan.read_export_settings`).
    **bounds
        ``min_range``, ``max_range``, ``min_deviation``, ``max_deviation``,
        ``min_reflectance``, ``max_reflectance``, ``min_amplitude``,
        ``max_amplitude``; None leaves a bound to the file (or open).
        Explicit bounds override the file, as in tlsalign.

    Returns
    -------
    dict
        For :attr:`CoregConfig.riegl_options`.
    """
    from ..riscan import read_export_settings

    options: dict = {}
    if settings:
        for name, (lo, hi) in read_export_settings(settings).items():
            if lo > -np.inf:
                options[f"min_{name}"] = lo
            if hi < np.inf:
                options[f"max_{name}"] = hi
    options.update({k: v for k, v in bounds.items() if v is not None})
    return options


def _read_scan(path: Path, cfg: CoregConfig) -> tuple[np.ndarray, dict]:
    """Points of a scan file in its own frame, filtered as tlsalign reads them.

    RiSCAN's import filter (:attr:`CoregConfig.riscan_filter`) is decided on
    the whole stream first; then the closed intervals on range (from the
    scanner), deviation, reflectance and amplitude are applied.
    """
    from .. import io
    from ..riscan import riscan_like_mask

    opts = dict(cfg.riegl_options)
    if path.suffix.lower() == ".rxp":
        read = {k: v for k, v in opts.items() if k in _READ_KEYS}
        read["min_range"] = 0.0  # every range is read; the gate below is tlsalign's
        cloud = io.read_rxp(path, **read)
    else:
        cloud = io.read(path)
    xyz = cloud.xyz
    keep = np.ones(len(xyz), dtype=bool)
    if cfg.riscan_filter != "none":
        if "amplitude" not in cloud.attrs:
            raise KeyError(f"the RiSCAN filter needs amplitude, which {path.name} does not carry")
        keep &= riscan_like_mask(xyz, cloud.attrs["amplitude"], cfg.riscan_filter)
    for name in _GATES:
        lo, hi = opts.get(f"min_{name}"), opts.get(f"max_{name}")
        if lo is None and hi is None:
            continue
        if name == "range":
            values = np.einsum("ij,ij->i", xyz, xyz)
            lo, hi = (None if lo is None else lo * lo), (None if hi is None else hi * hi)
        elif name in cloud.attrs:
            values = np.asarray(cloud.attrs[name], dtype=np.float64)
            if name == "deviation":
                values = np.where(values == 65535, -1.0, values)  # RIEGL's "not measured"
        else:
            raise KeyError(f"reading bounds {name}, which {path.name} does not carry")
        if lo is not None:
            keep &= values >= lo
        if hi is not None:
            keep &= values <= hi
    xyz = xyz[keep]
    if cfg.max_points_per_scan and len(xyz) > cfg.max_points_per_scan:
        xyz = xyz[: cfg.max_points_per_scan]
    return xyz, {}


def prepare_scan(
    cloud,
    config: CoregConfig | None = None,
    name: str = "",
    reflectors: list[Reflector] | None = None,
    levelling: np.ndarray | None = None,
    origin=None,
) -> ScanFeatures:
    """Fit the ground, detect stems and build the ICP subsample of one scan.

    Parameters
    ----------
    cloud
        A :class:`sylva.PointCloud`, an ``(n, 3)`` array, or a path (``.rxp``
        through RiVLib, anything else through :func:`sylva.read`).
    config
        Pipeline settings.
    name
        Scan name, for reports.
    reflectors
        Targets the scan saw, in its own frame.
    levelling
        ``(4, 4)`` rotation applied to the points (and targets) first, for a
        scanner that was not upright; the scanner's own attitude is the usual
        source, or the rotation of a RiSCAN SOP. Recorded, so poses can be
        expressed back in the scanner's frame.
    origin
        Scanner position in the (levelled) cloud; the origin for a raw scan.

    Returns
    -------
    ScanFeatures
    """
    from ..pointcloud import PointCloud

    cfg = config or CoregConfig()
    start = time.perf_counter()
    level = identity() if levelling is None else np.asarray(levelling, dtype=np.float64)
    source = None
    if isinstance(cloud, (str, Path)):
        source = Path(cloud)
        points, _ = _read_scan(source, cfg)
        name = name or source.stem
    elif isinstance(cloud, PointCloud):
        points = cloud.xyz
    else:
        points = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)
    reflectors = list(reflectors or [])
    if levelling is not None:
        points = transform_points(level, points)
        reflectors = [
            replace(
                r, **dict(zip("xyz", transform_points(level, r.position[None])[0], strict=True))
            )
            for r in reflectors
        ]
    scanner = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)

    if len(points) < cfg.min_points_per_scan:
        return _unusable_scan(
            name,
            len(points),
            source,
            start,
            f"only {len(points):,} points" + (" after filtering" if cfg.riegl_options else ""),
        )

    ground = fit_ground(points, cfg.ground_cell_size)
    heights = ground.normalise(points, dtype=np.float32)
    stem_map = detect_stems(points, ground, cfg.stems, name=name, heights=heights)
    below_canopy = points[heights <= cfg.icp_max_height]
    if cfg.icp_min_planarity > 0:
        icp_points = planar_filter(
            below_canopy, min_planarity=cfg.icp_min_planarity, voxel=cfg.icp_voxel
        )
    else:
        icp_points = voxel_downsample(below_canopy, cfg.icp_voxel)
    icp_points = np.ascontiguousarray(icp_points, dtype=np.float32)
    return ScanFeatures(
        name=name,
        n_points=len(points),
        ground=ground,
        stem_map=stem_map,
        icp_points=icp_points,
        reflectors=reflectors,
        icp_heights=ground.normalise(icp_points, dtype=np.float32),
        levelling=level,
        origin=scanner,
        source=source,
        seconds=time.perf_counter() - start,
    )


def _unusable_scan(
    name: str, n_points: int, source: Path | None, start: float, reason: str
) -> ScanFeatures:
    """A placeholder for a scan that cannot be registered: returned, not raised,
    so one damaged file does not cost a survey the scans already prepared."""
    return ScanFeatures(
        name=name,
        n_points=n_points,
        ground=None,
        stem_map=StemMap([], name=name),
        icp_points=np.zeros((0, 3), dtype=np.float32),
        source=source,
        seconds=time.perf_counter() - start,
        error=reason,
    )


def _resolve_workers(requested: int, memory_per_worker_gb: float | None, n_tasks: int) -> int:
    """Workers to use: ``requested`` if positive, else from the cores and free memory."""
    if n_tasks <= 1:
        return 1
    if requested and requested > 0:
        return min(requested, n_tasks)
    workers = os.cpu_count() or 1
    if memory_per_worker_gb:
        try:
            with open("/proc/meminfo") as f:
                available = (
                    next(int(line.split()[1]) for line in f if line.startswith("MemAvailable"))
                    / 2**20
                )
            workers = min(workers, max(int(available // memory_per_worker_gb), 1))
        except (OSError, StopIteration, ValueError):
            pass
    return max(1, min(workers, n_tasks))


def _parallel_map(func, items: list, workers: int, on_result=None) -> list:
    """``func`` over ``items`` in threads (the heavy parts release the GIL), in order."""
    if workers <= 1:
        out = []
        for item in items:
            out.append(func(item))
            if on_result:
                on_result(out[-1])
        return out
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(func, item) for item in items]
        out = []
        for f in futures:
            out.append(f.result())
            if on_result:
                on_result(out[-1])
        return out


# --------------------------------------------------------------------------- #
# Stage 2: pairs
# --------------------------------------------------------------------------- #


def register_pair(
    source: ScanFeatures,
    target: ScanFeatures,
    config: CoregConfig | None = None,
    *,
    initial: np.ndarray | None = None,
    match: MatchResult | None = None,
    i: int = 0,
    j: int = 1,
    target_icp: ICPTarget | None = None,
) -> PairResult:
    """Coarse-match then ICP-refine one pair of prepared scans.

    Parameters
    ----------
    source, target
        Prepared scans; the result maps ``source`` into ``target``.
    config
        Pipeline settings.
    initial
        Skip the coarse matching and start ICP here (GNSS, a target-based
        prealignment, a previous run).
    match
        A stem match already computed (by screening).
    i, j
        Indices recorded on the result.
    target_icp
        ``target``'s ICP pyramid, prepared once for all its pairs
        (:class:`ICPTarget`, built with ``config.icp``); built here if None.

    Returns
    -------
    PairResult
    """
    cfg = config or CoregConfig()
    start = time.perf_counter()
    result = PairResult(i=i, j=j, name_i=source.name, name_j=target.name)
    if initial is None:
        # Targets first: a target is located to millimetres and three fix all
        # six degrees of freedom. Stems stay as a fallback, because three
        # targets can form a congruent triangle by coincidence.
        return _register_with_fallback(result, source, target, cfg, match, start, target_icp)
    _refine_and_judge(result, source, target, cfg, np.asarray(initial, dtype=float), target_icp)
    result.seconds = time.perf_counter() - start
    return result


def _register_with_fallback(
    result, source, target, cfg, match, start, target_icp=None
) -> PairResult:
    """Each coarse method in turn; the first ICP accepts wins. A target match
    ICP rejects must not cost the pair its stem match."""
    attempts = (
        ["reflectors"] if cfg.use_reflectors and source.reflectors and target.reflectors else []
    ) + ["stems"]
    last = result
    for method in attempts:
        attempt = PairResult(i=result.i, j=result.j, name_i=result.name_i, name_j=result.name_j)
        coarse = _coarse_transform(attempt, source, target, cfg, match, method)
        if coarse is None:
            last = attempt if attempt.reason else last
            continue
        _refine_and_judge(attempt, source, target, cfg, coarse, target_icp)
        attempt.seconds = time.perf_counter() - start
        if attempt.success:
            return attempt
        last = attempt
    last.seconds = time.perf_counter() - start
    return last


def _coarse_transform(result, source, target, cfg, match, method) -> np.ndarray | None:
    if method == "reflectors":
        found = match_reflectors(
            source.reflectors,
            target.reflectors,
            tolerance=cfg.reflector_tolerance,
            min_inliers=cfg.min_reflector_matches,
        )
        if not found.success:
            return None
        result.reflector_match = found
        result.reason = f"{found.n_inliers} reflectors, {found.rmse * 1000:.1f} mm"
        return found.transform
    if match is None:
        match = match_stem_maps(source.stem_map, target.stem_map, cfg.matching)
    result.match = match
    _attach_matched_stems(result, source, target, match)
    if not _coarse_is_acceptable(result, match, cfg):
        return None
    return match.transform


def _refine_and_judge(
    result: PairResult,
    source: ScanFeatures,
    target: ScanFeatures,
    cfg: CoregConfig,
    coarse: np.ndarray,
    target_icp: ICPTarget | None = None,
) -> None:
    """Refine a coarse transform with ICP and decide whether to accept it."""
    if target_icp is None and result.rival is not None:
        target_icp = ICPTarget(target.icp_points, cfg.icp)  # two ICPs against it
    icp_target = target.icp_points if target_icp is None else target_icp
    result.coarse_transform = coarse
    refined = icp(source.icp_points, icp_target, coarse, cfg.icp)
    result.icp = refined
    result.transform = refined.transform
    # Cross-check ICP against the stems it was meant to refine.
    if len(result.matched_source) >= 3:
        result.coarse_stem_rmse = _stem_median_residual(coarse, result)
        result.fine_stem_rmse = _stem_median_residual(refined.transform, result)
        if result.fine_stem_rmse > result.coarse_stem_rmse + cfg.stem_agreement_tolerance:
            result.transform = coarse
            result.used_icp = False
    result.fitness_above = _above_ground_fitness(
        source.icp_points, source.icp_heights, target.icp_points, result.transform, cfg
    )
    if result.rival is not None:
        # An ambiguous stem pattern: refine the rival too and keep whichever
        # fits the above-ground points better, if the margin is clear.
        other = icp(source.icp_points, icp_target, result.rival.transform, cfg.icp)
        other_above = _above_ground_fitness(
            source.icp_points, source.icp_heights, target.icp_points, other.transform, cfg
        )
        mine = result.fitness_above if np.isfinite(result.fitness_above) else 0.0
        theirs = other_above if np.isfinite(other_above) else 0.0
        # Two hypotheses ICP pulls to the same pose were never rivals.
        _, apart = transform_difference(result.transform, other.transform)
        yaw_apart = np.degrees(
            abs(
                np.arctan2(result.transform[1, 0], result.transform[0, 0])
                - np.arctan2(other.transform[1, 0], other.transform[0, 0])
            )
        )
        yaw_apart = min(yaw_apart, 360 - yaw_apart)
        converged = (
            apart < cfg.matching.distinct_translation and yaw_apart < cfg.matching.distinct_yaw_deg
        )
        if converged:
            result.reason = "rivals converged in ICP; "
        elif theirs > mine:
            winner = result.rival
            result.match = winner
            _attach_matched_stems(result, source, target, winner)
            coarse = winner.transform
            result.coarse_transform = coarse
            refined, result.icp, result.transform = other, other, other.transform
            result.fitness_above, mine, theirs = other_above, theirs, mine
            result.used_icp = True
            if len(result.matched_source) >= 3:
                result.coarse_stem_rmse = _stem_median_residual(coarse, result)
                result.fine_stem_rmse = _stem_median_residual(refined.transform, result)
        if not converged:
            if mine < cfg.ambiguity_margin * theirs:
                result.reason = (
                    f"ambiguous stem pattern; ICP cannot separate the rivals "
                    f"(above-ground fitness {mine:.3f} vs {theirs:.3f})"
                )
                return
            result.reason = f"rival resolved by ICP ({mine:.3f} vs {theirs:.3f} above ground); "
    _, shift = transform_difference(coarse, result.transform)
    if shift > cfg.max_coarse_to_fine_shift:
        result.reason = f"ICP diverged from the coarse solution by {shift:.2f} m"
    elif refined.fitness < cfg.min_icp_fitness:
        result.reason = f"low ICP fitness ({refined.fitness:.3f} < {cfg.min_icp_fitness})"
    elif result.fitness_above < cfg.min_icp_fitness_above_ground:
        result.reason = (
            f"low above-ground fitness ({result.fitness_above:.3f} < "
            f"{cfg.min_icp_fitness_above_ground}); ground alone matched"
        )
    elif refined.inlier_rmse > cfg.max_icp_rmse:
        result.reason = f"high ICP rmse ({refined.inlier_rmse:.3f} m > {cfg.max_icp_rmse})"
    else:
        result.success = True
        if result.reflector_match is None:
            kept = (
                result.reason
                if result.reason.startswith(("rival resolved", "rivals converged"))
                else ""
            )
            result.reason = kept + f"shift from coarse {shift * 100:.1f} cm"
        elif not result.used_icp:
            result.reason += " (kept coarse)"
    if not result.success:
        _trust_reflectors(result, coarse, shift, cfg)


def _trust_reflectors(
    result: PairResult, coarse: np.ndarray, shift: float, cfg: CoregConfig
) -> None:
    """Accept a pair ICP refused if its reflector match is strong enough alone."""
    found = result.reflector_match
    if (
        found is None
        or cfg.trusted_reflector_matches <= 0
        or found.n_inliers < cfg.trusted_reflector_matches
        or found.rmse > cfg.trusted_reflector_rmse
        or result.reason.startswith("ambiguous")
    ):
        return
    why = result.reason
    if shift > cfg.reflector_tolerance:
        result.transform, result.used_icp = coarse, False
    result.success, result.trusted = True, True
    result.reason = (
        f"{found.n_inliers} reflectors, {found.rmse * 1000:.1f} mm, trusted "
        f"({'targets' if not result.used_icp else 'ICP'} pose; ICP alone: {why})"
    )


def _edge_quality(pair: PairResult) -> tuple[float, float, int]:
    """``(fitness, rmse, correspondences)`` weighting a pair's pose-graph edge.

    A trusted reflector pair that kept the targets' pose is weighted by the
    targets' own residual and count, not by an ICP that did not fit.
    """
    if pair.trusted and not pair.used_icp and pair.reflector_match is not None:
        found = pair.reflector_match
        return 1.0, max(found.rmse, 1e-3), found.n_inliers
    return pair.fitness, pair.rmse, pair.icp.n_correspondences if pair.icp else 0


def _above_ground_fitness(source, heights, target, transform, cfg: CoregConfig) -> float:
    """ICP fitness over source points more than ``fitness_min_height`` up; NaN
    ("no evidence", not a failure) if heights are missing or too few points qualify."""
    if cfg.fitness_min_height <= 0 or len(heights) != len(source):
        return float("nan")
    mask = heights > cfg.fitness_min_height
    if mask.sum() < 100:
        return float("nan")
    fitness, _, _ = evaluate_registration(
        source[mask], target, transform, threshold=cfg.icp.fitness_threshold
    )
    return float(fitness)


def _attach_matched_stems(
    result: PairResult, source: ScanFeatures, target: ScanFeatures, match: MatchResult
) -> None:
    """Store the positions of the matched stems (positions, not indices, so
    nothing later depends on the matcher's ordering)."""
    if not match.success or len(match.correspondences) == 0:
        return
    result.matched_source = source.stem_map.positions[match.correspondences[:, 0]]
    result.matched_target = target.stem_map.positions[match.correspondences[:, 1]]


def _coarse_is_acceptable(result: PairResult, match: MatchResult, cfg: CoregConfig) -> bool:
    """Screen a coarse match before paying for ICP; sets ``reason`` on failure."""
    if not match.success or match.n_inliers < cfg.min_match_inliers:
        result.reason = (
            f"stem matching failed ({match.n_inliers} inliers, need {cfg.min_match_inliers})"
        )
        return False
    if match.inlier_rmse > cfg.max_match_rmse:
        result.reason = (
            f"coarse match too loose ({match.inlier_rmse * 100:.1f} cm scatter, "
            f"limit {cfg.max_match_rmse * 100:.0f} cm)"
        )
        return False
    if match.ambiguity > cfg.max_match_ambiguity:
        if match.rival is None or not match.rival.success:
            result.reason = (
                f"ambiguous stem pattern (a rival alignment has {match.ambiguity:.0%} "
                f"of the inliers, limit {cfg.max_match_ambiguity:.0%})"
            )
            return False
        result.rival = match.rival
    result.coarse_stem_rmse = _stem_median_residual(match.transform, result)
    if result.coarse_stem_rmse > cfg.max_coarse_stem_rmse:
        result.reason = (
            f"coarse stems disagree by {result.coarse_stem_rmse:.2f} m "
            f"(limit {cfg.max_coarse_stem_rmse:.2f} m)"
        )
        return False
    return True


def _stem_median_residual(transform: np.ndarray, pair: PairResult) -> float:
    if len(pair.matched_source) == 0:
        return float("nan")
    residual = transform_points(transform, pair.matched_source) - pair.matched_target
    return float(np.median(np.linalg.norm(residual, axis=1)))


# --------------------------------------------------------------------------- #
# Stage 3: the survey
# --------------------------------------------------------------------------- #


def coregister(
    clouds: Sequence,
    config: CoregConfig | None = None,
    *,
    names: Sequence[str] | None = None,
    pairs: Iterable[tuple[int, int]] | None = None,
    reflectors: Sequence[list] | None = None,
    approximate_positions: np.ndarray | None = None,
    levelling: Sequence[np.ndarray | None] | None = None,
    fixed: dict[int, np.ndarray] | None = None,
    priors: Sequence[np.ndarray | None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> SurveyResult:
    """Coregister a whole survey.

    Parameters
    ----------
    clouds
        Scans: paths, :class:`sylva.PointCloud` or ``(n, 3)`` arrays.
    config
        Pipeline settings.
    names
        Scan names.
    pairs
        Pairs to try; all by default, which suits plot-scale surveys since the
        extra edges make the global solve robust.
    reflectors
        Per scan, the targets it saw (in its own frame).
    approximate_positions
        ``(n, 3)`` rough scanner positions (GNSS); pairs further apart than
        ``max_pair_distance`` are skipped. NaN rows are unknown.
    levelling
        Per scan, the ``(4, 4)`` rotation that levels it.
    fixed
        ``{index: world_from_levelled_scan}`` of scans whose poses are
        trusted and held; the others are registered into their frame.
    priors
        Per scan, an approximate ``world_from_levelled_scan`` (a RiSCAN SOP);
        see :func:`coregister_prepared`.
    progress
        Called with progress messages; printed if None and ``verbose``.

    Returns
    -------
    SurveyResult
        Poses are ``world_from_scan`` with the world fixed to the reference
        scan (or to the fixed scans).
    """
    cfg = config or CoregConfig()
    start = time.perf_counter()
    log = progress or _make_logger(cfg.verbose)
    scan_names = [(names[k] if names else "") or f"scan_{k:02d}" for k in range(len(clouds))]
    targets = list(reflectors) if reflectors else [[]] * len(clouds)
    levels = list(levelling) if levelling else [None] * len(clouds)
    paths_only = all(isinstance(c, (str, Path)) for c in clouds)
    workers = (
        _resolve_workers(cfg.workers, cfg.memory_per_worker_gb, len(clouds)) if paths_only else 1
    )
    log(
        f"Preparing {len(clouds)} scans" + (f" on {workers} workers ..." if workers > 1 else " ...")
    )

    def report(f: ScanFeatures) -> None:
        if f.error:
            log(f"  {f.name:<24s} SET ASIDE: {f.error}")
        else:
            log(
                f"  {f.name:<24s} {f.n_points:>10,d} pts -> {len(f.stem_map):3d} stems, "
                f"{len(f.icp_points):>8,d} ICP points ({f.seconds:.1f} s)"
            )

    def one(k: int) -> ScanFeatures:
        t0 = time.perf_counter()
        try:
            return prepare_scan(
                clouds[k], cfg, name=scan_names[k], reflectors=targets[k], levelling=levels[k]
            )
        except Exception as exc:  # one unreadable file costs that file, not the run
            src = Path(clouds[k]) if isinstance(clouds[k], (str, Path)) else None
            return _unusable_scan(scan_names[k], 0, src, t0, f"{type(exc).__name__}: {exc}")

    scans = _parallel_map(one, list(range(len(clouds))), workers, on_result=report)
    return coregister_prepared(
        scans,
        cfg,
        pairs=pairs,
        approximate_positions=approximate_positions,
        fixed=fixed,
        priors=priors,
        progress=log,
        started=start,
    )


def coregister_prepared(
    scans: Sequence[ScanFeatures],
    config: CoregConfig | None = None,
    *,
    pairs: Iterable[tuple[int, int]] | None = None,
    approximate_positions: np.ndarray | None = None,
    fixed: dict[int, np.ndarray] | None = None,
    priors: Sequence[np.ndarray | None] | None = None,
    progress: Callable[[str], None] | None = None,
    started: float | None = None,
) -> SurveyResult:
    """Pairwise registration and the global solve on prepared scans.

    Parameters
    ----------
    scans
        From :func:`prepare_scan`; the expensive part, which depends on no
        other scan, so it can be cached or computed elsewhere.
    config
        Pipeline settings.
    pairs
        Pairs to try; all usable pairs within ``max_pair_distance`` if None.
    approximate_positions
        ``(n, 3)`` rough scanner positions; taken from ``priors`` if None.
    fixed
        ``{index: world_from_levelled_scan}`` held fixed. Pairs between two
        fixed scans are not tried.
    priors
        Per scan, an approximate ``world_from_levelled_scan`` or None. Stem
        matching is global and can, rarely, confirm a wrong alignment in a
        repetitive stand: any pair or placement that puts a scanner more than
        ``max_prior_shift`` from its prior position is refused. Scans whose
        stems do not match are placed from their prior
        (:func:`place_from_prior`). Only the position is checked by default:
        recorded headings can be wrong by any amount.
    progress
        Called with progress messages.
    started
        ``time.perf_counter()`` at the start, for the reported duration.

    Returns
    -------
    SurveyResult
    """
    cfg = config or CoregConfig()
    scans = list(scans)
    n = len(scans)
    start = time.perf_counter() if started is None else started
    log = progress or _make_logger(cfg.verbose)
    fixed = {int(k): np.asarray(v, float) for k, v in (fixed or {}).items()}
    priors = (
        None if priors is None else [None if p is None else np.asarray(p, float) for p in priors]
    )
    if approximate_positions is None and priors is not None:
        approximate_positions = np.array(
            [
                scans[k].location(p) if p is not None else np.full(3, np.nan)
                for k, p in enumerate(priors)
            ]
        )

    usable = [k for k, s in enumerate(scans) if s.usable]
    if len(usable) < n:
        log(
            f"  {n - len(usable)} scan(s) set aside: "
            + ", ".join(s.name for s in scans if not s.usable)
        )
    if pairs is not None:
        candidate_pairs = list(pairs)
    else:
        candidate_pairs = [
            (i, j)
            for a, i in enumerate(usable)
            for j in usable[a + 1 :]
            if not (i in fixed and j in fixed)
        ]
        candidate_pairs = _within_reach(
            candidate_pairs, approximate_positions, cfg.max_pair_distance, log
        )
    workers = _resolve_workers(cfg.workers, None, len(candidate_pairs))

    results: list[PairResult] = []
    if cfg.screen_pairs:
        log(f"Screening {len(candidate_pairs)} pairs by stem matching ...")
        matches, rejected = _screen(candidate_pairs, scans, cfg, workers)
        results.extend(rejected)
        if len(rejected) <= 20:
            for pair in rejected:
                log("  " + pair.summary())
        log(
            f"  kept {len(matches)} of {len(candidate_pairs)} pairs for ICP "
            f"({len(rejected)} rejected by stem screening)"
        )
    else:
        matches = [((i, j), None) for i, j in candidate_pairs]
    log(
        f"Refining {len(matches)} pairs with ICP"
        + (f" on {workers} workers ..." if workers > 1 else " ...")
    )
    # Pairs sharing a target run together, so each target's ICP pyramid is
    # built once and a small cache is enough however large the survey.
    matches = sorted(matches, key=lambda t: (t[0][1], t[0][0]))
    targets = _TargetCache(scans, cfg.icp, capacity=workers + 2)
    refined = _parallel_map(
        lambda t: register_pair(
            scans[t[0][0]],
            scans[t[0][1]],
            cfg,
            match=t[1],
            i=t[0][0],
            j=t[0][1],
            target_icp=targets.get(t[0][1]),
        ),
        matches,
        workers,
        on_result=lambda p: log("  " + p.summary()),
    )
    if priors is not None:
        for p in refined:
            if p.success and priors[p.i] is not None and priors[p.j] is not None:
                good, why = _prior_ok(
                    priors[p.j] @ p.transform, priors[p.i], scans[p.i].origin, cfg
                )
                if not good:
                    p.success, p.reason = False, "refused by the prior: " + why
                    log(f"  {p.name_i} -> {p.name_j} refused by the prior: {why}")
    results.extend(refined)
    results.sort(key=lambda p: (p.i, p.j))

    reference = min(max(cfg.reference_scan, 0), n - 1)
    if fixed:
        reference = reference if reference in fixed else min(fixed)
    elif not scans[reference].usable:
        replacement = next(iter(usable), reference)
        if replacement != reference:
            log(f"  reference {scans[reference].name} is unusable; using {scans[replacement].name}")
        reference = replacement
    graph = PoseGraph(n, reference=reference, fixed=fixed)
    edge_to_pair: list[int] = []
    for k, pair in enumerate(results):
        if pair.success:
            edge_to_pair.append(k)
            fitness, rmse, n_corr = _edge_quality(pair)
            graph.add_edge(
                pair.i,
                pair.j,
                pair.transform,
                fitness=fitness,
                rmse=rmse,
                n_correspondences=n_corr,
                label=f"{pair.name_i}->{pair.name_j}",
            )
    if not fixed:
        # A reference no accepted pair touches would leave every other scan
        # "unregistered" even when they registered to each other.
        rerooted = _reference_in_largest_component(graph, reference, n)
        if rerooted != reference:
            log(
                f"  reference {scans[reference].name} has no accepted pair; anchoring on "
                f"{scans[rerooted].name}, the largest registered block"
            )
            reference = rerooted
            graph.reference = reference

    graph.initialise(reference)
    optimisation = None
    if cfg.optimise_globally and graph.edges:
        log("Optimising the pose graph ...")
        optimisation = graph.optimise(reject_outliers=cfg.reject_outlier_edges)
        log(f"  {optimisation}")
    registered = _registered_mask(graph, n)
    if cfg.recover_unregistered and not all(registered):
        recovered = _recover_unregistered(
            scans, graph, results, edge_to_pair, registered, cfg, log, priors
        )
        if recovered:
            log(f"Recovered {recovered} scan(s); re-optimising ...")
            optimisation = graph.optimise(reject_outliers=cfg.reject_outlier_edges)
            log(f"  {optimisation}")
            registered = _registered_mask(graph, n)
    if cfg.refine_multiview and sum(registered) > 1:
        _refine_multiview(scans, graph, registered, reference, cfg, log)

    survey = SurveyResult(
        scans=scans,
        pairs=results,
        poses=list(graph.poses),
        reference=reference,
        optimisation=optimisation,
        registered=registered,
        seconds=time.perf_counter() - start,
        edge_to_pair=edge_to_pair,
    )
    if not all(registered):
        missing = [scans[k].name for k, ok in enumerate(registered) if not ok]
        log(f"WARNING: {len(missing)} scan(s) could not be registered: {', '.join(missing)}")
    return survey


def _refine_multiview(
    scans, graph: PoseGraph, registered, reference, cfg: CoregConfig, log
) -> None:
    from .refine import refine_joint

    edges = [(e.i, e.j) for e in graph.edges if registered[e.i] and registered[e.j]]
    if not edges:
        return
    stems = [
        s.stem_map.positions if registered[k] else np.zeros((0, 3)) for k, s in enumerate(scans)
    ]
    points = [
        s.icp_points if registered[k] else np.zeros((0, 3), np.float32) for k, s in enumerate(scans)
    ]
    log(
        f"Joint multi-view refinement: {sum(registered)} scans, {len(edges)} pairs, "
        f"{sum(len(s) for s in stems)} stems ..."
    )
    outcome = refine_joint(
        points,
        list(graph.poses),
        edges,
        stems,
        reference,
        voxel_sizes=tuple(cfg.refinement_voxel_sizes),
        max_distances=tuple(cfg.refinement_max_distances),
        rounds=max(cfg.refinement_rounds, 1),
        stem_weight=cfg.refinement_stem_weight,
        stem_radius=cfg.refinement_stem_radius,
        min_voxel_points=cfg.refinement_min_voxel_points,
        points_per_scan=cfg.refinement_points_per_scan,
        log=log,
    )
    moved = outcome.shifts[[k for k in range(len(scans)) if registered[k]]]
    worst = int(np.argmax(outcome.shifts))
    log(
        f"  residual {outcome.residual_before * 100:.2f} -> {outcome.residual_after * 100:.2f} cm; "
        f"shifts median {np.median(moved) * 100:.1f} cm, "
        f"max {outcome.shifts[worst] * 100:.1f} cm ({scans[worst].name}), "
        f"rotation max {outcome.rotations.max():.3f} deg"
    )
    if outcome.shifts.max() > cfg.refinement_max_shift:
        log(
            f"  WARNING: {scans[worst].name} moved {outcome.shifts[worst] * 100:.0f} cm, more than "
            f"refinement_max_shift; the pairwise poses are kept"
        )
        return
    for k in range(len(scans)):
        if registered[k] and k not in graph.fixed:
            graph.poses[k] = outcome.poses[k]


class _TargetCache:
    """The ICP pyramids of the most recently used targets."""

    def __init__(self, scans, config: ICPConfig, capacity: int) -> None:
        self._scans, self._config, self._capacity = scans, config, max(capacity, 1)
        self._built: OrderedDict[int, ICPTarget] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, k: int) -> ICPTarget:
        with self._lock:  # built under the lock: a pyramid is built once, in parallel inside
            if k in self._built:
                self._built.move_to_end(k)
            else:
                self._built[k] = ICPTarget(self._scans[k].icp_points, self._config)
                while len(self._built) > self._capacity:
                    self._built.popitem(last=False)
            return self._built[k]


def _within_reach(candidate_pairs, positions, limit, log):
    """Drop pairs whose approximate positions are out of range; unknown positions keep the pair."""
    if positions is None or not np.isfinite(limit):
        return candidate_pairs
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or len(positions) == 0:
        return candidate_pairs
    kept = []
    for i, j in candidate_pairs:
        a, b = positions[i, :2], positions[j, :2]
        if (
            not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
            or float(np.hypot(*(a - b))) <= limit
        ):
            kept.append((i, j))
    if len(kept) < len(candidate_pairs):
        log(
            f"  {len(candidate_pairs) - len(kept):,} of {len(candidate_pairs):,} pairs skipped: "
            f"more than {limit:.0f} m apart"
        )
    return kept


def _screen(candidate_pairs, scans, cfg: CoregConfig, workers: int):
    """Coarse-match every candidate pair and split them into kept matches and rejected results."""
    matched = _parallel_map(
        lambda ij: (
            ij[0],
            ij[1],
            match_stem_maps(scans[ij[0]].stem_map, scans[ij[1]].stem_map, cfg.matching),
        ),
        candidate_pairs,
        workers,
    )
    kept, rejected = [], []
    for i, j, match in matched:
        probe = PairResult(i=i, j=j, name_i=scans[i].name, name_j=scans[j].name, match=match)
        _attach_matched_stems(probe, scans[i], scans[j], match)
        # Stems screen out a pair, but not its targets: those are tried in ICP.
        targets = cfg.use_reflectors and min(
            len(scans[i].reflectors), len(scans[j].reflectors)
        ) >= max(cfg.min_reflector_matches, 3)
        if _coarse_is_acceptable(probe, match, cfg) or targets:
            kept.append(((i, j), match))
        else:
            rejected.append(probe)
    if cfg.max_pairs_per_scan:
        kept = _limit_per_scan(kept, cfg.max_pairs_per_scan, len(scans))
    return kept, rejected


def _limit_per_scan(kept, limit: int, n_scans: int):
    """Each scan's strongest matches; a pair stays if either end has room."""
    order = sorted(range(len(kept)), key=lambda k: -kept[k][1].n_inliers)
    budget = [limit] * n_scans
    chosen = []
    for k in order:
        (i, j), _ = kept[k]
        if budget[i] > 0 or budget[j] > 0:
            chosen.append(k)
            budget[i] -= 1
            budget[j] -= 1
    return [kept[k] for k in sorted(chosen)]


def _recover_unregistered(
    scans, graph: PoseGraph, results, edge_to_pair, registered, cfg: CoregConfig, log, priors
) -> int:
    """Retry failed scans against the combined registered survey.

    Each scan's stems are matched against every registered stem at once, then
    refined against the merged points of the nearest registered scans; on
    success it is tied into the graph by an edge to each of them, so the
    global solve still decides its pose. With priors, a scan whose stems do
    not place it is placed from its prior instead.
    """
    recovered = 0
    for _ in range(max(cfg.recovery_rounds, 1)):
        pending = [k for k, ok in enumerate(registered) if not ok and scans[k].usable]
        if not pending:
            break
        combined = _combined_stem_map(scans, registered, graph)
        if len(combined) < cfg.min_match_inliers and priors is None:
            break
        log(
            f"Retrying {len(pending)} unregistered scan(s) against the "
            f"{len(combined)}-stem combined survey ..."
        )
        gained = 0
        for k in pending:
            placed = _place_against_survey(scans, k, combined, registered, graph, cfg)
            how = "recovered"
            if placed is not None and priors is not None and priors[k] is not None:
                good, why = _prior_ok(placed[0], priors[k], scans[k].origin, cfg)
                if not good:
                    log(f"  {scans[k].name:<24s} placement refused by the prior: {why}")
                    placed = None
            if placed is None and priors is not None and priors[k] is not None:
                reg = [m for m, ok in enumerate(registered) if ok]
                r, used = place_from_prior(
                    scans[k], [scans[m] for m in reg], [graph.poses[m] for m in reg], priors[k], cfg
                )
                if r.success:
                    placed, how = (
                        (r.transform, [reg[u] for u in used], r.icp),
                        "placed from its prior",
                    )
                else:
                    log(f"  {scans[k].name:<24s} not placed from its prior ({r.reason})")
            if placed is None:
                log(f"  {scans[k].name:<24s} still unplaced")
                continue
            world_from_scan, neighbours, refined = placed
            graph.poses[k] = world_from_scan
            for m in neighbours:
                relative = invert(graph.poses[m]) @ world_from_scan
                results.append(
                    PairResult(
                        i=k,
                        j=m,
                        name_i=scans[k].name,
                        name_j=scans[m].name,
                        transform=relative,
                        icp=refined,
                        success=True,
                        reason=f"{how} against {len(neighbours)} combined scans, "
                        f"fitness {refined.fitness:.3f}",
                    )
                )
                edge_to_pair.append(len(results) - 1)
                # The measured quality: a recovered scan is the marginal case.
                graph.add_edge(
                    k,
                    m,
                    relative,
                    fitness=refined.fitness,
                    rmse=refined.inlier_rmse,
                    n_correspondences=max(refined.n_correspondences // len(neighbours), 1),
                    label="recovered",
                )
            registered[k] = True
            gained += 1
            log(f"  {scans[k].name:<24s} {how} against {len(neighbours)} registered scan(s)")
        recovered += gained
        if gained == 0:
            break
    return recovered


def _combined_stem_map(scans, registered, graph: PoseGraph) -> StemMap:
    """Every registered scan's stems in the world frame, one per tree (best
    quality first): duplicates would wreck the one-to-one matcher."""
    stems = []
    for k, ok in enumerate(registered):
        if ok and len(scans[k].stem_map):
            stems.extend(scans[k].stem_map.transformed(graph.poses[k]).stems)
    if not stems:
        return StemMap([], name="combined")
    stems.sort(key=lambda s: -s.quality)
    keep, kept_xy = [], []
    for stem in stems:
        if kept_xy and KdTree(np.array(kept_xy)).query(np.array([[stem.x, stem.y]]))[0][0] < 0.3:
            continue
        keep.append(stem)
        kept_xy.append([stem.x, stem.y])
    return StemMap(keep, name="combined")


def _place_against_survey(
    scans, k, combined: StemMap, registered, graph: PoseGraph, cfg: CoregConfig
):
    """A world pose for one scan from the combined survey, or None."""
    scan = scans[k]
    if len(scan.stem_map) < 3 or len(combined) < cfg.min_match_inliers:
        return None
    match = match_stem_maps(scan.stem_map, combined, cfg.matching)
    if not match.success or match.n_inliers < cfg.min_match_inliers:
        return None
    if match.ambiguity > cfg.max_match_ambiguity:
        return None  # a lattice: nothing to gain by guessing
    here = match.transform[:3, 3]
    others = [m for m, ok in enumerate(registered) if ok and len(scans[m].icp_points)]
    if not others:
        return None
    others.sort(key=lambda m: float(np.linalg.norm(graph.poses[m][:3, 3] - here)))
    neighbours = others[: max(cfg.recovery_neighbours, 1)]
    target = np.vstack(
        [
            transform_points(graph.poses[m], scans[m].icp_points.astype(np.float64))
            for m in neighbours
        ]
    )
    refined = icp(scan.icp_points, target, match.transform, cfg.icp)
    if refined.fitness < cfg.min_icp_fitness or refined.inlier_rmse > cfg.max_icp_rmse:
        return None
    if (
        _above_ground_fitness(scan.icp_points, scan.icp_heights, target, refined.transform, cfg)
        < cfg.min_icp_fitness_above_ground
    ):
        return None
    return refined.transform, neighbours, refined


def _reference_in_largest_component(graph: PoseGraph, reference: int, n: int) -> int:
    """Keep ``reference`` unless it is isolated, then move it into the largest block."""
    components = sorted((set(c) for c in graph.components()), key=lambda c: (-len(c), min(c)))
    if not components or len(components[0]) < 2:
        return reference
    mine = next(c for c in components if reference in c)
    return reference if len(mine) > 1 else min(components[0])


def _registered_mask(graph: PoseGraph, n: int) -> list[bool]:
    """Scans connected to an anchor by accepted edges (rejected ones included, as tlsalign)."""
    adjacency: dict[int, list[int]] = {k: [] for k in range(n)}
    for e in graph.edges:
        adjacency[e.i].append(e.j)
        adjacency[e.j].append(e.i)
    anchors = {graph.reference, *graph.fixed}
    seen, stack = set(anchors), list(anchors)
    while stack:
        for nb in adjacency[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return [k in seen for k in range(n)]


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #


def _prior_ok(
    pose: np.ndarray, prior: np.ndarray, origin: np.ndarray, cfg: CoregConfig
) -> tuple[bool, str]:
    """Does ``pose`` put the scanner where its prior says it stood?"""
    shift = float(
        np.linalg.norm(transform_points(pose, origin[None]) - transform_points(prior, origin[None]))
    )
    if shift > cfg.max_prior_shift:
        return False, f"scanner {shift:.1f} m from its prior position"
    if cfg.max_prior_rotation is not None:
        rot = float(np.degrees(np.linalg.norm(se3_log(invert(prior) @ pose)[:3])))
        if rot > cfg.max_prior_rotation:
            return False, f"{rot:.1f} deg from the prior orientation"
    return True, ""


def _lowest_per_cell(xyz: np.ndarray, cell: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    key = np.floor(xyz[:, :2] / cell).astype(np.int64)
    k = key[:, 0] * 10_000_000 + key[:, 1]
    order = np.lexsort((xyz[:, 2], k))
    first = np.r_[True, k[order][1:] != k[order][:-1]]
    return k[order][first], xyz[order][first, 2]


def _ground_offset(points: np.ndarray, target: np.ndarray) -> float:
    """Median height of ``target``'s ground over ``points``' ground, per cell.

    NaN if they share too little ground.
    """
    ka, za = _lowest_per_cell(points)
    kb, zb = _lowest_per_cell(target)
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    return float(np.median(zb[ib] - za[ia])) if len(common) >= 50 else float("nan")


def place_from_prior(
    scan: ScanFeatures,
    survey: Sequence[ScanFeatures],
    poses: Sequence[np.ndarray],
    prior: np.ndarray,
    config: CoregConfig | None = None,
    neighbours: int | None = None,
) -> tuple[PairResult, list[int]]:
    """Place one scan from an approximate pose instead of from stems.

    For scans that see too few stems to match. The prior (a RiSCAN SOP, GNSS
    and compass) is first corrected in height by the median offset between
    the scan's ground and its registered neighbours' ground, since a vertical
    error of metres is common and beyond ICP's reach; ICP then starts with a
    1.5 m correspondence distance. Accepted on ICP fitness, above-ground
    fitness and RMSE, and only within ``max_prior_shift`` of the prior.

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
    config
        Pipeline settings.
    neighbours
        Registered scans nearest the prior that ICP runs against
        (``recovery_neighbours`` if None).

    Returns
    -------
    result : PairResult
        ``transform`` is ``world_from_scan``; check ``success``.
    used : list of int
        Indices into ``survey`` of the scans ICP ran against.
    """
    cfg = config or CoregConfig()
    start = time.perf_counter()
    result = PairResult(-1, -1, name_i=scan.name, name_j="survey")
    here = scan.location(prior)
    order = sorted(
        range(len(survey)), key=lambda m: float(np.linalg.norm(survey[m].location(poses[m]) - here))
    )
    used = [m for m in order if len(survey[m].icp_points)][: neighbours or cfg.recovery_neighbours]
    if not used:
        result.reason = "no registered scans to place against"
        return result, []
    target = np.vstack(
        [transform_points(poses[m], survey[m].icp_points.astype(np.float64)) for m in used]
    )
    coarse = np.asarray(prior, float).copy()
    dz = _ground_offset(transform_points(coarse, scan.icp_points), target)
    if not np.isfinite(dz):
        result.reason = "no ground shared with the registered scans"
        return result, []
    coarse[2, 3] += dz
    wide = replace(
        cfg.icp,
        voxel_sizes=(0.30, 0.30, 0.15, 0.07, 0.05),
        max_distances=(1.50, 0.80, 0.40, 0.20, 0.12),
    )
    refined = icp(scan.icp_points, target, coarse, wide)
    result.icp, result.transform, result.coarse_transform = refined, refined.transform, coarse
    result.fitness_above = _above_ground_fitness(
        scan.icp_points, scan.icp_heights, target, refined.transform, cfg
    )
    good, why = _prior_ok(refined.transform, prior, scan.origin, cfg)
    if refined.fitness < cfg.min_icp_fitness:
        result.reason = f"low ICP fitness ({refined.fitness:.3f} < {cfg.min_icp_fitness})"
    elif result.fitness_above < cfg.min_icp_fitness_above_ground:
        result.reason = (
            f"low above-ground fitness ({result.fitness_above:.3f}); ground alone matched"
        )
    elif refined.inlier_rmse > cfg.max_icp_rmse:
        result.reason = f"high ICP rmse ({refined.inlier_rmse:.3f} m)"
    elif not good:
        result.reason = why
    else:
        result.success = True
        result.reason = f"from the prior, height corrected by {dz:+.2f} m"
    result.seconds = time.perf_counter() - start
    return result, used


# --------------------------------------------------------------------------- #
# Applying the result
# --------------------------------------------------------------------------- #


def merge_clouds(
    clouds: Sequence,
    result: SurveyResult,
    *,
    voxel: float | None = 0.02,
    only_registered: bool = True,
    riegl_options: dict | None = None,
    riscan_filter: str = "none",
):
    """Every scan moved into the reference frame and concatenated.

    Parameters
    ----------
    clouds
        The scans as passed to :func:`coregister` (paths, clouds or arrays).
    result
        The registration.
    voxel
        Thin each scan, and the merged cloud, to one point per voxel (m); the
        overlap otherwise carries several copies of every surface.
    only_registered
        Leave out unregistered scans.
    riegl_options
        RXP reading options (:attr:`CoregConfig.riegl_options`).
    riscan_filter
        RiSCAN import filter applied when reading (:attr:`CoregConfig.riscan_filter`).

    Returns
    -------
    sylva.PointCloud
        With a ``scan_id`` attribute.
    """
    from .. import filters
    from ..pointcloud import PointCloud

    cfg = CoregConfig(riegl_options=dict(riegl_options or {}), riscan_filter=riscan_filter)
    parts, ids = [], []
    for k, cloud in enumerate(clouds):
        if only_registered and not result.registered[k]:
            continue
        if isinstance(cloud, (str, Path)):
            xyz, _ = _read_scan(Path(cloud), cfg)
        elif isinstance(cloud, PointCloud):
            xyz = cloud.xyz
        else:
            xyz = np.asarray(cloud, dtype=float).reshape(-1, 3)
        moved = transform_points(result.transform_for(k), xyz)
        if voxel:
            moved = voxel_downsample(moved, voxel, centroid=False)
        parts.append(moved)
        ids.append(np.full(len(moved), k, np.int32))
    merged = PointCloud(
        np.vstack(parts) if parts else np.zeros((0, 3)),
        {"scan_id": np.concatenate(ids) if ids else np.zeros(0, np.int32)},
    )
    return filters.voxel_downsample(merged, voxel) if voxel else merged
