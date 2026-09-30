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
   saw them, otherwise a global stem-map match with its height taken from the
   two terrain models, gives a coarse transform that ICP refines; the pair is
   accepted only if it fits over all points and above the ground, and its
   terrain agrees.
3. **Whole survey** (:func:`coregister_prepared`): accepted pairs become the
   edges of a pose graph, each weighted by the directions its surfaces
   constrain, solved with outlier rejection; scans left over are retried
   against the combined registered survey, and optionally every pose is
   refined jointly.

Every stage records its own quality, because the useful question is not "did
it run" but "which scans can I trust". Stem matches take
their height from the shared ground rather than from stem bases (vertical
error dominates stem-based registration, as Tremblay & Béland 2018 and
GlobalMatch, Wang et al. 2023, both found); edges carry the anisotropic
information of their point-to-plane correspondences instead of an isotropic
weight; outlier edges are rejected even while some scans are unregistered;
and recovered scans are tied in by pairwise measurements. Scans can be held
fixed at trusted poses, so that new scans are registered into an existing
project; and approximate poses (a RiSCAN SOP, GNSS and compass) can serve as
priors, refusing results that move a scanner implausibly far and placing
scans that see too few stems.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .. import _core
from .ground import GroundModel
from .icp import ICPConfig, ICPResult, ICPTarget, _plane_information
from .matching import MatchConfig, MatchResult
from .posegraph import OptimisationResult
from .reflectors import Reflector, ReflectorMatch
from .stems import Stem, StemDetectionConfig, StemMap, _detector_kwargs
from .transforms import _mat4, identity, transform_points

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
    """Settings of the whole pipeline."""

    ground_cell_size: float = 0.5
    ground_min_coverage: float | None = 0.8
    """Least fraction of the elevations from -30 to +5 degrees an azimuth of
    the scan must sample for the terrain in that direction to be fitted from
    it; see :func:`_refit_visible_ground`. Tilted scans sample almost none in
    the azimuths along their tilt axis, upright scans all of them. Needs the
    scanner's position, so it applies to scans read from file or given
    ``origin``. None fits the terrain from every return."""
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
    targets' own. 0 always requires ICP to agree."""
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
    """Optional cap (m) on the median horizontal disagreement of matched
    stems; off by default."""

    height_from_ground: bool = True
    """Take a stem match's vertical offset from the two terrain models where
    both saw ground, not from the matched stems. A stem's height is its own
    scan's terrain height under it plus 1.3 m, and under understory or on a
    slope that terrain is least certain exactly at the stems, while the
    shared ground between two scanners is wide and well sampled."""
    ground_radius: float = 30.0
    """Terrain cells further than this (m) from a scanner are left out of
    height comparisons: far from the scanner, ground returns are sparse and
    grazing, and the terrain model there is mostly the understory."""
    min_ground_cells: int = 50
    """Shared observed cells a height comparison needs."""
    max_ground_disagreement: float = 0.25
    """A pair whose terrain models still differ by more than this (m, median
    over the shared ground) after ICP is refused: ICP slid vertically, or
    the match is wrong. None disables the test."""

    max_pair_distance: float = 40.0
    """Pairs whose approximate positions are further apart (m) are not tried.
    Scale it with the range: with a 30 m gate no pair more than 35 m apart
    ever registered (0 of 1687)."""
    screen_pairs: bool = True
    """Match every pair first and run ICP only on the survivors."""
    max_pairs_per_scan: int | None = None
    """After screening keep only each scan's best N pairs; None keeps all."""

    min_icp_fitness: float = 0.04
    """Fitness decreases with the distance between scans, since less of each
    scan is shared. On a VZ-400 survey with positions 30-35 m apart, scored
    against its reflector-based registration, every pair ICP placed with a
    fitness of 0.045-0.10 was within 10 cm of the reference, and a gate of
    0.10 left 10 of its 14 scans unregistered."""
    max_icp_rmse: float = 0.15
    min_icp_fitness_above_ground: float = 0.03
    """Fitness over source points above :attr:`fitness_min_height`. Ground is
    planar and fits under any horizontal shift, so on open sites it can carry
    a wrong pair past :attr:`min_icp_fitness` alone; stems, branches and logs
    are what fix the horizontal position. On the same survey, correct pairs
    scored at least 0.043 and wrong ones at most 0.012."""
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
    information_patch_points: float = 100.0
    """ICP correspondences counted as one independent observation when
    weighting pose-graph edges (:func:`~sylva.coreg.plane_edge_information`):
    neighbouring residuals share the error of the surface they sample."""
    information_min_sigma: float = 0.005
    """Floor (m) on the point-to-plane residual that weights an edge."""

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
    default."""
    min_points_per_scan: int = 1000
    """Scans with fewer points are set aside rather than processed."""
    max_points_per_scan: int | None = None
    verbose: bool = True

    def to_dict(self) -> dict:
        """The whole configuration as plain data."""
        return asdict(self)


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
        False when the coarse transform was kept over ICP's; ``icp`` then
        scores the coarse transform, not the refinement that was discarded.
    ground_offset
        Median height (m) of the target's terrain over the source's where
        both saw ground, under :attr:`transform`; NaN if they share too little.
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
    ground_offset: float = float("nan")
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
        return _core.coreg_pair_summary(_pair_core(self))


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
        """Horizontal stem disagreement per accepted pair under the final poses (m).

        The field-usable quality check: no ground truth needed, it measures
        whether the same tree lands in the same place from two scans. Only
        the horizontal distance counts: a stem's height comes from its own
        scan's terrain model, which the pair's ground check covers.

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
        values = _core.coreg_survey_consistency(
            [_pair_core(p) for p in self.pairs], _poses(self.poses), bool(robust)
        )
        return {(i, j): v for i, j, v in values}

    def report(self) -> str:
        """Plain-text summary for a log or a QC record."""
        opt = self.optimisation
        return _core.coreg_survey_report(
            self._scan_summaries(),
            [_pair_core(p) for p in self.pairs],
            _poses(self.poses),
            int(self.reference),
            None
            if opt is None
            else (
                int(opt.iterations),
                bool(opt.converged),
                float(opt.initial_error),
                float(opt.final_error),
                [int(k) for k in opt.rejected_edges],
            ),
            [bool(r) for r in self.registered],
            float(self.seconds),
            [int(k) for k in self.edge_to_pair],
        )

    def save(self, path: str | Path) -> Path:
        """Write the transforms and quality as JSON (``transforms.json``).

        Returns
        -------
        pathlib.Path
        """
        path = Path(path)
        _core.coreg_survey_save(
            path,
            self._scan_summaries(),
            [_pair_core(p) for p in self.pairs],
            _poses(self.poses),
            int(self.reference),
            [bool(r) for r in self.registered],
            float(self.seconds),
        )
        return path

    def _scan_summaries(self) -> list[tuple]:
        return [
            (
                s.name,
                int(s.n_points),
                len(s.stem_map),
                s.error,
                str(s.source) if s.source else None,
                _mat4(s.levelling),
            )
            for s in self.scans
        ]


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
    return {name: np.asarray(rows) for name, rows in _core.coreg_load_transforms(Path(path))}


# --------------------------------------------------------------------------- #
# Crossing into the core
# --------------------------------------------------------------------------- #


def _config_core(cfg: CoregConfig) -> dict:
    """The settings as the core takes them: stem detection in the detector's
    coregistration mode, the rest field for field."""
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    d["stems"] = _detector_kwargs(cfg.stems)
    d["matching"] = asdict(cfg.matching)
    icp = asdict(cfg.icp)
    icp["voxel_sizes"] = [float(v) for v in cfg.icp.voxel_sizes]
    if cfg.icp.max_distances is not None:
        icp["max_distances"] = [float(v) for v in cfg.icp.max_distances]
    d["icp"] = icp
    d["refinement_voxel_sizes"] = [float(v) for v in cfg.refinement_voxel_sizes]
    d["refinement_max_distances"] = [float(v) for v in cfg.refinement_max_distances]
    d["riegl_options"] = dict(cfg.riegl_options)
    return d


def _stem_rows(stem_map: StemMap) -> np.ndarray:
    return np.array(
        [
            [
                s.x,
                s.y,
                s.z,
                s.dbh,
                *np.asarray(s.axis, dtype=float).reshape(3),
                s.reference_height,
                s.n_slices,
                s.n_points,
                s.rmse,
                s.coverage,
                s.lean_deg,
            ]
            for s in stem_map
        ],
        dtype=np.float64,
    ).reshape(-1, 13)


def _stems_from_rows(rows: np.ndarray) -> list[Stem]:
    return [
        Stem(
            float(r[0]),
            float(r[1]),
            float(r[2]),
            float(r[3]),
            np.array(r[4:7]),
            float(r[7]),
            int(r[8]),
            int(r[9]),
            float(r[10]),
            float(r[11]),
            float(r[12]),
        )
        for r in rows
    ]


def _reflector_tuples(reflectors) -> list[tuple]:
    return [
        (
            float(r.x),
            float(r.y),
            float(r.z),
            float(r.reflectance),
            float(r.diameter),
            int(r.n_points),
            str(r.name),
        )
        for r in reflectors or []
    ]


def _scan_core(scan: ScanFeatures) -> tuple:
    g = scan.ground
    return (
        scan.name,
        int(scan.n_points),
        None
        if g is None
        else (
            np.ascontiguousarray(g.elevation, dtype=np.float64),
            (float(g.origin[0]), float(g.origin[1])),
            float(g.cell_size),
            np.ascontiguousarray(g.observed, dtype=bool),
        ),
        _stem_rows(scan.stem_map),
        scan.stem_map.name,
        np.ascontiguousarray(np.asarray(scan.icp_points, dtype=np.float64).reshape(-1, 3)),
        _reflector_tuples(scan.reflectors),
        np.ascontiguousarray(np.asarray(scan.icp_heights, dtype=np.float32).reshape(-1)),
        _mat4(scan.levelling),
        [float(v) for v in np.asarray(scan.origin, dtype=float).reshape(3)],
        None if scan.source is None else str(scan.source),
        float(scan.seconds),
        scan.error,
    )


def _scan_from_core(d: dict) -> ScanFeatures:
    ground = None if d["ground"] is None else GroundModel(*d["ground"])
    return ScanFeatures(
        name=d["name"],
        n_points=d["n_points"],
        ground=ground,
        stem_map=StemMap(_stems_from_rows(d["stems"]), name=d["stem_map_name"], ground=ground),
        icp_points=d["icp_points"],
        reflectors=[Reflector(*r) for r in d["reflectors"]],
        icp_heights=d["icp_heights"],
        levelling=d["levelling"],
        origin=d["origin"],
        source=None if d["source"] is None else Path(d["source"]),
        seconds=d["seconds"],
        error=d["error"],
    )


def _match_core(m: MatchResult | None) -> dict | None:
    if m is None:
        return None
    return {
        "transform": _mat4(m.transform),
        "n_inliers": int(m.n_inliers),
        "inlier_rmse": float(m.inlier_rmse),
        "score": float(m.score),
        "correspondences": np.ascontiguousarray(
            np.asarray(m.correspondences, dtype=np.int64).reshape(-1, 2)
        ),
        "n_source": int(m.n_source),
        "n_target": int(m.n_target),
        "success": bool(m.success),
        "ambiguity": float(m.ambiguity),
        "rival": _match_core(m.rival),
    }


def _icp_core(r: ICPResult | None) -> dict | None:
    if r is None:
        return None
    info = r.information
    return {
        "transform": _mat4(r.transform),
        "fitness": float(r.fitness),
        "inlier_rmse": float(r.inlier_rmse),
        "n_correspondences": int(r.n_correspondences),
        "iterations": int(r.iterations),
        "converged": bool(r.converged),
        "history": [float(v) for v in r.history],
        "hessian": None
        if info is None
        else np.ascontiguousarray(np.asarray(info.hessian, dtype=float)),
        "plane_sigma": None if info is None else float(info.sigma),
        "plane_n": None if info is None else int(info.n),
    }


def _icp_from_core(d: dict | None) -> ICPResult | None:
    if d is None:
        return None
    return ICPResult(
        np.asarray(d["transform"]),
        float(d["fitness"]),
        float(d["inlier_rmse"]),
        int(d["n_correspondences"]),
        int(d["iterations"]),
        bool(d["converged"]),
        list(d["history"]),
        _plane_information(d),
    )


def _pair_core(p: PairResult) -> dict:
    r = p.reflector_match
    return {
        "i": int(p.i),
        "j": int(p.j),
        "name_i": p.name_i,
        "name_j": p.name_j,
        "transform": _mat4(p.transform),
        "coarse_transform": _mat4(p.coarse_transform),
        "match": _match_core(p.match),
        "reflector_match": None
        if r is None
        else {
            "transform": _mat4(r.transform),
            "n_inliers": int(r.n_inliers),
            "rmse": float(r.rmse),
            "correspondences": np.ascontiguousarray(
                np.asarray(r.correspondences, dtype=np.int64).reshape(-1, 2)
            ),
            "success": bool(r.success),
        },
        "icp": _icp_core(p.icp),
        "success": bool(p.success),
        "reason": p.reason,
        "seconds": float(p.seconds),
        "matched_source": np.ascontiguousarray(
            np.asarray(p.matched_source, dtype=np.float64).reshape(-1, 3)
        ),
        "matched_target": np.ascontiguousarray(
            np.asarray(p.matched_target, dtype=np.float64).reshape(-1, 3)
        ),
        "coarse_stem_rmse": float(p.coarse_stem_rmse),
        "fine_stem_rmse": float(p.fine_stem_rmse),
        "fitness_above": float(p.fitness_above),
        "rival": _match_core(p.rival),
        "used_icp": bool(p.used_icp),
        "ground_offset": float(p.ground_offset),
        "trusted": bool(p.trusted),
    }


def _pair_from_core(d: dict) -> PairResult:
    r = d["reflector_match"]
    return PairResult(
        i=d["i"],
        j=d["j"],
        name_i=d["name_i"],
        name_j=d["name_j"],
        transform=d["transform"],
        coarse_transform=d["coarse_transform"],
        match=None if d["match"] is None else MatchResult._from_core(d["match"]),
        reflector_match=None
        if r is None
        else ReflectorMatch(
            r["transform"], r["n_inliers"], r["rmse"], r["correspondences"], r["success"]
        ),
        icp=_icp_from_core(d["icp"]),
        success=d["success"],
        reason=d["reason"],
        seconds=d["seconds"],
        matched_source=d["matched_source"],
        matched_target=d["matched_target"],
        coarse_stem_rmse=d["coarse_stem_rmse"],
        fine_stem_rmse=d["fine_stem_rmse"],
        fitness_above=d["fitness_above"],
        rival=None if d["rival"] is None else MatchResult._from_core(d["rival"]),
        used_icp=d["used_icp"],
        ground_offset=d["ground_offset"],
        trusted=d["trusted"],
    )


def _survey_from_core(scans: list[ScanFeatures], d: dict) -> SurveyResult:
    poses = [np.asarray(p) for p in d["poses"]]
    o = d["optimisation"]
    optimisation = None
    if o is not None:
        optimisation = OptimisationResult(
            list(poses),
            int(o["iterations"]),
            bool(o["converged"]),
            float(o["initial_error"]),
            float(o["final_error"]),
            [int(k) for k in o["rejected_edges"]],
            np.asarray(o["edge_errors"]),
        )
    return SurveyResult(
        scans=scans,
        pairs=[_pair_from_core(p) for p in d["pairs"]],
        poses=poses,
        reference=int(d["reference"]),
        optimisation=optimisation,
        registered=[bool(r) for r in d["registered"]],
        seconds=float(d["seconds"]),
        edge_to_pair=[int(k) for k in d["edge_to_pair"]],
    )


def _poses(poses) -> list[np.ndarray]:
    return [_mat4(p) for p in poses]


def _targets(targets) -> tuple[list[tuple], list[np.ndarray]]:
    targets = list(targets)
    return [_scan_core(s) for s, _ in targets], [_mat4(p) for _, p in targets]


def _points(cloud) -> np.ndarray:
    from ..pointcloud import PointCloud

    if isinstance(cloud, PointCloud):
        return np.ascontiguousarray(cloud.xyz, dtype=np.float64)
    return np.ascontiguousarray(np.asarray(cloud, dtype=np.float64).reshape(-1, 3))


def _input_core(cloud):
    """A scan as the core reads it: a path (str) or ``(n, 3)`` points."""
    if isinstance(cloud, (str, Path)):
        return str(Path(cloud))
    return _points(cloud)


def _positions(positions) -> list[list[float]] | None:
    """Approximate positions for the core; None where they would be ignored."""
    if positions is None:
        return None
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or len(positions) == 0:
        return None
    return positions.tolist()


# --------------------------------------------------------------------------- #
# Stage 1: per scan
# --------------------------------------------------------------------------- #


def _make_logger(verbose: bool) -> Callable[[str], None]:
    return functools.partial(print, flush=True) if verbose else (lambda _message: None)


def _log_for(progress, cfg: CoregConfig):
    """The callable the core reports to, or None for silence."""
    if progress is not None:
        return progress
    return _make_logger(True) if cfg.verbose else None


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
        Explicit bounds override the file.

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
    """Points of a scan file in its own frame, filtered for coregistration.

    RiSCAN's import filter (:attr:`CoregConfig.riscan_filter`) is decided on
    the whole stream first; then the closed intervals on range (from the
    scanner), deviation, reflectance and amplitude are applied.
    """
    return _core.coreg_read_scan(Path(path), _config_core(cfg)), {}


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
    cfg = config or CoregConfig()
    d = _core.coreg_prepare_scan(
        _input_core(cloud),
        _config_core(cfg),
        name,
        _reflector_tuples(reflectors),
        None if levelling is None else _mat4(levelling),
        None if origin is None else [float(v) for v in np.asarray(origin, dtype=float).reshape(3)],
    )
    return _scan_from_core(d)


def _refit_visible_ground(
    points: np.ndarray, scanner: np.ndarray, ground: GroundModel, cfg: CoregConfig
) -> GroundModel:
    """Refit the terrain without the directions in which the scanner could not see it.

    A scanner tilted on its side misses a band of near-horizontal directions
    along its tilt axis. In those azimuths the lowest return over distant
    ground is foliage, and the fitted terrain sits metres high. An azimuth
    (1 degree) is blind when the scan sampled less than
    :attr:`CoregConfig.ground_min_coverage` of the elevations from -30 to +5
    degrees in it; blind azimuths, widened by 3 degrees either side, keep only
    their returns more than 45 degrees below the horizon (the ground around
    the tripod, which a tilted scanner sees), and the terrain beyond is
    filled from the azimuths that saw it. On a VZ-400 survey of upright and tilted
    scans, this cut the terrain disagreement between scans (95th percentile
    over pairs, under the reflector-based registration) from 2.7 m to 0.6 m.
    Applied only when the scanner's position is known: a scan read from
    file, or one given ``origin``.
    """
    refitted = _core.coreg_refit_visible_ground(
        _points(points),
        [float(v) for v in np.asarray(scanner, dtype=float).reshape(3)],
        _config_core(cfg),
    )
    return ground if refitted is None else GroundModel(*refitted)


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
    d = _core.coreg_register_pair(
        _scan_core(source),
        _scan_core(target),
        _config_core(cfg),
        initial=None if initial is None else _mat4(initial),
        stem_match=_match_core(match),
        i=int(i),
        j=int(j),
        target_icp=None if target_icp is None else target_icp._core,
    )
    return _pair_from_core(d)


def _trust_reflectors(
    result: PairResult, coarse: np.ndarray, shift: float, cfg: CoregConfig
) -> None:
    """Accept a pair ICP refused if its reflector match is strong enough alone."""
    d = _core.coreg_trust_reflectors(
        _pair_core(result), _mat4(coarse), float(shift), _config_core(cfg)
    )
    if d["success"] != result.success or d["trusted"] != result.trusted:
        result.transform = d["transform"]
        result.used_icp = d["used_icp"]
        result.success = d["success"]
        result.trusted = d["trusted"]
        result.reason = d["reason"]


def _terrain_samples(scan: ScanFeatures, radius: float) -> np.ndarray:
    """``(n, 3)`` observed terrain cells within ``radius`` of the scanner, in the scan's frame."""
    return _core.coreg_terrain_samples(_scan_core(scan), float(radius))


def _height_offset(
    source: ScanFeatures,
    world_from_source: np.ndarray,
    targets: Sequence[tuple[ScanFeatures, np.ndarray]],
    cfg: CoregConfig,
) -> float:
    """Median height of the targets' terrain over the source's where both saw ground.

    Parameters
    ----------
    source
        The scan being placed.
    world_from_source
        Its pose (or the pair transform, with the target as the world).
    targets
        ``(scan, world_from_scan)`` of the scans compared against.

    Returns
    -------
    float
        Metres to add to the source's height; NaN if fewer than
        ``min_ground_cells`` shared cells.
    """
    scans, poses = _targets(targets)
    return float(
        _core.coreg_height_offset(
            _scan_core(source), _mat4(world_from_source), scans, poses, _config_core(cfg)
        )
    )


def _on_ground(
    transform: np.ndarray,
    source: ScanFeatures,
    targets: Sequence[tuple[ScanFeatures, np.ndarray]],
    cfg: CoregConfig,
) -> np.ndarray:
    """``transform`` with its height set from the terrain
    (:attr:`CoregConfig.height_from_ground`)."""
    if not cfg.height_from_ground:
        return transform
    scans, poses = _targets(targets)
    return _core.coreg_on_ground(
        _mat4(transform), _scan_core(source), scans, poses, _config_core(cfg)
    )


def _stem_median_residual(transform: np.ndarray, pair: PairResult) -> float:
    """Median horizontal distance between matched stems: their heights come
    from each scan's own terrain model, which says nothing about the match."""
    return float(
        _core.coreg_stem_median_residual(
            _mat4(transform),
            np.ascontiguousarray(np.asarray(pair.matched_source, dtype=np.float64).reshape(-1, 3)),
            np.ascontiguousarray(np.asarray(pair.matched_target, dtype=np.float64).reshape(-1, 3)),
        )
    )


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
    n = len(clouds)
    targets = list(reflectors) if reflectors else [[]] * n
    levels = list(levelling) if levelling else [None] * n
    inputs, target_tuples, level_mats = [], [], []
    for k in range(n):
        # A scan that cannot even be converted costs that scan, not the run.
        try:
            inputs.append(_input_core(clouds[k]))
            target_tuples.append(_reflector_tuples(targets[k]))
            level_mats.append(None if levels[k] is None else _mat4(levels[k]))
        except Exception as exc:
            src = str(Path(clouds[k])) if isinstance(clouds[k], (str, Path)) else None
            inputs.append(("failed", f"{type(exc).__name__}: {exc}", src))
            target_tuples.append([])
            level_mats.append(None)
    scan_dicts, d = _core.coreg_coregister(
        inputs,
        _config_core(cfg),
        [str(v) for v in names] if names else [],
        target_tuples,
        level_mats,
        **_survey_options(pairs, approximate_positions, fixed, priors),
        log=_log_for(progress, cfg),
    )
    return _survey_from_core([_scan_from_core(s) for s in scan_dicts], d)


def _survey_options(pairs, approximate_positions, fixed, priors) -> dict:
    return {
        "pairs": None if pairs is None else [(int(i), int(j)) for i, j in pairs],
        "positions": _positions(approximate_positions),
        "fixed": [(int(k), _mat4(v)) for k, v in (fixed or {}).items()],
        "priors": None if priors is None else [None if p is None else _mat4(p) for p in priors],
    }


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
    already = 0.0 if started is None else time.perf_counter() - started
    d = _core.coreg_coregister_prepared(
        [_scan_core(s) for s in scans],
        _config_core(cfg),
        **_survey_options(pairs, approximate_positions, fixed, priors),
        log=_log_for(progress, cfg),
        already=already,
    )
    return _survey_from_core(scans, d)


def _within_reach(candidate_pairs, positions, limit, log):
    """Drop pairs whose approximate positions are out of range; unknown positions keep the pair."""
    if positions is None or not np.isfinite(limit):
        return candidate_pairs
    rows = _positions(positions)
    if rows is None:
        return candidate_pairs
    kept, messages = _core.coreg_within_reach(
        [(int(i), int(j)) for i, j in candidate_pairs], rows, float(limit)
    )
    for message in messages:
        log(message)
    return kept


def _limit_per_scan(kept, limit: int, n_scans: int):
    """Each scan's strongest matches; a pair stays if either end has room."""
    chosen = _core.coreg_limit_per_scan(
        [(int(i), int(j), int(m.n_inliers)) for (i, j), m in kept], int(limit), int(n_scans)
    )
    return [kept[k] for k in chosen]


def _combined_stem_map(scans, registered, graph) -> StemMap:
    """Every registered scan's stems in the world frame, one per tree (best
    quality first): duplicates would wreck the one-to-one matcher."""
    rows = _core.coreg_combined_stem_map(
        [_scan_core(s) for s in scans], [bool(r) for r in registered], _poses(graph.poses)
    )
    return StemMap(_stems_from_rows(rows), name="combined")


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #


def _prior_ok(
    pose: np.ndarray, prior: np.ndarray, origin: np.ndarray, cfg: CoregConfig
) -> tuple[bool, str]:
    """Does ``pose`` put the scanner where its prior says it stood?"""
    good, why = _core.coreg_prior_ok(
        _mat4(pose),
        _mat4(prior),
        [float(v) for v in np.asarray(origin, dtype=float).reshape(3)],
        _config_core(cfg),
    )
    return bool(good), why


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
    d, used = _core.coreg_place_from_prior(
        _scan_core(scan),
        [_scan_core(s) for s in survey],
        _poses(poses),
        _mat4(prior),
        _config_core(cfg),
        None if neighbours is None else int(neighbours),
    )
    return _pair_from_core(d), list(used)


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
    from ..pointcloud import PointCloud

    cfg = CoregConfig(riegl_options=dict(riegl_options or {}), riscan_filter=riscan_filter)
    xyz, ids = _core.coreg_merge_clouds(
        [_input_core(c) for c in clouds],
        _poses(result.poses),
        [_mat4(s.levelling) for s in result.scans],
        [bool(r) for r in result.registered],
        bool(only_registered),
        None if not voxel else float(voxel),
        _config_core(cfg),
    )
    return PointCloud(xyz, {"scan_id": ids})
