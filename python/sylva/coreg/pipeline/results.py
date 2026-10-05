# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The settings and results of the coregistration pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ... import _core
from ..ground import GroundModel
from ..icp import ICPConfig, ICPResult
from ..matching import MatchConfig, MatchResult
from ..posegraph import OptimisationResult
from ..reflectors import Reflector
from ..stems import StemDetectionConfig, StemMap
from ..transforms import _mat4, identity, transform_points


# ``convert`` builds these classes, so it is imported when first needed.
def _pair_core(pair):
    from .convert import _pair_core as convert

    return convert(pair)


def _poses(poses):
    from .convert import _poses as convert

    return convert(poses)


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
