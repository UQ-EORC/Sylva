# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Global coarse registration by stem-map matching.

Aligning two stem maps is finding the rigid motion that best superimposes two
partly overlapping 2-D point patterns whose points carry a diameter. It is
searched exhaustively over pairs of stems, not sampled: the distance between
two stems does not depend on the unknown transform, so a pair in one scan can
only correspond to a pair at the same separation in the other, and a sorted
pair table makes that a binary search. Two correspondences give yaw and
translation, the right model for levelled scans; ICP recovers any residual
tilt. The search runs in the Rust core.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .. import _core
from .stems import StemMap

__all__ = ["MatchConfig", "MatchResult", "match_stem_maps"]


@dataclass
class MatchConfig:
    """Settings of :func:`match_stem_maps`; the defaults are tlsalign's."""

    min_pair_distance: float = 2.0
    """Stem pairs closer than this (m) are ambiguous and ignored."""
    max_pair_distance: float = 35.0
    """Stem pairs further apart than this (m) are rarely seen together."""
    pair_distance_tolerance: float = 0.25
    inlier_tolerance: float = 0.40
    """Horizontal distance (m) at which two stems count as the same tree."""
    diameter_rel_tolerance: float = 0.30
    diameter_abs_tolerance: float = 0.04
    use_diameters: bool = True
    max_stems: int = 70
    """The best stems of each scan taken into matching."""
    max_hypotheses: int = 60_000
    min_inliers: int = 4
    early_exit_inliers: int = 40
    """Stop once a hypothesis has this many inliers. High, so that the search
    normally runs its budget: in a planted lattice the first convincing
    hypothesis is often a shifted copy, and only a fuller search shows the
    true one and how close the two are."""
    distinct_translation: float = 1.0
    distinct_yaw_deg: float = 5.0
    """A rival must differ by ``distinct_translation`` m or this much yaw."""
    refine_iterations: int = 6
    seed: int = 0


@dataclass
class MatchResult:
    """Outcome of :func:`match_stem_maps`; ``transform`` maps source into target.

    Attributes
    ----------
    correspondences
        ``(n, 2)`` indices ``(source, target)`` of matched stems, into the
        stem maps as given.
    success
        At least ``min_inliers`` stems matched consistently. Check it first.
    ambiguity
        Inliers of the best distinctly different alignment over the best's.
        Near 1 the pattern matches itself elsewhere (a planted lattice) and
        the winner is a coin toss.
    rival
        That alternative, refined, for ICP to decide between.
    """

    transform: np.ndarray
    n_inliers: int
    inlier_rmse: float
    score: float
    correspondences: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    n_source: int = 0
    n_target: int = 0
    success: bool = False
    ambiguity: float = 0.0
    rival: MatchResult | None = field(default=None, repr=False)

    @property
    def inlier_fraction(self) -> float:
        """Inliers over the smaller stem map."""
        return self.n_inliers / max(min(self.n_source, self.n_target), 1)

    @classmethod
    def _from_core(cls, d: dict) -> MatchResult:
        return cls(
            np.asarray(d["transform"]),
            int(d["n_inliers"]),
            float(d["inlier_rmse"]),
            float(d["score"]),
            np.asarray(d["correspondences"], dtype=int).reshape(-1, 2),
            int(d["n_source"]),
            int(d["n_target"]),
            bool(d["success"]),
            float(d["ambiguity"]),
            cls._from_core(d["rival"]) if d["rival"] is not None else None,
        )


def match_stem_maps(
    source: StemMap, target: StemMap, config: MatchConfig | None = None
) -> MatchResult:
    """Yaw and translation aligning ``source`` stems onto ``target`` stems, with no initial guess.

    Parameters
    ----------
    source, target
        Stem maps.
    config
        Matcher settings.

    Returns
    -------
    MatchResult
        Check ``success`` before trusting ``transform``.
    """
    cfg = config or MatchConfig()
    kw = asdict(cfg)
    kw.pop("seed")
    d = _core.match_stem_maps(
        np.ascontiguousarray(source.positions),
        np.ascontiguousarray(source.diameters),
        np.ascontiguousarray(source.qualities),
        np.ascontiguousarray(target.positions),
        np.ascontiguousarray(target.diameters),
        np.ascontiguousarray(target.qualities),
        **kw,
    )
    return MatchResult._from_core(d)
