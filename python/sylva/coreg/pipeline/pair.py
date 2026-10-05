# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Stage 2 of the pipeline: registering one pair of scans."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ... import _core
from ..icp import ICPTarget
from ..matching import MatchResult
from ..transforms import _mat4
from .convert import _config_core, _match_core, _pair_core, _pair_from_core, _scan_core, _targets
from .results import CoregConfig, PairResult, ScanFeatures


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
