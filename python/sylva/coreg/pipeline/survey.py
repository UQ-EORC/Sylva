# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Stage 3 of the pipeline: the whole survey, priors and merging the result."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from ... import _core
from ..stems import StemMap
from ..transforms import _mat4
from .convert import (
    _config_core,
    _input_core,
    _pair_from_core,
    _poses,
    _positions,
    _reflector_tuples,
    _scan_core,
    _scan_from_core,
    _stems_from_rows,
    _survey_from_core,
)
from .results import CoregConfig, PairResult, ScanFeatures, SurveyResult
from .scan import _log_for


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
    from ...pointcloud import PointCloud

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
