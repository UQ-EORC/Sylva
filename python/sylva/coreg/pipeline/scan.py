# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Stage 1 of the pipeline: features of each scan."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ... import _core
from ..ground import GroundModel
from ..reflectors import Reflector
from ..transforms import _mat4
from .convert import _config_core, _input_core, _points, _reflector_tuples, _scan_from_core
from .results import CoregConfig, ScanFeatures


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
    from ...riscan import read_export_settings

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
