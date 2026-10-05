# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Change in area-based metrics and in the plant area index."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ... import _core
from ...als.catalogue import _as_catalog, _workers
from ...raster import Raster
from ._common import _chunk_size, _confidence, _count, _num, _pos, _raster
from .align import ALSAlignment, _alignment

_PAI_CLASSES = tuple(_core.CHANGE_ALS_PAI_CLASSES)


@dataclass
class MetricChange:
    """Area-based metrics of two surveys compared, from :func:`metric_change`.

    Attributes
    ----------
    names
        The metrics.
    a, b
        Per metric (name to :class:`~sylva.Raster`): each survey's value.
    difference
        ``b - a``.
    bias, sigma
        Mean and standard deviation of the change the cell would show from
        sampling alone (the permutation distribution).
    lower, upper, lod
        The interval a change must leave to be significant, and its
        half-width.
    classes
        Per metric, ``(rows, cols)`` codes of :data:`SURFACE_CLASSES`
        (``no_data`` also where the permutations gave no interval).
    settings
        The settings of the call.
    """

    names: list
    a: dict
    b: dict
    difference: dict
    bias: dict
    lower: dict
    upper: dict
    lod: dict
    sigma: dict
    classes: dict
    settings: dict = field(default_factory=dict)

    _LAYERS = ("a", "b", "difference", "bias", "lower", "upper", "lod", "sigma", "classes")

    def __getitem__(self, name: str) -> dict:
        if name not in self.names:
            raise KeyError(f"no metric {name!r}; have {self.names}")
        return {k: getattr(self, k)[name] for k in self._LAYERS}

    def report(self) -> str:
        """Per metric: median difference and the shares of cells with a
        significant gain and loss."""
        lines = [f"Metric change at {self.settings.get('resolution', float('nan')):g} m"]
        for n in self.names:
            c = self.classes[n]
            assessed = int((c > 0).sum())
            d = self.difference[n].data
            med = float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")
            g = (c == 2).sum() / assessed * 100 if assessed else float("nan")
            lo = (c == 3).sum() / assessed * 100 if assessed else float("nan")
            lines.append(f"  {n:<14} median change {med:+.3f}, gain in {g:.0f} %, loss in "
                         f"{lo:.0f} % of {assessed} cells")
        return "\n".join(lines)


def metric_change(catalog_a, catalog_b, resolution: float = 20.0, metrics=None,
                  alignment: ALSAlignment | None = None, harmonise: bool = False,
                  density_cell: float = 10.0, seed: int = 0, permutations: int = 100,
                  stratum: float = 2.0, threshold: float = 2.0, entropy_bin: float = 1.0,
                  cover_break: float = 2.0, min_height: float | None = None,
                  drop_noise: bool = True, dtm_method: str = "plane",
                  dtm_resolution: float = 1.0, confidence: float = 0.95,
                  chunk_size: float | None = None, buffer: float = 20.0,
                  workers: int | None = None) -> MetricChange:
    """Differences of area-based metrics between two surveys, with their
    level of detection.

    The metrics of :func:`sylva.als.grid_metrics` are computed per cell for
    both surveys in one pass, heights above each survey's own DTM, the second
    survey moved by ``alignment`` and, with ``harmonise``, the denser survey
    thinned as :func:`surface_change` thins it.

    Whether a metric changed is tested by a stratified permutation test
    (Pitman 1937): if nothing changed, the pulses of both surveys that fell
    in one ``stratum`` m square of the cell sample the same canopy, and any
    reassignment of them between the surveys that keeps each survey's number
    of pulses in each square is as likely as the observed one. ``bias`` and
    ``[lower, upper]`` are the mean and central ``confidence`` interval of
    ``metric(b) - metric(a)`` over ``permutations`` such reassignments, drawn
    from a stream seeded by the cell and ``seed`` so that the result does not
    depend on the chunks. Keeping pulses in their squares keeps the pattern
    in which each survey sampled the cell; resampling the cell's pulses as
    independent draws would ignore it and understate the uncertainty. The
    test covers sampling, not differences between the sensors (footprint,
    sensitivity), which bias metrics systematically: harmonisation removes
    the part due to pulse density only.

    Parameters
    ----------
    catalog_a, catalog_b
        The earlier and the later survey, with ground classified.
    resolution
        Cell size (m).
    metrics
        Names (see :func:`sylva.als.metric_names`); all by default.
    alignment, harmonise, density_cell, seed
        As for :func:`surface_change` (``seed`` also seeds the permutations).
    permutations
        Reassignments per cell; 0 for none (no level of detection, every
        change ``no_data``).
    stratum
        Side (m) of the squares within which pulses are reassigned.
    threshold, entropy_bin, cover_break, min_height, drop_noise
        As for :func:`sylva.als.grid_metrics`.
    dtm_method, dtm_resolution
        Each survey's normalising DTM, as for :func:`surface_change`.
    confidence
        Of the interval.
    chunk_size, buffer, workers
        As for :func:`sylva.als.apply`.

    Returns
    -------
    MetricChange
    """
    a, b = _as_catalog(catalog_a), _as_catalog(catalog_b)
    if dtm_method not in ("plane", "tin", "lowest"):
        raise ValueError(f"dtm_method must be 'plane', 'tin' or 'lowest', got {dtm_method!r}")
    if isinstance(metrics, str):
        metrics = [metrics]
    names = None if metrics is None else [str(m) for m in metrics]
    settings = dict(resolution=_pos("resolution", resolution), metrics=names,
                    harmonise=bool(harmonise), density_cell=_pos("density_cell", density_cell),
                    seed=_count("seed", seed), permutations=_count("permutations", permutations),
                    stratum=_pos("stratum", stratum), threshold=_num("threshold", threshold),
                    entropy_bin=_pos("entropy_bin", entropy_bin),
                    cover_break=_num("cover_break", cover_break),
                    min_height=None if min_height is None else _num("min_height", min_height),
                    drop_noise=bool(drop_noise), dtm_method=dtm_method,
                    dtm_resolution=_pos("dtm_resolution", dtm_resolution),
                    confidence=_confidence(confidence), aligned=alignment is not None)
    s = settings
    d = _core.change_als_metrics(a._core(), b._core(), _alignment(alignment), s["resolution"],
                                 names,
                                 s["threshold"], s["entropy_bin"], s["cover_break"],
                                 s["min_height"], s["drop_noise"], s["dtm_resolution"], dtm_method,
                                 s["permutations"], s["stratum"], s["seed"], s["confidence"],
                                 s["density_cell"] if harmonise else None, s["seed"],
                                 _chunk_size(chunk_size), _num("buffer", buffer, 0.0),
                                 _workers(workers))
    nm = list(d["names"])

    def layer(k):
        return {n: _raster(a, r) for n, r in zip(nm, d[k], strict=True)}

    return MetricChange(nm, layer("a"), layer("b"), layer("difference"), layer("bias"),
                        layer("lower"), layer("upper"), layer("lod"), layer("sigma"),
                        {n: np.asarray(c) for n, c in zip(nm, d["classes"], strict=True)}, settings)


@dataclass
class PAIChange:
    """Plant area index of two surveys compared, from :func:`pai_change`.

    Attributes
    ----------
    pai_a, pai_b, sigma_a, sigma_b
        PAI of each survey per cell and its standard deviation.
    difference, sigma, lod
        ``pai_b - pai_a``, its standard deviation and level of detection.
    classes
        ``(rows, cols)`` codes: 0 ``no_data``, 1 ``below_detection``,
        2 ``gain``, 3 ``loss``, 4 ``saturated`` (no pulse crossed the
        canopy in one survey, so its PAI is only a lower bound).
    profile
        The pooled profile of the whole area: ``height``, ``pad_a``,
        ``pad_b``, ``difference`` and ``sigma`` per layer.
    """

    pai_a: Raster
    pai_b: Raster
    sigma_a: Raster
    sigma_b: Raster
    difference: Raster
    sigma: Raster
    lod: Raster
    classes: np.ndarray
    profile: dict

    def mask(self, name: str) -> np.ndarray:
        """Cells of one class.

        Parameters
        ----------
        name : {"no_data", "below_detection", "gain", "loss", "saturated"}
            The class.

        Returns
        -------
        numpy.ndarray
            ``(rows, cols)`` bool.
        """
        if name not in _PAI_CLASSES:
            raise ValueError(f"name must be one of {_PAI_CLASSES}, got {name!r}")
        return self.classes == _PAI_CLASSES.index(name)


def _profile_dict(p, name: str) -> dict:
    need = ("xmin", "ymin", "resolution", "min_height", "bin_size", "weight", "weight_k")
    if not all(hasattr(p, k) for k in need):
        raise ValueError(f"{name} must be an ALSProfile (from sylva.als.gap_profile), got "
                         f"{type(p).__name__}")
    return {"xmin": float(p.xmin), "ymin": float(p.ymin), "resolution": float(p.resolution),
            "min_height": float(p.min_height), "bin_size": float(p.bin_size),
            "weight": np.ascontiguousarray(p.weight, dtype=np.float64),
            "weight_k": np.ascontiguousarray(p.weight_k, dtype=np.float64)}


def pai_change(profile_a, profile_b, confidence: float = 0.95) -> PAIChange:
    """Plant area index change from two gap-fraction profiles.

    Both :class:`sylva.als.ALSProfile` must come from
    :func:`sylva.als.gap_profile` on the same grid and layers (the same
    ``resolution``, ``bin_size``, ``min_height`` and ``max_height`` over the
    same area), each with its survey's trajectory. With ``W`` pulses in a
    cell, a share ``P`` reaching the ground and a mean extinction ``k``,
    ``PAI = -ln P / k`` and, by the delta method on the binomial ``P``,
    its standard deviation is ``sqrt((1 - P) / (W P)) / k``. A cell no pulse
    crossed is ``saturated``. The profiles are not moved by an alignment:
    compare cells much larger than the horizontal offset.

    Parameters
    ----------
    profile_a, profile_b
        The two profiles.
    confidence
        Of the level of detection.

    Returns
    -------
    PAIChange
    """
    pa, pb = _profile_dict(profile_a, "profile_a"), _profile_dict(profile_b, "profile_b")
    d = _core.change_als_pai(pa, pb, _confidence(confidence))
    crs = getattr(profile_a, "crs", None)

    def r(k):
        x = Raster._from_core(d[k])
        x.crs = crs
        return x

    prof = {k: np.asarray(v) for k, v in _core.change_als_profile(pa, pb, None).items()}
    return PAIChange(r("pai_a"), r("pai_b"), r("sigma_a"), r("sigma_b"), r("difference"),
                     r("sigma"), r("lod"), np.asarray(d["classes"]), prof)


def profile_change(profile_a, profile_b, mask=None) -> dict:
    """Plant area density profile of an area in two surveys, from the
    pooled counts of its cells.

    Parameters
    ----------
    profile_a, profile_b
        :class:`sylva.als.ALSProfile` on one grid and layers.
    mask
        ``(ny, nx)`` bool, the cells of the area; all by default.

    Returns
    -------
    dict
        ``height`` (layer centres, m), ``pad_a``, ``pad_b``, ``difference``
        and ``sigma`` (m²/m³) per layer; ``sigma`` is NaN for a layer no
        pulse crossed in either survey.
    """
    pa, pb = _profile_dict(profile_a, "profile_a"), _profile_dict(profile_b, "profile_b")
    m = None if mask is None else np.ascontiguousarray(mask, dtype=bool)
    if m is not None and m.shape != pa["weight"].shape[1:]:
        raise ValueError(f"mask must have shape {pa['weight'].shape[1:]}, got {m.shape}")
    return {k: np.asarray(v) for k, v in _core.change_als_profile(pa, pb, m).items()}
