# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Surface (CHM and DTM) change between two airborne surveys."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ... import _core
from ...als.catalogue import Catalog, _as_catalog, _format, _workers, _written
from ...raster import Raster
from ..points import DoD
from ._common import _chunk_size, _confidence, _count, _num, _pos, _raster
from .align import ALSAlignment, _alignment

#: Class of each cell of a surface change, by code (``SurfaceChange.classes``).
SURFACE_CLASSES = tuple(_core.CHANGE_ALS_CLASSES)


@dataclass
class SurfaceChange:
    """A surface of two airborne surveys compared, from :func:`surface_change`.

    Attributes
    ----------
    surface : {"chm", "dsm", "dtm"}
        What was compared.
    a, b
        The surfaces of each survey on the first survey's catalogue grid
        (for ``"chm"`` and ``"dsm"`` the highest return per cell; NaN
        without returns).
    difference
        ``b - a`` (m) where the cell could be assessed, NaN elsewhere.
    bias
        Change expected from the sampling of the surface alone (m): the mean
        of the permutation distribution. Negative where the second survey
        samples the cell more sparsely, since its highest return then falls
        shorter of the canopy top.
    lower, upper
        The interval (m) a change must leave to be significant: the
        permutation interval widened by the DTM and alignment errors.
    lod
        Its half-width (m), the level of detection.
    sigma
        Standard deviation of the change a cell would show without any (m).
    classes
        ``(rows, cols)`` codes of :data:`SURFACE_CLASSES`: 0 ``no_data``,
        1 ``below_detection``, 2 ``gain`` (above ``upper``), 3 ``loss``
        (below ``lower``).
    sigma_a, sigma_b
        Standard deviation of each survey's value: the spread of its highest
        pulse under the permutation, and its DTM's error.
    pulses_a, pulses_b
        Pulses with a return in the cell, per m².
    density_a, density_b
        First returns per m² in each cell, after harmonisation.
    dod
        The difference with volumes and areas over the significant cells
        (:class:`sylva.change.DoD`; its ``significant`` marks the cells
        outside ``[lower, upper]`` and its ``lod`` is ``lod``).
    sensors
        Per survey (``"a"``, ``"b"``) and as compared after harmonisation
        (``"a_compared"``, ``"b_compared"``): returns, pulses, area, pulse
        density, returns per pulse, share of single-return pulses, mean
        absolute scan angle and most returns per pulse.
    median_bias
        Median of ``bias`` over cells that are canopy (above 2 m) in both.
    notes
        Differences between the sensors that bias the comparison, in words.
    settings
        The settings of the call.
    """

    surface: str
    a: Raster
    b: Raster
    difference: Raster
    bias: Raster
    lower: Raster
    upper: Raster
    lod: Raster
    sigma: Raster
    classes: np.ndarray
    sigma_a: Raster
    sigma_b: Raster
    pulses_a: Raster
    pulses_b: Raster
    density_a: Raster
    density_b: Raster
    dod: DoD
    sensors: dict
    median_bias: float
    notes: list
    settings: dict = field(default_factory=dict)

    def mask(self, name: str) -> np.ndarray:
        """Cells of one class.

        Parameters
        ----------
        name
            One of :data:`SURFACE_CLASSES`.

        Returns
        -------
        numpy.ndarray
            ``(rows, cols)`` bool.
        """
        if name not in SURFACE_CLASSES:
            raise ValueError(f"name must be one of {SURFACE_CLASSES}, got {name!r}")
        return self.classes == SURFACE_CLASSES.index(name)

    def areas(self) -> dict[str, float]:
        """Area (m²) of each class."""
        cell = self.a.resolution ** 2
        return {n: float((self.classes == k).sum() * cell) for k, n in enumerate(SURFACE_CLASSES)}

    def report(self) -> str:
        """A summary in words, with the sensor notes."""
        ar = self.areas()
        assessed = ar["below_detection"] + ar["gain"] + ar["loss"]
        lines = [f"{self.surface.upper()} change at {self.a.resolution:g} m: "
                 f"{assessed:,.0f} m² assessed, {ar['no_data']:,.0f} m² no data"]
        for n in ("gain", "loss", "below_detection"):
            share = ar[n] / assessed * 100 if assessed else float("nan")
            lines.append(f"  {n:<16} {ar[n]:>12,.0f} m²  ({share:.1f} %)")
        lod = self.lod.data[np.isfinite(self.lod.data)]
        if len(lod):
            lo, hi = np.percentile(lod, 5), np.percentile(lod, 95)
            lines.append(f"  level of detection  median {np.median(lod):.2f} m "
                         f"(5 to 95 %: {lo:.2f} to {hi:.2f})")
        lines.append(f"  volume  +{self.dod.volume_gained:,.0f} / -{self.dod.volume_lost:,.0f} m³ "
                     "over the significant cells")
        for k in ("a", "b"):
            s = self.sensors[k]
            lines.append(f"  survey {k}  {s['pulse_density']:.1f} pulses/m², "
                         f"{s['returns_per_pulse']:.2f} returns per pulse, "
                         f"mean |scan angle| {s['mean_abs_scan_angle']:.1f} deg")
        if self.settings.get("harmonise"):
            lines.append(f"  compared  {self.sensors['a_compared']['pulse_density']:.1f} and "
                         f"{self.sensors['b_compared']['pulse_density']:.1f} pulses/m² "
                         "after harmonisation")
        for n in self.notes:
            lines.append("  note: " + n)
        return "\n".join(lines)

    def write(self, directory, format: str = "asc") -> list[str]:
        """Write ``a``, ``b``, ``difference``, ``lod`` and ``classes`` as
        rasters named ``<surface>_<layer>``.

        Parameters
        ----------
        directory
            Output directory (created if needed).
        format : {"asc", "tif"}
            ESRI ASCII grids (with a ``.prj``), or GeoTIFF.

        Returns
        -------
        list of str
            The files written.
        """
        if format not in ("asc", "tif"):
            raise ValueError(f"format must be 'asc' or 'tif', got {format!r}")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        cls = Raster(self.classes.astype(np.float64), self.a.xmin, self.a.ymin, self.a.resolution,
                     self.a.crs)
        out = []
        for name, r in (("a", self.a), ("b", self.b), ("difference", self.difference),
                        ("lod", self.lod), ("classes", cls)):
            path = d / f"{self.surface}_{name}.{format}"
            if format == "tif":
                r.to_geotiff(path)
            else:
                r.to_ascii_grid(path)
            out.append(str(path))
        return out


def surface_change(catalog_a, catalog_b, surface: str = "chm", resolution: float = 1.0,
                   alignment: ALSAlignment | None = None, harmonise: bool = False,
                   density_cell: float = 10.0, seed: int = 0, first_returns: bool = False,
                   subcircle: float = 0.0, dtm_method: str = "plane", dtm_resolution: float = 1.0,
                   min_returns: int = 2, noise_a: float | None = None,
                   noise_b: float | None = None, interpolation_error: float = 0.02,
                   confidence: float = 0.95, horizontal_sigma: float = 0.0,
                   vertical_sigma: float = 0.0, chunk_size: float | None = None,
                   buffer: float = 20.0, workers: int | None = None) -> SurfaceChange:
    """Difference of a CHM, DSM or DTM of two surveys with a level of
    detection per cell.

    Both surveys are gridded in one pass with one algorithm (the highest
    return per cell), each normalised by a DTM of its own ground returns.

    The highest of a cell's returns is a sample of its canopy: it falls
    short of the top by an amount that depends on the number of pulses and
    on how the heights are spread in the cell, and at a crown's edge one
    survey may hit the crown where the other hits only ground. Within a
    cell each pulse contributes its highest return there; if nothing
    changed, the pulses of both surveys sample one surface and every split
    of the pooled values into the two surveys' numbers is equally likely.
    The distribution of ``max(b) - max(a)`` over the splits, exact from the
    order statistics of the pooled values (a permutation test on the
    maxima), is the change sampling alone would give: its mean is
    ``bias``, and its central ``confidence`` interval, widened by the DTM
    errors (``noise / sqrt(n_ground)`` plus ``interpolation_error`` times
    the distance to the nearest ground return) and the alignment's
    uncertainty (the horizontal part times the surface's gradient; for DSMs
    and DTMs also the vertical part), is ``[lower, upper]``. A change above
    it is a gain, below it a loss; cells with fewer than ``min_returns``
    pulses in either survey are ``no_data``. A DTM is compared with the DTM
    and alignment errors alone.

    The permutation assumes the two sensors sample the canopy alike except
    for their numbers of pulses. ``notes`` names the sensor differences
    (pulse density, returns per pulse, scan angle) that bias CHM change
    otherwise; ``harmonise=True`` thins the denser survey, pulse by pulse,
    to the other's pulse density in each ``density_cell`` square (as
    :func:`harmonise` does to whole catalogues), and ``first_returns=True``
    compares first returns only.

    Parameters
    ----------
    catalog_a, catalog_b
        The earlier and the later survey, with ground classified.
    surface : {"chm", "dsm", "dtm"}
        Canopy heights, surface elevations, or terrain.
    resolution
        Cell size (m).
    alignment
        From :func:`align_surveys`: moves the second survey into the
        first's frame, and gives the alignment uncertainty (and, when
        ``noise_a`` and ``noise_b`` are not given, the noise of each survey).
    harmonise, density_cell, seed
        Thin the denser survey to the other's pulse density in squares of
        ``density_cell`` m, choosing pulses by a hash of the pulse and
        ``seed``.
    first_returns
        Grid first returns only.
    subcircle
        Replace each return by eight points on a circle of this radius (m),
        as lidR's ``p2r(subcircle)``; 0 for the returns themselves.
    dtm_method : {"plane", "tin", "lowest"}
        Each survey's DTM (for normalisation, and the surface of ``"dtm"``):
        at each cell a plane fitted to the ground returns around it (the
        same whatever the chunks, with its standard error), a triangulation
        of the ground returns, or the lowest per cell.
    dtm_resolution
        Cell size of the normalising DTM (m).
    min_returns
        Fewest pulses in a cell of each survey.
    noise_a, noise_b
        Return noise of each survey (m), for the DTM error; by default the
        alignment's plane residuals, else 0.05 m.
    interpolation_error
        Growth of the DTM error with distance to the nearest ground return
        (m per m).
    confidence
        Of the interval.
    horizontal_sigma, vertical_sigma
        Alignment uncertainty (m) when no ``alignment`` is given.
    chunk_size, buffer, workers
        As for :func:`sylva.als.apply` (the buffer is raised to two cells,
        and with ``harmonise`` to ``density_cell`` plus two cells).

    Returns
    -------
    SurfaceChange

    Raises
    ------
    ValueError
        For bad settings, or if no chunk has ground returns in both surveys.
    """
    a, b = _as_catalog(catalog_a), _as_catalog(catalog_b)
    if surface not in ("chm", "dsm", "dtm"):
        raise ValueError(f"surface must be 'chm', 'dsm' or 'dtm', got {surface!r}")
    if dtm_method not in ("plane", "tin", "lowest"):
        raise ValueError(f"dtm_method must be 'plane', 'tin' or 'lowest', got {dtm_method!r}")
    if alignment is not None and not isinstance(alignment, ALSAlignment):
        raise ValueError("alignment must be an ALSAlignment (from align_surveys), got "
                         f"{type(alignment).__name__}")

    def noise(given, estimated):
        if given is not None:
            return given
        return estimated if alignment is not None and np.isfinite(estimated) else 0.05

    noise_a = noise(noise_a, alignment.noise_a if alignment is not None else np.nan)
    noise_b = noise(noise_b, alignment.noise_b if alignment is not None else np.nan)
    settings = dict(surface=surface, resolution=_pos("resolution", resolution),
                    harmonise=bool(harmonise), density_cell=_pos("density_cell", density_cell),
                    seed=_count("seed", seed), first_returns=bool(first_returns),
                    subcircle=_num("subcircle", subcircle, 0.0), dtm_method=dtm_method,
                    dtm_resolution=_pos("dtm_resolution", dtm_resolution),
                    min_returns=_count("min_returns", min_returns, 1),
                    noise_a=_num("noise_a", noise_a, 0.0), noise_b=_num("noise_b", noise_b, 0.0),
                    interpolation_error=_num("interpolation_error", interpolation_error, 0.0),
                    confidence=_confidence(confidence),
                    horizontal_sigma=_num("horizontal_sigma", horizontal_sigma, 0.0),
                    vertical_sigma=_num("vertical_sigma", vertical_sigma, 0.0),
                    aligned=alignment is not None)
    s = settings
    d = _core.change_als_surface(a._core(), b._core(), _alignment(alignment), surface,
                                 s["resolution"], s["dtm_resolution"], dtm_method,
                                 s["first_returns"], s["subcircle"], s["min_returns"],
                                 s["noise_a"], s["noise_b"], s["interpolation_error"],
                                 s["confidence"], s["density_cell"] if harmonise else None,
                                 s["seed"], s["horizontal_sigma"], s["vertical_sigma"],
                                 _chunk_size(chunk_size), _num("buffer", buffer, 0.0),
                                 _workers(workers))
    r = {k: _raster(a, d[k]) for k in ("a", "b", "difference", "bias", "lower", "upper", "lod",
                                        "sigma", "sigma_a", "sigma_b", "pulses_a", "pulses_b",
                                        "density_a", "density_b")}
    classes = np.asarray(d["classes"])
    dod = DoD(r["difference"], r["lod"], (classes == 2) | (classes == 3), d["volume_gained"],
              d["volume_lost"], d["volume_gained"] - d["volume_lost"], d["area_changed"],
              d["area_compared"])
    sensors = {"a": d["raw_a"], "b": d["raw_b"], "a_compared": d["stats_a"],
               "b_compared": d["stats_b"]}
    return SurfaceChange(surface, r["a"], r["b"], r["difference"], r["bias"], r["lower"],
                         r["upper"], r["lod"], r["sigma"], classes, r["sigma_a"], r["sigma_b"],
                         r["pulses_a"], r["pulses_b"], r["density_a"], r["density_b"], dod,
                         sensors, float(d["median_bias"]), list(d["notes"]), settings)


def chm_change(catalog_a, catalog_b, resolution: float = 1.0, **kw) -> SurfaceChange:
    """Canopy height model change: :func:`surface_change` with ``surface="chm"``.

    Parameters
    ----------
    catalog_a, catalog_b
        The earlier and the later survey, with ground classified.
    resolution
        Cell size (m).
    **kw
        Any other argument of :func:`surface_change`.

    Returns
    -------
    SurfaceChange
    """
    return surface_change(catalog_a, catalog_b, "chm", resolution, **kw)


def dtm_change(catalog_a, catalog_b, resolution: float = 1.0, **kw) -> SurfaceChange:
    """Terrain change: :func:`surface_change` with ``surface="dtm"``.

    Parameters
    ----------
    catalog_a, catalog_b
        The earlier and the later survey, with ground classified.
    resolution
        Cell size (m).
    **kw
        Any other argument of :func:`surface_change`.

    Returns
    -------
    SurfaceChange
    """
    return surface_change(catalog_a, catalog_b, "dtm", resolution, **kw)


def harmonise(catalog_a, catalog_b, out_a, out_b, density_cell: float = 10.0, seed: int = 0,
              format: str | None = None, chunk_size: float | None = None,
              workers: int | None = None) -> tuple[Catalog, Catalog]:
    """Thin two surveys to a common pulse density and write them as tiles.

    In each ``density_cell`` square (on a grid through the origin), the
    survey with more first returns keeps each of its pulses with probability
    ``n_other / n_own``, all returns of a pulse together, chosen by a hash of
    the pulse (its GPS time and flight line) and ``seed``: the same pulses
    whatever the tiling, and the same that :func:`surface_change` keeps with
    ``harmonise=True``. Tree detection and metrics on the thinned tiles then
    compare like with like.

    Parameters
    ----------
    catalog_a, catalog_b
        The two surveys.
    out_a, out_b
        Directories for the thinned tiles of each (one file per chunk).
    density_cell
        Side (m) of the squares in which densities are compared.
    seed
        Seed of the pulse selection.
    format : {"las", "laz"}, optional
        Output format (that of the input tiles by default).
    chunk_size, workers
        As for :func:`sylva.als.apply`.

    Returns
    -------
    (Catalog, Catalog)
        The thinned surveys.
    """
    a, b = _as_catalog(catalog_a), _as_catalog(catalog_b)
    cell = _pos("density_cell", density_cell)
    seed = _count("seed", seed)
    fmt, cs, w = _format(format), _chunk_size(chunk_size), _workers(workers)
    pa = _core.change_als_harmonise(a._core(), b._core(), str(out_a), cell, seed, fmt, cs, w)
    pb = _core.change_als_harmonise(b._core(), a._core(), str(out_b), cell, seed, fmt, cs, w)
    return _written(pa, a), _written(pb, b)
