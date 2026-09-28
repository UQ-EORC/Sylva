# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Change between two airborne lidar surveys of one area.

Every function works on catalogues of tiles (:func:`sylva.als.catalog`)
through the chunk engine of :mod:`sylva.als`: each chunk reads the same
buffered box from both surveys, and the results do not depend on the chunk
size or the number of workers. As elsewhere in :mod:`sylva.change`, every
change carries its uncertainty or level of detection, and what the data
cannot support is labelled (below detection, no data, uncertain,
undetected, unobserved) rather than reported as change.

::

    from sylva import als, change

    a, b = als.catalog("2019/"), als.catalog("2024/")          # ground classified
    al = change.align_surveys(a, b, stable_classes=(2, 6))      # ground and roofs
    chm = change.chm_change(a, b, resolution=1.0, alignment=al, harmonise=True)
    gaps = change.gap_change(chm, height=2.0, min_area=10.0, years=5)
    trees = change.tree_change(als.find_trees(a), als.find_trees(b), chm, alignment=al)
    metrics = change.metric_change(a, b, resolution=20.0, metrics=["zq95", "cover"], alignment=al)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .. import _core
from ..als import Catalog, _as_catalog, _format, _workers, _written
from ..pointcloud import PointCloud
from ..raster import Raster
from .points import DoD

__all__ = [
    "ALSAlignment", "align_surveys",
    "SurfaceChange", "surface_change", "chm_change", "dtm_change", "harmonise",
    "Gaps", "canopy_gaps", "GapChange", "gap_change",
    "ALSTreeChange", "tree_change",
    "MetricChange", "metric_change", "PAIChange", "pai_change", "profile_change",
    "SURFACE_CLASSES", "GAP_CELLS",
]

#: Class of each cell of a surface change, by code (``SurfaceChange.classes``).
SURFACE_CLASSES = tuple(_core.CHANGE_ALS_CLASSES)
_PAI_CLASSES = tuple(_core.CHANGE_ALS_PAI_CLASSES)


# ------------------------------------------------------------------ checks

def _num(name: str, v, lo: float | None = None, strict: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    if lo is not None and (v < lo or (strict and v == lo)):
        bound = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {bound} {lo}, got {v!r}")
    return float(v)


def _pos(name: str, v) -> float:
    return _num(name, v, 0.0, strict=True)


def _count(name: str, v, lo: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, Integral) or v < lo:
        raise ValueError(f"{name} must be an integer of at least {lo}, got {v!r}")
    return int(v)


def _share(name: str, v) -> float:
    v = _num(name, v)
    if not 0 <= v <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {v}")
    return v


def _confidence(v) -> float:
    v = _num("confidence", v)
    if not 0 < v < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {v}")
    return v


def _chunk_size(v) -> float | None:
    return None if v is None else _pos("chunk_size", v)


def _raster_arg(r: Raster, name: str) -> tuple:
    if not isinstance(r, Raster):
        raise ValueError(f"{name} must be a Raster, got {type(r).__name__}")
    return (np.ascontiguousarray(r.data, dtype=np.float64), float(r.xmin), float(r.ymin),
            float(r.resolution))


def _raster(cat: Catalog, d: dict) -> Raster:
    return cat._raster(d)


# ------------------------------------------------------------------ alignment

@dataclass
class ALSAlignment:
    """Offsets of a second airborne survey relative to a first, from
    :func:`align_surveys`.

    A point ``p`` of the second survey belongs at ``p - offset`` in the frame
    of the first. The offsets are estimated per block of a grid and given as
    one constant, per block, or as a smooth field (``model``).

    Attributes
    ----------
    xmin, ymin, block_size
        South-west corner and side (m) of the block grid.
    model : {"constant", "blocks", "field"}
        How the offset at a point is obtained.
    smoothing
        Standard deviation (m) of the Gaussian weights of the field.
    values, sigmas
        ``(ny, nx, 3)`` offset ``(dx, dy, dz)`` of the model at each block
        centre (m) and its standard deviation.
    offset, sigma
        ``(3,)`` constant offset over all blocks and its standard deviation.
    birge
        ``(3,)`` Birge ratio of the blocks about the model: above 1 where
        the blocks disagree by more than their uncertainties, which have been
        widened by it.
    noise_a, noise_b
        Median residual RMS (m) of planes fitted to each survey's stable
        returns: return noise plus the roughness of the surface within the
        plane radius. An upper bound on the return noise.
    blocks
        Per block (``(ny, nx)`` arrays, ``(ny, nx, 3)`` for vectors):
        ``fitted``, ``n_samples``, ``n_used``, ``n_eff`` (independent
        samples), ``offset`` and ``sigma`` (the block's own estimate),
        ``median_before`` (median height of the second survey above the
        first before any offset), ``spread_before`` and ``spread_after``
        (robust spread of those heights before and after),
        ``horizontal_determined``, ``noise_a``, ``noise_b`` and ``iterations``.
    residuals
        ``(ny, nx, 3)`` block estimate minus the model at the block centre.
    crs
        Coordinate system of the first survey.
    settings
        The settings of the call.
    """

    xmin: float
    ymin: float
    block_size: float
    model: str
    smoothing: float
    values: np.ndarray
    sigmas: np.ndarray
    offset: np.ndarray
    sigma: np.ndarray
    birge: np.ndarray = field(default_factory=lambda: np.ones(3))
    noise_a: float = float("nan")
    noise_b: float = float("nan")
    blocks: dict = field(default_factory=dict)
    residuals: np.ndarray | None = None
    crs: str | None = None
    settings: dict = field(default_factory=dict)

    @classmethod
    def _from_core(cls, d: dict, crs=None, settings=None) -> ALSAlignment:
        ny, nx = int(d["ny"]), int(d["nx"])
        shape3 = (ny, nx, 3)
        blocks = {k: np.asarray(v).reshape(shape3 if np.asarray(v).ndim == 2 else (ny, nx))
                  for k, v in d["blocks"].items()}
        return cls(float(d["xmin"]), float(d["ymin"]), float(d["block_size"]), d["model"],
                   float(d["smoothing"]), np.asarray(d["values"]).reshape(shape3),
                   np.asarray(d["sigmas"]).reshape(shape3), np.asarray(d["global"]),
                   np.asarray(d["global_sigma"]), np.asarray(d["birge"]), float(d["noise_a"]),
                   float(d["noise_b"]), blocks, np.asarray(d["residuals"]).reshape(shape3), crs,
                   settings or {})

    @classmethod
    def constant(cls, offset, sigma=(0.0, 0.0, 0.0)) -> ALSAlignment:
        """A known constant offset.

        Parameters
        ----------
        offset
            ``(dx, dy, dz)`` of the second survey relative to the first (m).
        sigma
            Its standard deviation (m).

        Returns
        -------
        ALSAlignment
        """
        o = np.asarray(offset, dtype=float).reshape(3)
        s = np.asarray(sigma, dtype=float).reshape(3)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(s)) and np.all(s >= 0)):
            raise ValueError("offset must be three finite numbers and sigma three "
                             "non-negative ones")
        return cls(0.0, 0.0, 1.0, "constant", float("nan"), o.reshape(1, 1, 3), s.reshape(1, 1, 3),
                   o, s)

    def _core(self) -> dict:
        ny, nx = self.values.shape[:2]
        return {"xmin": float(self.xmin), "ymin": float(self.ymin),
                "block_size": float(self.block_size), "nx": int(nx), "ny": int(ny),
                "model": self.model, "smoothing": float(self.smoothing),
                "values": np.ascontiguousarray(self.values.reshape(-1, 3), dtype=np.float64),
                "sigmas": np.ascontiguousarray(self.sigmas.reshape(-1, 3), dtype=np.float64),
                "global": [float(v) for v in self.offset],
                "global_sigma": [float(v) for v in self.sigma]}

    def offset_at(self, x, y) -> np.ndarray:
        """Offset ``(dx, dy, dz)`` of the second survey at points.

        Parameters
        ----------
        x, y
            Coordinates (arrays of one length, or numbers).

        Returns
        -------
        numpy.ndarray
            ``(N, 3)`` offsets (m).
        """
        x = np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64)
        y = np.ascontiguousarray(np.atleast_1d(y), dtype=np.float64)
        return _core.change_als_offsets(self._core(), x, y)[0]

    def sigma_at(self, x, y) -> np.ndarray:
        """Standard deviation of :meth:`offset_at`.

        Parameters
        ----------
        x, y
            Coordinates (arrays of one length, or numbers).

        Returns
        -------
        numpy.ndarray
            ``(N, 3)`` standard deviations (m).
        """
        x = np.ascontiguousarray(np.atleast_1d(x), dtype=np.float64)
        y = np.ascontiguousarray(np.atleast_1d(y), dtype=np.float64)
        return _core.change_als_offsets(self._core(), x, y)[1]

    def apply(self, cloud: PointCloud) -> PointCloud:
        """Move a cloud of the second survey into the frame of the first.

        Parameters
        ----------
        cloud
            Points of the second survey.

        Returns
        -------
        PointCloud
            A copy with ``xyz - offset_at(x, y)``.
        """
        if not isinstance(cloud, PointCloud):
            raise ValueError(f"cloud must be a PointCloud, got {type(cloud).__name__}")
        out = cloud.copy()
        out.xyz = np.ascontiguousarray(cloud.xyz - self.offset_at(cloud.x, cloud.y))
        return out

    def table(self) -> dict[str, np.ndarray]:
        """One row per block with an estimate: centre ``x``, ``y``, the
        block's ``dx``, ``dy``, ``dz`` and their standard deviations
        ``sd_dx``, ``sd_dy``, ``sd_dz``, the model's ``model_dx``,
        ``model_dy``, ``model_dz`` there, ``n_used``, ``n_eff``,
        ``horizontal_determined``, ``spread_before`` and ``spread_after``."""
        if not self.blocks:
            return {}
        f = self.blocks["fitted"]
        rows, cols = np.nonzero(f)
        out = {"x": self.xmin + (cols + 0.5) * self.block_size,
               "y": self.ymin + (rows + 0.5) * self.block_size}
        for k, name in enumerate(("dx", "dy", "dz")):
            out[name] = self.blocks["offset"][rows, cols, k]
            out["sd_" + name] = self.blocks["sigma"][rows, cols, k]
            out["model_" + name] = self.values[rows, cols, k]
        for name in ("n_used", "n_eff", "horizontal_determined", "spread_before", "spread_after"):
            out[name] = self.blocks[name][rows, cols]
        return out

    def to_csv(self, path) -> None:
        """Write :meth:`table` as CSV.

        Parameters
        ----------
        path
            Output file.
        """
        t = self.table()
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(t))
            for row in zip(*t.values(), strict=True):
                w.writerow([_cell(v) for v in row])

    def report(self) -> str:
        """A summary in words."""
        o, s = self.offset, self.sigma
        lines = [f"ALS alignment ({self.model}): constant offset dx {o[0]:+.3f} ± {s[0]:.3f}, "
                 f"dy {o[1]:+.3f} ± {s[1]:.3f}, dz {o[2]:+.3f} ± {s[2]:.3f} m"]
        if self.blocks:
            f = self.blocks["fitted"]
            prior = self.settings.get("horizontal_prior", np.inf)
            fixed = [int((self.blocks["sigma"][..., k][f] < 0.5 * prior).sum()) for k in (0, 1)]
            lines.append(f"  blocks   {int(f.sum())} of {f.size} of {self.block_size:g} m with "
                         f"an estimate; dx fixed by the data in {fixed[0]}, dy in {fixed[1]}")
            sb, sa = self.blocks["spread_before"][f], self.blocks["spread_after"][f]
            lines.append(f"  spread   {np.nanmedian(sb):.3f} m before, {np.nanmedian(sa):.3f} m "
                         "after (median over blocks, robust)")
            lines.append(f"  Birge    {self.birge[0]:.2f}, {self.birge[1]:.2f}, {self.birge[2]:.2f}"
                         " (x, y, z)")
            lines.append(f"  noise    {self.noise_a:.3f} m (a), {self.noise_b:.3f} m (b)")
        return "\n".join(lines)


def _cell(v):
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (Integral, np.integer)):
        return int(v)
    if isinstance(v, str):
        return v
    v = float(v)
    return "" if not np.isfinite(v) else v


def align_surveys(catalog_a, catalog_b, block_size: float = 100.0, stable_classes=(2,),
                  model: str = "field", smoothing: float | None = None, horizontal: bool = True,
                  horizontal_prior: float = 0.3, sample_spacing: float = 1.0, radius: float = 1.5,
                  min_neighbours: int = 6, max_roughness: float = 0.15, max_slope: float = 1.0,
                  max_offset: float = 2.0, correlation_length: float = 5.0,
                  min_samples: int = 30, huber: float = 1.5, iterations: int = 20,
                  chunk_size: float | None = None, workers: int | None = None) -> ALSAlignment:
    """Offsets between two airborne surveys, on stable surfaces.

    Every stable return of the second survey (one per ``sample_spacing``
    square) is compared with a plane fitted to the first survey's stable
    returns within ``radius`` of it. A survey displaced by ``(dx, dy, dz)``
    sits ``dz - gx dx - gy dy`` above a plane of gradient ``(gx, gy)``, so
    the vertical offset follows from every sample and the horizontal ones
    from the variety of slopes and aspects (Nuth and Kääb 2011). Each block
    is solved by iteratively reweighted least squares (Huber weights),
    refitting the planes at the moved positions until the offsets settle.

    A horizontal offset along a slope of one aspect, or on flat ground, looks
    like a vertical one; a Gaussian prior of ``horizontal_prior`` keeps it
    near zero there, and its standard deviation stays near the prior.
    Surfaces of several aspects (terrain, roofs: ``stable_classes=(2, 6)``;
    roads are class 11) fix it. Samples within ``correlation_length`` of each
    other count as one independent sample.

    Parameters
    ----------
    catalog_a, catalog_b
        The reference and the later survey, with ground (and any other
        stable class) classified.
    block_size
        Side of the blocks (m); use the tile size for one estimate per tile.
    stable_classes
        Classes of the returns taken as unchanged: 2 ground; 6 buildings;
        11 roads.
    model : {"field", "blocks", "constant"}
        A smooth field (the blocks pooled with Gaussian weights of
        ``smoothing`` m and interpolated between block centres), each block
        on its own, or one offset for the whole area.
    smoothing
        Standard deviation (m) of the field's weights; ``block_size`` by
        default.
    horizontal
        Estimate horizontal offsets (else only the vertical one).
    horizontal_prior
        Prior standard deviation (m) of each horizontal offset.
    sample_spacing, radius, min_neighbours
        Sampling of the second survey (m), radius (m) and fewest returns of
        the reference planes.
    max_roughness, max_slope
        Planes rougher than this RMS (m) or steeper than this gradient are
        left out (edges, walls, vegetation misclassified as ground).
    max_offset
        Largest offset searched (m).
    correlation_length
        Side (m) of the squares within which samples are not independent.
    min_samples
        Fewest usable samples in a block.
    huber, iterations
        Huber constant (robust standard deviations) and largest number of
        refits.
    chunk_size
        Side of the chunks (m), rounded up to whole blocks; by default the
        median tile size.
    workers
        Chunks at once (one per core by default).

    Returns
    -------
    ALSAlignment

    Raises
    ------
    ValueError
        For bad settings, tiles without a classification, or if no block has
        enough stable samples in both surveys.
    """
    a, b = _as_catalog(catalog_a), _as_catalog(catalog_b)
    block_size = _pos("block_size", block_size)
    if model not in ("field", "blocks", "constant"):
        raise ValueError(f"model must be 'field', 'blocks' or 'constant', got {model!r}")
    smoothing = block_size if smoothing is None else _pos("smoothing", smoothing)
    classes = [int(c) for c in np.atleast_1d(stable_classes)]
    if not classes or any(not 0 <= c <= 255 for c in classes):
        raise ValueError(f"stable_classes must be LAS classes (0 to 255), got {stable_classes!r}")
    if chunk_size is None:
        sizes = [max(t.bounds[3] - t.bounds[0], t.bounds[4] - t.bounds[1]) for t in a.tiles]
        chunk_size = float(np.median(sizes)) if sizes else block_size
    per = max(1, int(np.ceil(_pos("chunk_size", chunk_size) / block_size - 1e-9)))
    settings = dict(block_size=block_size, stable_classes=classes, model=model,
                    smoothing=smoothing, horizontal=bool(horizontal),
                    horizontal_prior=_pos("horizontal_prior", horizontal_prior),
                    sample_spacing=_pos("sample_spacing", sample_spacing),
                    radius=_pos("radius", radius),
                    min_neighbours=_count("min_neighbours", min_neighbours, 3),
                    max_roughness=_pos("max_roughness", max_roughness),
                    max_slope=_pos("max_slope", max_slope),
                    max_offset=_pos("max_offset", max_offset),
                    correlation_length=_pos("correlation_length", correlation_length),
                    min_samples=_count("min_samples", min_samples, 3), huber=_pos("huber", huber),
                    iterations=_count("iterations", iterations, 1))
    s = settings
    d = _core.change_als_align(a._core(), b._core(), s["block_size"], s["stable_classes"],
                               s["sample_spacing"], s["radius"], s["min_neighbours"],
                               s["max_roughness"], s["max_slope"], s["max_offset"], s["horizontal"],
                               s["horizontal_prior"], s["correlation_length"], s["min_samples"],
                               s["huber"], s["iterations"], model, smoothing, per,
                               _workers(workers))
    return ALSAlignment._from_core(d, a.crs, settings)


def _alignment(al) -> dict | None:
    if al is None:
        return None
    if not isinstance(al, ALSAlignment):
        raise ValueError("alignment must be an ALSAlignment (from align_surveys), got "
                         f"{type(al).__name__}")
    return al._core()


# ------------------------------------------------------------------ surfaces

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
            ESRI ASCII grids, or GeoTIFF (needs rasterio).

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


# ------------------------------------------------------------------ gaps

def _polygons(parts) -> list:
    return [(np.asarray(ext), [np.asarray(h) for h in holes]) for ext, holes in parts]


def _geojson_polygon(parts) -> dict:
    def ring(r):
        c = [[float(x), float(y)] for x, y in r]
        return c + [c[0]]
    polys = [[ring(ext)] + [ring(h) for h in holes] for ext, holes in parts]
    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}


def _write_geojson(path, geoms: list, props: list, crs: str | None) -> None:
    feats = [{"type": "Feature", "geometry": g, "properties": p}
             for g, p in zip(geoms, props, strict=True)]
    fc = {"type": "FeatureCollection", "features": feats}
    if crs is not None and crs.startswith("EPSG:"):
        fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs[5:]}"}}
    Path(path).write_text(json.dumps(fc))


def _json_value(v):
    if isinstance(v, (np.integer, Integral)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return v
    v = float(v)
    return v if np.isfinite(v) else None


@dataclass
class Gaps:
    """Canopy gaps of one CHM, from :func:`canopy_gaps`.

    Attributes
    ----------
    labels
        ``(rows, cols)`` gap id of each cell (0 for none).
    id, area, n_cells, x, y, mean_height, max_height
        Per gap: id (1, 2, ... in order of the first cell from the
        south-west), area (m²), cells, centre, and mean and largest CHM
        height of its cells (m).
    polygons
        Per gap, its outline along cell edges: a list of parts, each
        ``(exterior, holes)`` with ``(k, 2)`` rings (exteriors
        counter-clockwise). Parts touch only at corners.
    area_with_data
        Area (m²) of the cells with a CHM value.
    xmin, ymin, resolution
        The CHM's grid.
    crs
        Its coordinate system.
    settings
        ``height``, ``min_area``, ``max_area``, ``connectivity``.
    """

    labels: np.ndarray
    id: np.ndarray
    area: np.ndarray
    n_cells: np.ndarray
    x: np.ndarray
    y: np.ndarray
    mean_height: np.ndarray
    max_height: np.ndarray
    polygons: list
    area_with_data: float
    xmin: float
    ymin: float
    resolution: float
    crs: str | None = None
    settings: dict = field(default_factory=dict)

    @classmethod
    def _from_core(cls, d: dict, r: Raster, settings: dict) -> Gaps:
        return cls(np.asarray(d["labels"]), np.asarray(d["id"]), np.asarray(d["area"]),
                   np.asarray(d["n_cells"]), np.asarray(d["x"]), np.asarray(d["y"]),
                   np.asarray(d["mean_height"]), np.asarray(d["max_height"]),
                   [_polygons(p) for p in d["polygons"]], float(d["area_with_data"]),
                   float(r.xmin), float(r.ymin), float(r.resolution), r.crs, dict(settings))

    def __len__(self) -> int:
        return len(self.id)

    def __repr__(self) -> str:
        return f"Gaps({len(self)} gaps, {self.gap_fraction * 100:.1f} % of the area)"

    @property
    def gap_fraction(self) -> float:
        """Share of the area with data that is gap."""
        if self.area_with_data <= 0:
            return float("nan")
        return float(self.area.sum() / self.area_with_data)

    def table(self) -> dict[str, np.ndarray]:
        """``id``, ``area``, ``n_cells``, ``x``, ``y``, ``mean_height`` and
        ``max_height`` per gap."""
        return {"id": self.id, "area": self.area, "n_cells": self.n_cells, "x": self.x,
                "y": self.y, "mean_height": self.mean_height, "max_height": self.max_height}

    def size_distribution(self, bins=None) -> tuple[np.ndarray, np.ndarray]:
        """Number of gaps per size class.

        Parameters
        ----------
        bins
            Bin edges (m²); by default powers of two from the smallest
            gap size allowed.

        Returns
        -------
        (edges, counts)
        """
        lo = max(self.settings.get("min_area", 0.0), self.resolution ** 2)
        if bins is None:
            top = max(float(self.area.max()) if len(self) else lo, lo)
            k = int(np.ceil(np.log2(top / lo))) + 1
            bins = lo * 2.0 ** np.arange(k + 1)
        edges = np.asarray(bins, dtype=float)
        counts, _ = np.histogram(self.area, bins=edges)
        return edges, counts

    def size_exponent(self, xmin: float | None = None) -> tuple[float, float, int]:
        """Exponent of a power law ``p(a) ~ a^-alpha`` fitted to the gap
        sizes by maximum likelihood (Clauset et al. 2009).

        Parameters
        ----------
        xmin
            Smallest size included (m²); the smallest allowed by default.

        Returns
        -------
        (alpha, standard error, number of gaps)
            NaN with fewer than two gaps.
        """
        if xmin is None:
            xmin = max(self.settings.get("min_area", 0.0), self.resolution ** 2)
        a, se, n = _core.change_als_size_exponent(np.ascontiguousarray(self.area, dtype=np.float64),
                                                  _pos("xmin", xmin))
        return float(a), float(se), int(n)

    def to_geojson(self, path, properties: dict | None = None) -> None:
        """Write the gap outlines as GeoJSON polygons with :meth:`table` as
        properties.

        Parameters
        ----------
        path
            Output file.
        properties
            Extra per-gap columns (name to sequence).
        """
        t = self.table()
        if properties:
            t = {**t, **properties}
        props = [{k: _json_value(v[i]) for k, v in t.items()} for i in range(len(self))]
        _write_geojson(path, [_geojson_polygon(p) for p in self.polygons], props, self.crs)


def canopy_gaps(chm: Raster, height: float = 2.0, min_area: float = 10.0,
                max_area: float | None = None, connectivity: int = 8) -> Gaps:
    """Canopy gaps of a CHM: connected cells no higher than ``height``.

    As ForestGapR's ``getForestGaps`` (Silva et al. 2019): cells at or below
    the height threshold are joined across edges and corners
    (``connectivity=8``) or edges only (4), and regions whose area lies
    between ``min_area`` and ``max_area`` are gaps. A threshold of 2 m
    follows Brokaw's (1982) definition of a gap as an opening reaching down
    to within 2 m of the ground. NaN cells are never gap.

    Parameters
    ----------
    chm
        Canopy height model.
    height
        Gap height threshold (m).
    min_area, max_area
        Size limits (m²); no upper limit by default.
    connectivity : {8, 4}
        Cell neighbourhood.

    Returns
    -------
    Gaps
    """
    r = _raster_arg(chm, "chm")
    settings = _gap_settings(height, min_area, max_area, connectivity)
    d = _core.change_als_gaps(r, settings["height"], settings["min_area"],
                              _inf(settings["max_area"]), settings["connectivity"])
    return Gaps._from_core(d, chm, settings)


def _inf(v):
    return float("inf") if v is None else v


def _gap_settings(height, min_area, max_area, connectivity) -> dict:
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    lo = _num("min_area", min_area, 0.0)
    hi = None if max_area is None else _num("max_area", max_area, lo)
    return dict(height=_num("height", height), min_area=lo, max_area=hi,
                connectivity=int(connectivity))


#: Names of the cell codes of a gap change (``GapChange.cells``).
GAP_CELLS = ("canopy", "stable_gap", "formed", "closed", "uncertain", "no_data")


@dataclass
class GapChange:
    """Canopy gaps of two surveys and their dynamics, from :func:`gap_change`.

    Attributes
    ----------
    a, b
        The gaps of each survey.
    cells
        ``(rows, cols)`` codes of :data:`GAP_CELLS`: ``canopy`` (no gap in
        either), ``stable_gap``, ``formed`` (gap only in the second survey,
        with a significant loss), ``closed`` (gap only in the first, with a
        significant gain), ``uncertain`` (crossed the threshold without a
        significant change) and ``no_data``.
    status_a, closed_area
        Per gap of the first survey: ``closed`` (no part is gap any more),
        ``shrunk``, ``stable`` or ``uncertain``; and its area closed (m²).
    status_b, formed_area
        Per gap of the second survey: ``new``, ``expanded``, ``stable`` or
        ``uncertain``; and its area formed (m²).
    areas
        Area (m²) per cell code.
    years
        Time between the surveys, if given.
    """

    a: Gaps
    b: Gaps
    cells: np.ndarray
    status_a: list
    closed_area: np.ndarray
    status_b: list
    formed_area: np.ndarray
    areas: dict
    years: float | None = None

    def summary(self) -> dict:
        """Gap fractions, areas formed and closed (with annual rates as a
        share of the area compared when ``years`` is known), gap counts by
        status and the size-distribution exponents of both surveys."""
        compared = sum(v for k, v in self.areas.items() if k != "no_data")
        out = {"gap_fraction_a": self.a.gap_fraction, "gap_fraction_b": self.b.gap_fraction,
               "area_compared": compared, "area_formed": self.areas["formed"],
               "area_closed": self.areas["closed"], "area_uncertain": self.areas["uncertain"]}
        for k in ("new", "expanded", "stable", "uncertain"):
            out[f"gaps_b_{k}"] = int(sum(1 for s in self.status_b if s == k))
        for k in ("closed", "shrunk", "stable", "uncertain"):
            out[f"gaps_a_{k}"] = int(sum(1 for s in self.status_a if s == k))
        if self.years and compared > 0:
            out["formation_rate"] = self.areas["formed"] / compared / self.years
            out["closure_rate"] = self.areas["closed"] / compared / self.years
        for k, g in (("a", self.a), ("b", self.b)):
            alpha, se, n = g.size_exponent()
            out[f"size_exponent_{k}"], out[f"size_exponent_se_{k}"] = alpha, se
        return out

    def report(self) -> str:
        """A summary in words."""
        s = self.summary()
        lines = [f"Gaps (at most {self.a.settings['height']:g} m high, at least "
                 f"{self.a.settings['min_area']:g} m²): {len(self.a)} in survey a "
                 f"({s['gap_fraction_a'] * 100:.1f} %), {len(self.b)} in survey b "
                 f"({s['gap_fraction_b'] * 100:.1f} %)",
                 f"  formed   {s['area_formed']:,.0f} m² (gaps new: {s['gaps_b_new']}, "
                 f"expanded: {s['gaps_b_expanded']})",
                 f"  closed   {s['area_closed']:,.0f} m² (gaps closed: {s['gaps_a_closed']}, "
                 f"shrunk: {s['gaps_a_shrunk']})",
                 f"  uncertain {s['area_uncertain']:,.0f} m² crossed the threshold without "
                 "a significant change"]
        if "formation_rate" in s:
            lines.append(f"  rates    formation {s['formation_rate'] * 100:.2f} %/yr, closure "
                         f"{s['closure_rate'] * 100:.2f} %/yr of the area compared")
        return "\n".join(lines)

    def to_geojson(self, path, survey: str = "b") -> None:
        """Write one survey's gaps with their status and area formed or
        closed.

        Parameters
        ----------
        path
            Output file.
        survey : {"a", "b"}
            Which gaps.
        """
        if survey == "a":
            self.a.to_geojson(path, {"status": self.status_a, "closed_area": self.closed_area})
        elif survey == "b":
            self.b.to_geojson(path, {"status": self.status_b, "formed_area": self.formed_area})
        else:
            raise ValueError(f"survey must be 'a' or 'b', got {survey!r}")


def gap_change(change, chm_b: Raster | None = None, height: float = 2.0, min_area: float = 10.0,
               max_area: float | None = None, connectivity: int = 8,
               years: float | None = None) -> GapChange:
    """Gap formation and closure between two surveys.

    Gaps are found in each CHM as :func:`canopy_gaps` finds them. A cell
    *forms* a gap when it is in a gap of the second survey only and its CHM
    fell significantly, and *closes* one when it is in a gap of the first
    only and its CHM rose significantly (as ForestGapR's ``GapChangeDec``,
    Silva et al. 2019, but with the level of detection of
    :func:`surface_change`); a cell that crossed the threshold without a
    significant change is ``uncertain``. Each gap of the second survey is
    ``new``, ``expanded``, ``stable`` or ``uncertain``, each of the first
    ``closed``, ``shrunk``, ``stable`` or ``uncertain``.

    Parameters
    ----------
    change
        A CHM :class:`SurfaceChange`, or the first survey's CHM (a
        :class:`~sylva.Raster`) with ``chm_b``; without a surface change
        every transition counts.
    chm_b
        The second survey's CHM, on the grid of the first.
    height, min_area, max_area, connectivity
        As for :func:`canopy_gaps`.
    years
        Time between the surveys, for annual rates.

    Returns
    -------
    GapChange
    """
    if isinstance(change, SurfaceChange):
        if change.surface != "chm":
            raise ValueError(f"gap change needs a CHM change, got a {change.surface!r} change")
        ra, rb, cls = change.a, change.b, np.ascontiguousarray(change.classes, dtype=np.uint8)
    elif isinstance(change, Raster) and isinstance(chm_b, Raster):
        ra, rb, cls = change, chm_b, None
    else:
        raise ValueError("give a SurfaceChange, or two CHMs (Raster) on one grid")
    if years is not None:
        years = _pos("years", years)
    s = _gap_settings(height, min_area, max_area, connectivity)
    d = _core.change_als_gap_change(_raster_arg(ra, "chm_a"), _raster_arg(rb, "chm_b"), cls,
                                    s["height"], s["min_area"], _inf(s["max_area"]),
                                    s["connectivity"])
    return GapChange(Gaps._from_core(d["a"], ra, s), Gaps._from_core(d["b"], rb, s),
                     np.asarray(d["cells"]), list(d["status_a"]), np.asarray(d["closed_area"]),
                     list(d["status_b"]), np.asarray(d["formed_area"]),
                     dict(zip(d["cell_names"], d["areas"], strict=True)), years)


# ------------------------------------------------------------------ trees

_TREE_COLUMNS = ("id_a", "id_b", "status", "x", "y", "distance", "height_a", "height_b", "dh",
                 "sigma", "lod", "dh_change", "crown_area_a", "crown_area_b", "crown_loss",
                 "crown_gain", "observed")


@dataclass
class ALSTreeChange:
    """Trees of two airborne surveys compared, from :func:`tree_change`.

    ``table`` has one row per tree of the first survey (with its partner in
    the second, if any) and one per tree found only in the second:

    ``id_a``, ``id_b``
        Ids in each survey (0 for none).
    ``status``
        ``survivor``, ``damaged``, ``dead``, ``undetected`` or
        ``unobserved`` for a tree of the first survey; ``recruit``,
        ``released``, ``undetected`` or ``unobserved`` for one found only
        in the second.
    ``x``, ``y``
        Top in the first survey's frame.
    ``distance``
        Between the partners' tops (m).
    ``height_a``, ``height_b``, ``dh``, ``sigma``, ``lod``
        Heights (m), height change, its standard deviation and level of
        detection.
    ``dh_change``
        ``growth``, ``decrease``, ``below_detection`` or ``unmeasured``
        (empty without a partner).
    ``crown_area_a``, ``crown_area_b``
        Crown areas (m²).
    ``crown_loss``, ``crown_gain``
        Shares of the crown with a significant canopy loss or gain.
    ``observed``
        Share of the crown with data in both surveys.

    Attributes
    ----------
    table
        The columns above, as NumPy arrays.
    crs
        Coordinate system of the positions.
    settings
        The settings of the call.
    """

    table: dict
    crs: str | None = None
    settings: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.table["id_a"])

    def __repr__(self) -> str:
        c = self.counts()
        return "ALSTreeChange(" + ", ".join(f"{k}={v}" for k, v in c.items()) + ")"

    def _core(self) -> dict:
        t = self.table
        out = {k: np.ascontiguousarray(t[k], dtype=np.float64) for k in _TREE_COLUMNS
               if k not in ("id_a", "id_b", "status", "dh_change")}
        out["id_a"] = np.ascontiguousarray(t["id_a"], dtype=np.int64)
        out["id_b"] = np.ascontiguousarray(t["id_b"], dtype=np.int64)
        out["status"] = [str(s) for s in t["status"]]
        out["dh_change"] = [str(s) for s in t["dh_change"]]
        return out

    def counts(self) -> dict[str, int]:
        """Rows per status (``undetected`` and ``unobserved`` split by the
        survey the tree was found in)."""
        out: dict[str, int] = {}
        for s, a in zip(self.table["status"], self.table["id_a"], strict=True):
            key = s if s not in ("undetected", "unobserved") else f"{s}_{'a' if a else 'b'}"
            out[key] = out.get(key, 0) + 1
        return out

    def summary(self, area: float | None = None, years: float | None = None, mask=None) -> dict:
        """Totals: trees by fate, mean height growth of the survivors with
        its standard error (from their scatter, and from their measurement
        uncertainties alone), crown area lost and, with ``years``, annual
        mortality and recruitment rates (Sheil et al. 1995). With ``area``
        (m²), counts and crown areas are also given per hectare.

        Parameters
        ----------
        area
            Area covered (m²).
        years
            Time between the surveys.
        mask
            Rows to include (bool array), e.g. the trees in a plot.

        Returns
        -------
        dict
        """
        m = None if mask is None else np.ascontiguousarray(mask, dtype=bool)
        if years is not None:
            years = _pos("years", years)
        s = dict(_core.change_als_tree_summary(self._core(), m, years))
        if area is not None:
            ha = _pos("area", area) / 1e4
            for k in ("survivors", "damaged", "dead", "recruits", "released"):
                s[k + "_per_ha"] = s[k] / ha
            s["crown_area_lost_per_ha"] = (s["crown_area_dead"] + s["crown_area_damaged"]) / ha
        return s

    def grid(self, resolution: float, bounds=None) -> dict[str, Raster]:
        """Totals per grid cell: ``survivors``, ``damaged``, ``dead``,
        ``recruits``, ``mean_growth``, ``growth_se`` and
        ``crown_area_lost`` (m²).

        Parameters
        ----------
        resolution
            Cell size (m).
        bounds
            ``(xmin, ymin, xmax, ymax)``; the trees' extent snapped to
            multiples of ``resolution`` by default.

        Returns
        -------
        dict of Raster
        """
        res = _pos("resolution", resolution)
        x, y = np.asarray(self.table["x"]), np.asarray(self.table["y"])
        if bounds is None:
            if len(x) == 0:
                raise ValueError("no trees to grid")
            bounds = (np.floor(x.min() / res) * res, np.floor(y.min() / res) * res,
                      x.max(), y.max())
        x0, y0, x1, y1 = (float(v) for v in bounds)
        nc = int(np.floor((x1 - x0) / res)) + 1
        nr = int(np.floor((y1 - y0) / res)) + 1
        d = _core.change_als_tree_grid(self._core(), x0, y0, res, nr, nc)
        out = {}
        for k, v in d.items():
            r = Raster._from_core(v)
            r.crs = self.crs
            out[k] = r
        return out

    def to_csv(self, path) -> None:
        """Write the table as CSV.

        Parameters
        ----------
        path
            Output file.
        """
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(_TREE_COLUMNS))
            for k in range(len(self)):
                w.writerow([_cell(self.table[c][k]) for c in _TREE_COLUMNS])

    def to_pandas(self):
        """The table as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame({c: self.table[c] for c in _TREE_COLUMNS})

    def report(self, years: float | None = None) -> str:
        """A summary in words.

        Parameters
        ----------
        years
            Time between the surveys, for annual rates.
        """
        s = self.summary(years=years)
        lines = [f"Trees: {s['survivors']} survivors, {s['damaged']} damaged, {s['dead']} dead, "
                 f"{s['recruits']} recruits, {s['released']} released",
                 f"  not assessed  {s['undetected_a']} + {s['undetected_b']} undetected, "
                 f"{s['unobserved_a']} + {s['unobserved_b']} unobserved (survey a + b)"]
        if s["n_growth"]:
            lines.append(f"  height growth {s['mean_growth']:+.2f} ± {s['growth_se']:.2f} m "
                         f"(mean ± SE over {s['n_growth']} survivors; measurement alone "
                         f"± {s['growth_measurement_se']:.2f}); {s['n_growth_detected']} "
                         "individually above their level of detection")
        if years:
            lines.append(f"  rates         mortality {s['mortality_rate'] * 100:.2f} %/yr, "
                         f"recruitment {s['recruitment_rate'] * 100:.2f} %/yr")
        return "\n".join(lines)


def _tree_dict(trees, name: str) -> dict:
    if hasattr(trees, "table") and hasattr(trees, "crowns"):
        t = {"id": trees.id, "x": trees.x, "y": trees.y, "height": trees.height,
             "crown_area": trees.crown_area, "crowns": trees.crowns}
    elif isinstance(trees, dict):
        missing = [k for k in ("x", "y", "height") if k not in trees]
        if missing:
            raise ValueError(f"{name} is missing {missing}")
        t = dict(trees)
    else:
        raise ValueError(f"{name} must be sylva.als.Trees or a dict of columns, got "
                         f"{type(trees).__name__}")
    n = len(np.asarray(t["x"]))
    out = {"id": np.ascontiguousarray(t.get("id", np.arange(1, n + 1)), dtype=np.int64),
           "x": np.ascontiguousarray(t["x"], dtype=np.float64),
           "y": np.ascontiguousarray(t["y"], dtype=np.float64),
           "height": np.ascontiguousarray(t["height"], dtype=np.float64),
           "crown_area": np.ascontiguousarray(t.get("crown_area", np.full(n, np.nan)),
                                              dtype=np.float64)}
    crowns = t.get("crowns") or [np.zeros((0, 2))] * n
    out["crowns"] = [np.ascontiguousarray(np.asarray(c, dtype=np.float64).reshape(-1, 2))
                     for c in crowns]
    if any(len(v) != n for k, v in out.items()):
        raise ValueError(f"the columns of {name} differ in length")
    if not (np.all(np.isfinite(out["x"])) and np.all(np.isfinite(out["y"]))
            and np.all(np.isfinite(out["height"]))):
        raise ValueError(f"{name} has non-finite positions or heights")
    return out


def tree_change(trees_a, trees_b, change: SurfaceChange, alignment: ALSAlignment | None = None,
                max_distance: float = 1.5, max_growth: float = 0.3, max_drop: float = 0.2,
                height_weight: float = 1.0, dead_fraction: float = 0.5,
                damage_fraction: float = 0.3, min_observed: float = 0.5,
                confidence: float = 0.95) -> ALSTreeChange:
    """Airborne trees of two surveys matched, with height growth, mortality,
    damage and recruitment.

    The trees are matched by the optimal assignment of
    :func:`sylva.change.match_trees` (Kuhn 1955; Munkres 1957) with the tree
    height in place of the diameter: a pair must lie within
    ``max_distance`` and its height may grow by at most ``max_growth`` or
    fall by at most ``max_drop`` of the larger height. The CHM change says
    what became of the others:

    - a matched tree is a ``survivor``, or ``damaged`` when a significant
      loss covers ``damage_fraction`` of its crown or its height fell by
      more than its level of detection;
    - an unmatched tree of the first survey is ``damaged`` when a tree of
      the second stands within ``2 * max_distance`` of its top on
      significantly lowered canopy (a broken or collapsed crown), ``dead``
      when a significant loss covers ``dead_fraction`` of its crown,
      ``unobserved`` when less than ``min_observed`` of its crown has data,
      and otherwise ``undetected`` (its canopy is still there);
    - an unmatched tree of the second survey is a ``recruit`` when the
      canopy at its top rose significantly from below ``1 - max_growth`` of
      its height (more than an existing crown can grow), ``released`` when
      it fell (an understorey tree exposed by the loss of a neighbour),
      ``unobserved`` without data, and otherwise ``undetected`` (no
      significant change, or a rise a growing crown explains: a top split
      off a neighbour's crown).

    The standard deviation of a height is that of its top's CHM cell in
    ``change`` (noise, sampling of the apex, DTM); the level of detection of
    a height change is the normal quantile of ``confidence`` times the
    standard deviation of the difference.

    Parameters
    ----------
    trees_a, trees_b
        The trees of each survey, from :func:`sylva.als.find_trees` (or a
        dict with ``x``, ``y``, ``height`` and optionally ``id``,
        ``crown_area`` and ``crowns``), each in its own survey's frame.
    change
        The CHM change of the same surveys (:func:`chm_change`), best at
        the resolution the trees were found at.
    alignment
        Moves the second survey's trees into the first's frame (the one
        ``change`` was made with).
    max_distance, max_growth, max_drop, height_weight
        Matching: largest distance (m), largest relative height gain and
        loss, and the weight of the squared relative height difference
        next to the squared distance over ``max_distance``.
    dead_fraction, damage_fraction, min_observed
        Shares of the crown, as above.
    confidence
        Of the levels of detection.

    Returns
    -------
    ALSTreeChange
    """
    if not isinstance(change, SurfaceChange) or change.surface != "chm":
        raise ValueError("change must be a CHM SurfaceChange (from chm_change)")
    ta, tb = _tree_dict(trees_a, "trees_a"), _tree_dict(trees_b, "trees_b")
    settings = dict(max_distance=_pos("max_distance", max_distance),
                    max_growth=_pos("max_growth", max_growth),
                    max_drop=_share("max_drop", max_drop),
                    height_weight=_num("height_weight", height_weight, 0.0),
                    dead_fraction=_share("dead_fraction", dead_fraction),
                    damage_fraction=_share("damage_fraction", damage_fraction),
                    min_observed=_share("min_observed", min_observed),
                    confidence=_confidence(confidence))
    if settings["max_drop"] >= 1:
        raise ValueError("max_drop must be less than 1")
    s = settings
    d = _core.change_als_trees(ta, tb, _raster_arg(change.a, "chm_a"),
                               _raster_arg(change.b, "chm_b"),
                               _raster_arg(change.sigma_a, "sigma_a"),
                               _raster_arg(change.sigma_b, "sigma_b"),
                               np.ascontiguousarray(change.classes, dtype=np.uint8),
                               _alignment(alignment), s["max_distance"], s["max_growth"],
                               s["max_drop"], s["height_weight"], s["confidence"],
                               s["dead_fraction"], s["damage_fraction"], s["min_observed"])
    table = {k: np.asarray(d[k]) for k in _TREE_COLUMNS if k not in ("status", "dh_change")}
    table["status"] = np.asarray(d["status"], dtype="U10")
    table["dh_change"] = np.asarray(d["dh_change"], dtype="U15")
    return ALSTreeChange(table, change.a.crs, settings)


# ------------------------------------------------------------------ metrics

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


# ------------------------------------------------------------------ PAI

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


# ------------------------------------------------------------------ command line

def _cmd_als_change(args, write_raster):
    from .. import als
    a = als.catalog(args.a, pattern=args.pattern)
    b = als.catalog(args.b, pattern=args.pattern)
    run = {"chunk_size": args.chunk_size, "buffer": args.buffer, "workers": args.workers}
    al = None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.align:
        classes = [int(c) for c in args.stable_classes.split(",")]
        al = align_surveys(a, b, block_size=args.block_size, stable_classes=classes,
                           model=args.model, workers=args.workers)
        al.to_csv(out / "alignment.csv")
        print(al.report())
    ch = surface_change(a, b, args.surface, args.resolution, alignment=al,
                        harmonise=args.harmonise, density_cell=args.density_cell,
                        first_returns=args.first_returns, confidence=args.confidence, **run)
    ext = "tif" if args.format == "tif" else "asc"
    for name, r in (("a", ch.a), ("b", ch.b), ("difference", ch.difference), ("lod", ch.lod)):
        write_raster(r, str(out / f"{args.surface}_{name}.{ext}"))
    write_raster(Raster(ch.classes.astype(float), ch.a.xmin, ch.a.ymin, ch.a.resolution, ch.a.crs),
                 str(out / f"{args.surface}_classes.{ext}"))
    print(ch.report())
    if args.gaps and args.surface == "chm":
        g = gap_change(ch, height=args.gap_height, min_area=args.gap_min_area, years=args.years)
        g.to_geojson(out / "gaps_a.geojson", "a")
        g.to_geojson(out / "gaps_b.geojson", "b")
        print(g.report())
    print(f"-> {out}")


def _cmd_als_tree_change(args, write_raster):
    import tempfile

    from .. import als
    a = als.catalog(args.a, pattern=args.pattern)
    b = als.catalog(args.b, pattern=args.pattern)
    run = {"chunk_size": args.chunk_size, "buffer": args.buffer, "workers": args.workers}
    al = None
    if args.align:
        classes = [int(c) for c in args.stable_classes.split(",")]
        al = align_surveys(a, b, block_size=args.block_size, stable_classes=classes,
                           workers=args.workers)
        print(al.report())
    window = als.LinearWindow(*args.window_linear) if args.window_linear else args.window
    kw = dict(method=args.method, resolution=args.resolution, window=window, hmin=args.hmin,
              max_cr=args.max_cr, **run)
    with tempfile.TemporaryDirectory() as tmp:
        ta_cat, tb_cat = a, b
        if args.harmonise:
            ta_cat, tb_cat = harmonise(a, b, Path(tmp) / "a", Path(tmp) / "b",
                                       density_cell=args.density_cell, workers=args.workers)
        ta = als.find_trees(ta_cat, **kw)
        tb = als.find_trees(tb_cat, **kw)
    ch = chm_change(a, b, args.resolution, alignment=al, harmonise=args.harmonise,
                    density_cell=args.density_cell, confidence=args.confidence, **run)
    tc = tree_change(ta, tb, ch, alignment=al, max_distance=args.max_distance,
                     confidence=args.confidence)
    tc.to_csv(args.output)
    print(tc.report(years=args.years))
    if args.grid:
        g = tc.grid(args.grid_resolution)
        d = Path(args.grid)
        d.mkdir(parents=True, exist_ok=True)
        for k, r in g.items():
            write_raster(r, str(d / f"{k}.asc"))
    print(f"{len(ta)} and {len(tb)} trees compared -> {args.output}")


def _add_commands(sub, fmt, common, write_raster) -> None:
    """Add ``als-change`` and ``als-tree-change`` to the command line."""
    s = sub.add_parser("als-change", help="CHM, DSM or DTM change between two ALS surveys, "
                       "with a level of detection per cell and gap dynamics", **fmt)
    s.add_argument("a", help="directory of the earlier survey's tiles (ground classified)")
    s.add_argument("b", help="directory of the later survey's tiles (ground classified)")
    s.add_argument("output", help="directory for the rasters, gap polygons and alignment")
    s.add_argument("--surface", choices=["chm", "dsm", "dtm"], default="chm",
                   help="what to compare")
    s.add_argument("--resolution", type=float, default=1.0, help="cell size (m)")
    s.add_argument("--align", action="store_true",
                   help="estimate the offsets between the surveys on stable surfaces first")
    s.add_argument("--stable-classes", default="2",
                   help="comma-separated classes of stable returns (2 ground, 6 roofs, 11 roads)")
    s.add_argument("--block-size", type=float, default=100.0, help="alignment block size (m)")
    s.add_argument("--model", choices=["field", "blocks", "constant"], default="field",
                   help="alignment model")
    s.add_argument("--harmonise", action="store_true",
                   help="thin the denser survey to the other's pulse density")
    s.add_argument("--density-cell", type=float, default=10.0,
                   help="square in which pulse densities are compared (m)")
    s.add_argument("--first-returns", action="store_true", help="grid first returns only")
    s.add_argument("--confidence", type=float, default=0.95, help="of the level of detection")
    s.add_argument("--gaps", action="store_true", help="also find gaps and their change (CHM)")
    s.add_argument("--gap-height", type=float, default=2.0, help="gap height threshold (m)")
    s.add_argument("--gap-min-area", type=float, default=10.0, help="smallest gap (m²)")
    s.add_argument("--years", type=float, default=None, help="time between the surveys")
    s.add_argument("--format", choices=["asc", "tif"], default="asc",
                   help="raster format (tif needs rasterio)")
    common(s)
    s.set_defaults(func=lambda args: _cmd_als_change(args, write_raster))

    s = sub.add_parser("als-tree-change", help="trees of two ALS surveys matched: growth, "
                       "mortality, damage and recruitment", **fmt)
    s.add_argument("a", help="directory of the earlier survey's tiles (ground classified)")
    s.add_argument("b", help="directory of the later survey's tiles (ground classified)")
    s.add_argument("output", help="CSV with one row per tree")
    s.add_argument("--resolution", type=float, default=0.5, help="CHM cell size (m)")
    s.add_argument("--method", choices=["dalponte2016", "watershed", "li2012"],
                   default="dalponte2016", help="crown segmentation")
    s.add_argument("--window", type=float, default=5.0, help="local maximum window diameter (m)")
    s.add_argument("--window-linear", type=float, nargs=4,
                   metavar=("INTERCEPT", "SLOPE", "MIN", "MAX"),
                   help="window growing with height h: clip(INTERCEPT + SLOPE h, MIN, MAX)")
    s.add_argument("--hmin", type=float, default=2.0, help="lowest tree top (m)")
    s.add_argument("--max-cr", type=float, default=10.0,
                   help="dalponte2016 crown extent from the top (cells)")
    s.add_argument("--max-distance", type=float, default=1.5,
                   help="farthest apart the tops of one tree can be (m)")
    s.add_argument("--align", action="store_true", help="estimate the offsets first")
    s.add_argument("--stable-classes", default="2", help="comma-separated stable classes")
    s.add_argument("--block-size", type=float, default=100.0, help="alignment block size (m)")
    s.add_argument("--harmonise", action="store_true",
                   help="thin the denser survey to the other's pulse density before comparing")
    s.add_argument("--density-cell", type=float, default=10.0,
                   help="square in which pulse densities are compared (m)")
    s.add_argument("--confidence", type=float, default=0.95, help="of the levels of detection")
    s.add_argument("--years", type=float, default=None, help="time between the surveys")
    s.add_argument("--grid", help="also write per-cell totals as rasters into this directory")
    s.add_argument("--grid-resolution", type=float, default=50.0, help="cell size of --grid (m)")
    common(s, buffer=30.0)
    s.set_defaults(func=lambda args: _cmd_als_tree_change(args, write_raster))
