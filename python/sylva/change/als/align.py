# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Horizontal and vertical alignment of two airborne surveys."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ... import _core
from ...als.catalogue import _as_catalog, _workers
from ...pointcloud import PointCloud
from ._common import _count, _pos


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
