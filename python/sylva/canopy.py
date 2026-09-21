"""Canopy structure: voxels, plant area density, gap fraction, LAI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .raster import Raster
from .shots import Shots

__all__ = [
    "VoxelGrid", "voxelize", "vertical_profile", "pad_profile_voxel", "gap_fraction_zenith",
    "gap_fraction_pattern", "lai_from_gap_fraction", "canopy_cover", "DensityGrid",
    "density_grid",
]


@dataclass
class VoxelGrid:
    """Point counts per voxel. ``counts[k, j, i]`` indexes z, y, x."""

    counts: np.ndarray
    origin: np.ndarray
    voxel_size: float

    @property
    def occupied(self) -> np.ndarray:
        return self.counts > 0

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(nx, ny, nz)``."""
        return self.counts.shape[::-1]

    def z_levels(self) -> np.ndarray:
        return self.origin[2] + np.arange(self.counts.shape[0]) * self.voxel_size

    def occupied_centers(self) -> np.ndarray:
        k, j, i = np.nonzero(self.occupied)
        return self.origin + (np.column_stack([i, j, k]) + 0.5) * self.voxel_size

    def vertical_profile(self) -> np.ndarray:
        """Fraction of voxels occupied per vertical layer."""
        occ = self.occupied
        return occ.sum(axis=(1, 2)) / (occ.shape[1] * occ.shape[2])


def voxelize(cloud: PointCloud, voxel_size: float, origin=None, shape=None) -> VoxelGrid:
    """Count points per voxel."""
    d = _core.voxelize(cloud.xyz, voxel_size,
                       None if origin is None else tuple(float(v) for v in origin),
                       None if shape is None else tuple(int(v) for v in shape))
    return VoxelGrid(d["counts"], d["origin"], d["voxel_size"])


def vertical_profile(cloud: PointCloud, bin_size: float = 0.5, height_attr: str = "height",
                     max_height: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of point counts by height. Returns ``(bin_bottoms, counts)``."""
    h = cloud.heights(height_attr)
    top = max_height if max_height is not None else np.nanmax(h)
    edges = np.arange(0, top + bin_size, bin_size)
    counts, _ = np.histogram(h, bins=edges)
    return edges[:-1], counts


def pad_profile_voxel(cloud: PointCloud, voxel_size: float = 0.5, height_attr: str = "height",
                      max_height: float | None = None,
                      clumping: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Plant area density profile by the vertical contact-frequency method
    (simplified Hosoi & Omasa 2006, vertical beams, G = 0.5):
    ``PAD_i = -ln(1 - N_i) / (G dz)``. Returns ``(layer_bottoms, pad)``.

    For a rigorous path-length model use :func:`density_grid` with pulse data.
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    return _core.pad_profile_voxel(cloud.xyz, h, voxel_size, max_height, clumping)


def gap_fraction_zenith(shots: Shots, echo_heights: np.ndarray, min_height: float = 0.0,
                        zenith_edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Directional gap fraction P(theta) by zenith ring: a shot is a gap when
    none of its echoes is above ``min_height``. Returns ``(centres_deg, gap)``."""
    edges = None if zenith_edges is None else np.ascontiguousarray(zenith_edges, dtype=float)
    return _core.gap_fraction_zenith(shots._to_core(),
                                     np.ascontiguousarray(echo_heights, dtype=float),
                                     min_height, edges)


def gap_fraction_pattern(shots: Shots, echo_heights: np.ndarray, pattern: dict,
                         min_height: float = 0.0, zenith_edges: np.ndarray | None = None,
                         pulses_per_line: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Gap fraction by zenith ring for a RIEGL scan whose no-return pulses are
    absent from the stream (RiVLib). The number of pulses fired per ring comes
    from the angular scan ``pattern`` (see :meth:`Shots.expected_per_zenith`);
    a fired pulse with no echo above ``min_height`` is a gap. Pass the shots
    in the scanner frame, or ``pulses_per_line`` from the scanner-frame shots.

    Returns ``(centres_deg, gap)``; rings outside the scanned zenith range are NaN.
    """
    edges = np.arange(0, 95, 5.0) if zenith_edges is None else np.asarray(zenith_edges, float)
    if pulses_per_line is None:
        pulses_per_line = shots.pulses_per_line(pattern)
    expected = shots.expected_per_zenith(pattern, edges, pulses_per_line).astype(float)
    zen, _ = shots.zenith_azimuth()
    h = np.asarray(echo_heights, dtype=float)
    hit_shot = np.zeros(shots.n_shots, dtype=bool)
    np.logical_or.at(hit_shot, shots.shot_of_echo(), h > min_height)
    hits, _ = np.histogram(zen[hit_shot], bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = 1.0 - np.minimum(hits / expected, 1.0)
    gap[expected == 0] = np.nan
    return 0.5 * (edges[:-1] + edges[1:]), gap


def lai_from_gap_fraction(zenith_deg, gap_fraction, method: str = "hinge") -> float:
    """Effective PAI from directional gap fraction.

    ``"hinge"``: ``-ln P(57.5°) cos(57.5°) / 0.5`` (Wilson 1963).
    ``"miller"``: ``2 ∫ -ln P(θ) cos θ sin θ dθ`` (Miller 1967).

    NaN rings are skipped; a gap fraction of 0 (saturated ring, common in
    dense forest) is floored at 1e-5, so the result is large but finite.
    """
    return _core.lai_from_gap_fraction(np.ascontiguousarray(zenith_deg, dtype=float),
                                       np.ascontiguousarray(gap_fraction, dtype=float), method)


def canopy_cover(chm_data: np.ndarray, threshold: float = 2.0) -> float:
    """Fraction of CHM cells at or above ``threshold`` (m)."""
    return _core.canopy_cover(np.ascontiguousarray(chm_data, dtype=float), threshold)


@dataclass
class DensityGrid:
    """Ray-traced voxel statistics (raycloudtools / Lowe et al. 2020 style).

    Arrays are ``[k, j, i]`` (z, y, x). ``density`` is
    ``2 (n-1)/n · hits / path_length`` per voxel (spherical leaf angles),
    NaN where fewer than ``min_hits`` hits.
    """

    n_rays: np.ndarray
    n_hits: np.ndarray
    path_length: np.ndarray
    density: np.ndarray
    profile: np.ndarray
    origin: np.ndarray
    voxel_size: float

    def z_levels(self) -> np.ndarray:
        return self.origin[2] + np.arange(self.n_rays.shape[0]) * self.voxel_size

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxel-centre coordinate grids ``(X, Y, Z)``, each shaped like ``density``."""
        nz, ny, nx = self.density.shape
        xs = self.origin[0] + (np.arange(nx) + 0.5) * self.voxel_size
        ys = self.origin[1] + (np.arange(ny) + 0.5) * self.voxel_size
        zs = self.origin[2] + (np.arange(nz) + 0.5) * self.voxel_size
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        return X, Y, Z

    def height_above(self, dtm: Raster) -> np.ndarray:
        """Height of each voxel centre above the DTM, shaped like ``density``."""
        X, Y, Z = self.centers()
        return Z - dtm.sample(X.ravel(), Y.ravel()).reshape(Z.shape)

    @property
    def pai(self) -> float:
        """Plant area index: column-integrated mean density."""
        return float(np.nansum(self.profile) * self.voxel_size)

    def mask_ground(self, dtm: Raster, margin: float | None = None) -> DensityGrid:
        """Copy with voxels at or below the terrain (centre height < ``margin``,
        default one voxel) set to NaN, so ground returns do not count as plant
        material. Also recomputes ``profile``."""
        margin = self.voxel_size if margin is None else margin
        h = self.height_above(dtm)
        density = np.where(h < margin, np.nan, self.density)
        flat = density.reshape(density.shape[0], -1)
        profile = np.array([np.nanmean(row) if np.isfinite(row).any() else np.nan for row in flat])
        return DensityGrid(self.n_rays, self.n_hits, self.path_length, density, profile,
                           self.origin, self.voxel_size)

    def profile_above_ground(self, dtm: Raster, bin_size: float | None = None,
                             max_height: float | None = None, margin: float | None = None,
                             pooled: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Plant area density by height above the terrain.

        Bins voxel centres by their height above ``dtm`` (default bin =
        voxel size), ignoring voxels whose centre is below ``margin`` (default
        one voxel, which drops the ground-containing layer). With ``pooled=True``
        each bin is treated as one big voxel — ``2 Σhits / Σpath`` over all
        its rays — which weights voxels by how well they were sampled and is
        robust to the sparsely-sampled, occluded voxels of a single scan.
        ``pooled=False`` averages the per-voxel ``density`` instead.
        Returns ``(bin_bottoms, pad)``; integrate with ``bin_size`` for PAI.
        """
        bin_size = self.voxel_size if bin_size is None else bin_size
        margin = self.voxel_size if margin is None else margin
        h = self.height_above(dtm).ravel()
        d = self.density.ravel()
        ok = (h >= margin) & (self.n_rays.ravel() > 0)
        if not pooled:
            ok &= np.isfinite(d)
        top = max_height if max_height is not None else (h[ok].max() if ok.any() else bin_size)
        edges = np.arange(0, top + bin_size, bin_size)
        nb = len(edges) - 1
        idx = np.digitize(h[ok], edges) - 1
        valid = (idx >= 0) & (idx < nb)
        idx = idx[valid]
        with np.errstate(invalid="ignore", divide="ignore"):
            if pooled:
                hits = np.bincount(idx, weights=self.n_hits.ravel()[ok][valid], minlength=nb)
                path = np.bincount(idx, weights=self.path_length.ravel()[ok][valid], minlength=nb)
                pad = np.where(path > 0, 2.0 * hits / path, np.nan)
            else:
                sums = np.bincount(idx, weights=d[ok][valid], minlength=nb)
                counts = np.bincount(idx, minlength=nb)
                pad = sums / counts
        return edges[:-1], pad


def density_grid(shots: Shots, voxel_size: float, origin=None, shape=None,
                 min_hits: int = 2) -> DensityGrid:
    """Trace every shot through a voxel grid (unbounded shots traverse to the
    grid edge) and estimate plant area density per voxel."""
    d = _core.density_grid(shots._to_core(), voxel_size,
                           None if origin is None else tuple(float(v) for v in origin),
                           None if shape is None else tuple(int(v) for v in shape), min_hits)
    return DensityGrid(d["n_rays"], d["n_hits"], d["path_length"], d["density"], d["profile"],
                       d["origin"], d["voxel_size"])
