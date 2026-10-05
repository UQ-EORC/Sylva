# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Merging terrestrial and airborne clouds and canopy profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..raster import Raster
from ._common import _matrix


def merge_clouds(tls: PointCloud, als: PointCloud, dtm: Raster, split=5.0, transform=None,
                 fallback: float | None = None) -> PointCloud:
    """One cloud from both instruments: TLS below the split, ALS above.

    Parameters
    ----------
    tls
        The TLS points (in the ALS frame, or moved there by ``transform``).
    als
        The ALS returns.
    dtm
        Terrain for the heights of both, in the ALS frame.
    split
        Height above ground (m) where the ALS takes over: a number, a
        :class:`~sylva.Raster` of heights (one per column; NaN cells and
        points off the grid take ``fallback``), or a :class:`FusedProfile`,
        whose :attr:`FusedProfile.split_height` is used.
    transform
        :class:`Registration` or ``(4, 4)`` matrix applied to ``tls`` first.
    fallback
        Split height where a split raster has no value; the raster's median
        if None.

    Returns
    -------
    PointCloud
        TLS points lower than the split, then ALS returns at or above it,
        with ``source`` (uint8: 1 TLS, 2 ALS), ``height`` above the DTM and
        the attributes both clouds have. The CRS is the ALS cloud's.

    Raises
    ------
    ValueError
        For a bad split or transform.
    """
    m = _matrix(transform)
    if m is not None:
        tls = tls.transform(m)
    kw: dict = {"split_height": None, "split_raster": None, "fallback": 0.0}
    if isinstance(split, FusedProfile):
        split = split.split_height
    if isinstance(split, Raster):
        fb = float(np.nanmedian(split.data)) if fallback is None else float(fallback)
        kw["split_raster"] = (np.ascontiguousarray(split.data), float(split.xmin), float(split.ymin),
                              float(split.resolution))
        kw["fallback"] = fb
    else:
        kw["split_height"] = float(split)

    def heights(c: PointCloud) -> np.ndarray:
        return np.ascontiguousarray(_core.raster_heights_above(c.xyz, dtm.data, float(dtm.xmin),
                                                               float(dtm.ymin), float(dtm.resolution)))
    ht, ha = heights(tls), heights(als)
    kt = _core.fusion_select(np.ascontiguousarray(tls.xyz), ht, below=True, **kw)
    ka = _core.fusion_select(np.ascontiguousarray(als.xyz), ha, below=False, **kw)
    common = sorted(set(tls.attrs) & set(als.attrs) - {"source", "height"})
    attrs = {k: np.concatenate([np.asarray(tls.attrs[k])[kt], np.asarray(als.attrs[k])[ka]]) for k in common}
    attrs["source"] = np.concatenate([np.full(int(kt.sum()), 1, np.uint8), np.full(int(ka.sum()), 2, np.uint8)])
    attrs["height"] = np.concatenate([ht[kt], ha[ka]])
    return PointCloud(np.vstack([tls.xyz[kt], als.xyz[ka]]), attrs, als.crs if als.crs is not None else tls.crs)


@dataclass
class FusedProfile:
    """A plant area density profile from both instruments, from
    :func:`fuse_profiles`.

    Attributes
    ----------
    height
        Bottom of each height bin above ground (m).
    pad
        Fused plant area density (m² m⁻³); NaN where neither saw the bin.
    weight_tls, weight_als
        Weight of each instrument in each bin (summing to 1).
    tls, als
        Each instrument's bins: ``pad``, ``beams`` (mean pulses entering a
        voxel of the bin, unreached voxels counting 0), ``observed`` (share
        of the bin's voxels with at least ``min_beams`` pulses) and
        ``n_voxels``.
    split_height
        Lowest height above which the ALS carries at least half the weight
        in every bin: where :func:`merge_clouds` can switch instruments.
    bin_size
        Bin height (m).
    settings
        The settings of the call.
    """

    height: np.ndarray
    pad: np.ndarray
    weight_tls: np.ndarray
    weight_als: np.ndarray
    tls: dict
    als: dict
    split_height: float
    bin_size: float
    settings: dict = field(default_factory=dict)

    def pai(self, source: str = "fused") -> float:
        """Plant area index: the sum of the PAD of the bins times their height.

        Parameters
        ----------
        source : {"fused", "tls", "als"}
            Which profile. Bins without a value count as 0.

        Returns
        -------
        float
        """
        pad = {"fused": self.pad, "tls": self.tls["pad"], "als": self.als["pad"]}.get(source)
        if pad is None:
            raise ValueError(f"source must be 'fused', 'tls' or 'als', got {source!r}")
        return float(np.nansum(pad) * self.bin_size)

    def table(self) -> dict:
        """The profile as columns: ``height``, ``pad``, ``weight_tls``,
        ``weight_als``, ``pad_tls``, ``pad_als``, ``beams_tls``,
        ``beams_als``, ``observed_tls``, ``observed_als``."""
        return {"height": self.height, "pad": self.pad, "weight_tls": self.weight_tls,
                "weight_als": self.weight_als, "pad_tls": self.tls["pad"], "pad_als": self.als["pad"],
                "beams_tls": self.tls["beams"], "beams_als": self.als["beams"],
                "observed_tls": self.tls["observed"], "observed_als": self.als["observed"]}


def _voxel_arrays(grid, names: list[str]) -> tuple:
    """Origin, voxel size and ``(nz, ny, nx)`` arrays of a RayVoxelGrid or ALSVoxels."""
    if not hasattr(grid, "voxel_size") or not hasattr(grid, "origin"):
        raise ValueError(f"expected a voxels.RayVoxelGrid or als.ALSVoxels, got {type(grid).__name__}")
    out = []
    for n in names:
        try:
            out.append(np.ascontiguousarray(grid[n], dtype=np.float64))
        except (KeyError, ValueError) as e:
            raise ValueError(f"the grid has no {n!r} field; ray-trace it with that field") from e
    o = np.asarray(grid.origin, dtype=float).reshape(3)
    return (float(o[0]), float(o[1]), float(o[2])), float(grid.voxel_size), out


def _area_mask(area, origin, voxel_size, shape) -> np.ndarray | None:
    if area is None:
        return None
    ny, nx = shape
    xc = origin[0] + (np.arange(nx) + 0.5) * voxel_size
    yc = origin[1] + (np.arange(ny) + 0.5) * voxel_size
    X, Y = np.meshgrid(xc, yc)
    a = np.asarray(area, dtype=float)
    if a.shape == (4,):
        return (X >= a[0]) & (X < a[2]) & (Y >= a[1]) & (Y < a[3])
    if a.shape == (3,):
        return np.hypot(X - a[0], Y - a[1]) <= a[2]
    if a.ndim == 2 and a.shape[1] == 2 and len(a) >= 3:
        inside = np.zeros(X.shape, bool)
        for (x0, y0), (x1, y1) in zip(a, np.roll(a, -1, axis=0), strict=True):
            crosses = (y0 > Y) != (y1 > Y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = x0 + (Y - y0) / (y1 - y0) * (x1 - x0)
            inside ^= crosses & (X < xc)
        return inside
    raise ValueError("area must be (xmin, ymin, xmax, ymax), (x, y, radius) or a (k, 2) polygon")


def fuse_profiles(tls_grid, als_grid, dtm: Raster | None = None, area=None, *,
                  bin_size: float | None = None, min_beams: float = 5.0, estimator: str = "mean",
                  g: float = 0.5, pad_field: str | None = None, mode: str = "beams",
                  min_observed: float = 0.0) -> FusedProfile:
    """One plant area density profile from TLS and ALS voxels.

    Each grid is summarised per bin of height above ``dtm`` over ``area``:
    PAD, the mean number of pulses entering a voxel (voxels no pulse
    reached count 0, so occlusion lowers it) and the share of voxels crossed
    by at least ``min_beams`` pulses. Each bin then takes its PAD from both
    instruments, weighted by

    - ``"beams"`` (the default): the pulses, since the variance of a
      gap-fraction estimate falls in proportion to the beams that sampled
      it. Near the ground the TLS dominates; high in a closed canopy, where
      its pulses have been stopped, the ALS takes over;
    - ``"observed"``: the share of the bin each instrument saw;
    - ``"best"``: the one with more pulses alone.

    An instrument whose bin is seen less than ``min_observed`` gets no
    weight there. The weights are reported per bin.

    Both grids must be in one frame (move the TLS shots with
    :meth:`sylva.Shots.transform` and a :class:`Registration` before
    voxelising) and should share the voxel size.

    Parameters
    ----------
    tls_grid
        :class:`sylva.voxels.RayVoxelGrid` of the TLS.
    als_grid
        :class:`sylva.als.ALSVoxels` (or a RayVoxelGrid) of the ALS.
    dtm
        Terrain (e.g. the ALS DTM); heights above the grid floor if None.
    area
        ``(xmin, ymin, xmax, ymax)``, ``(x, y, radius)`` or a ``(k, 2)``
        polygon: the columns
        (by their centres) to summarise; every column of each grid if None.
        Use the plot, so that both describe the same ground.
    bin_size
        Height bin (m); the larger voxel size if None.
    min_beams
        Pulses a voxel needs to count as observed; voxels crossed by fewer
        give noisy estimates that are biased upwards.
    estimator : {"mean", "pooled"}
        ``"mean"``: the mean of ``pad_field`` over the observed voxels of
        the bin, each voxel's own estimate, so a crown and the gap beside it
        count by their volume. ``"pooled"``: ``Σ num_hits_weighted / (g Σ
        free_path_length)`` over the observed voxels, a ratio of sums that
        is not biased upwards by poorly sampled voxels but weights each voxel
        by the pulses that reached it; where the canopy is clumped the
        shadowed crowns get fewer pulses than the gaps and the density reads
        low (30 % on the validation stands), so use it for layers that are
        horizontally uniform.
    g
        Leaf projection G of the pooled estimator (0.5: spherical).
    pad_field
        PAD field of the mean estimator; ``"pad_fpl"`` if None.
    mode : {"beams", "observed", "best"}
        Weighting.
    min_observed
        Share of a bin an instrument must have seen for its PAD to count.

    Returns
    -------
    FusedProfile

    Raises
    ------
    ValueError
        For a grid without the fields asked for, or bad settings.
    """
    if estimator not in ("pooled", "mean"):
        raise ValueError(f"estimator must be 'pooled' or 'mean', got {estimator!r}")
    names = ["num_beams"] + (["num_hits_weighted", "free_path_length"] if estimator == "pooled"
                             else [pad_field or "pad_fpl"])
    stats = []
    sizes = []
    for grid in (tls_grid, als_grid):
        origin, vs, arrs = _voxel_arrays(grid, names)
        sizes.append(vs)
        stats.append((origin, vs, arrs))
    bs = float(max(sizes) if bin_size is None else bin_size)
    d = None if dtm is None else (np.ascontiguousarray(dtm.data, dtype=np.float64), float(dtm.xmin),
                                  float(dtm.ymin), float(dtm.resolution))
    layers = []
    for origin, vs, arrs in stats:
        mask = _area_mask(area, origin, vs, arrs[0].shape[1:])
        if mask is not None and not mask.any():
            raise ValueError("the area holds no column of one of the grids")
        pooled = estimator == "pooled"
        layers.append(_core.fusion_layer_stats(origin, vs, arrs[0], None if pooled else arrs[1],
                                               arrs[1] if pooled else None, arrs[2] if pooled else None,
                                               None if mask is None else np.ascontiguousarray(mask), d, bs,
                                               float(min_beams), float(g)))
    f = _core.fusion_fuse(layers[0], layers[1], bs, str(mode), float(min_observed))
    return FusedProfile(f["height"], f["pad"], f["weight_tls"], f["weight_als"], dict(f["tls"]),
                        dict(f["als"]), float(f["split_height"]), bs,
                        {"area": area, "min_beams": min_beams, "estimator": estimator, "g": g,
                         "pad_field": pad_field, "mode": mode, "min_observed": min_observed})
