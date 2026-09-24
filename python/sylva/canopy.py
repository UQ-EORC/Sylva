# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Canopy structure: voxels, plant area density, gap fraction, LAI."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .raster import Raster
from .shots import Shots

__all__ = [
    "VoxelGrid", "voxelize", "vertical_profile", "pad_profile_voxel", "gap_fraction_zenith",
    "gap_fraction_pattern", "lai_from_gap_fraction", "canopy_cover", "DensityGrid",
    "density_grid", "fit_ground_plane", "fired_pulses_per_ring", "GapProfile",
]


@dataclass
class VoxelGrid:
    """Point counts on a regular 3D grid; build with :func:`voxelize`.

    Attributes
    ----------
    counts
        ``(nz, ny, nx)`` integer counts; ``counts[k, j, i]`` indexes z, y, x.
    origin
        Minimum corner of the grid, ``(x, y, z)``.
    voxel_size
        Voxel edge length (m).

    Notes
    -----
    Occupancy is not density: an empty voxel may simply not have been seen.
    For plant area density use the ray-traced :func:`sylva.voxels.ray_voxelize`.
    """

    counts: np.ndarray
    origin: np.ndarray
    voxel_size: float

    @property
    def occupied(self) -> np.ndarray:
        """Boolean ``(nz, ny, nx)``, True where a voxel holds at least one point."""
        return self.counts > 0

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid size as ``(nx, ny, nz)`` (the reverse of ``counts.shape``)."""
        return self.counts.shape[::-1]

    def z_levels(self) -> np.ndarray:
        """Bottom z of each voxel layer.

        Returns
        -------
        numpy.ndarray
            Length ``nz``, bottom first (m).
        """
        return self.origin[2] + np.arange(self.counts.shape[0]) * self.voxel_size

    def occupied_centers(self) -> np.ndarray:
        """Centres of the occupied voxels, e.g. for plotting.

        Returns
        -------
        numpy.ndarray
            ``(n, 3)`` x, y, z of each voxel with at least one point.
        """
        k, j, i = np.nonzero(self.occupied)
        return self.origin + (np.column_stack([i, j, k]) + 0.5) * self.voxel_size

    def vertical_profile(self) -> np.ndarray:
        """Fraction of voxels occupied in each layer.

        Returns
        -------
        numpy.ndarray
            Length ``nz``, bottom first, values 0 to 1.
        """
        occ = self.occupied
        return occ.sum(axis=(1, 2)) / (occ.shape[1] * occ.shape[2])


def voxelize(cloud: PointCloud, voxel_size: float, origin=None, shape=None) -> VoxelGrid:
    """Count points per voxel.

    Parameters
    ----------
    cloud
        Input points.
    voxel_size
        Voxel edge length (m).
    origin
        Minimum corner ``(x, y, z)``; the cloud's minimum if None. Fix it
        (with ``shape``) to compare grids across dates or plots.
    shape
        ``(nx, ny, nz)``; enough to cover the cloud if None. Points outside
        the grid are ignored.

    Returns
    -------
    VoxelGrid
    """
    d = _core.voxelize(cloud.xyz, voxel_size,
                       None if origin is None else tuple(float(v) for v in origin),
                       None if shape is None else tuple(int(v) for v in shape))
    return VoxelGrid(d["counts"], d["origin"], d["voxel_size"])


def vertical_profile(cloud: PointCloud, bin_size: float = 0.5, height_attr: str = "height",
                     max_height: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Point counts by height above ground.

    Parameters
    ----------
    cloud
        A height-normalised cloud.
    bin_size
        Bin height (m).
    height_attr
        Attribute holding heights; z is used if absent.
    max_height
        Top of the last bin; the highest point if None.

    Returns
    -------
    bin_bottoms, counts : numpy.ndarray
        Point counts depend on scanner distance and occlusion, so this is a
        description of the data, not of the canopy.
    """
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
    ``PAD_i = -ln(1 - N_i) / (G dz)``, where ``N_i`` is the fraction of voxel
    columns not yet intercepted above layer ``i`` that are occupied in it.

    Quick and needs only points, but it ignores beam direction and
    occlusion. For a path-length estimate use
    :func:`sylva.voxels.ray_voxelize` or :class:`GapProfile` with pulse
    data.

    Parameters
    ----------
    cloud
        A height-normalised cloud.
    voxel_size
        Voxel edge and layer thickness (m); results depend strongly on it.
    height_attr
        Attribute holding heights.
    max_height
        Top of the profile; the highest point if None.
    clumping
        Factor the PAD is multiplied by (1 = none).

    Returns
    -------
    layer_bottoms, pad : numpy.ndarray
        Heights (m) and plant area density (m² m⁻³).
    """
    h = np.ascontiguousarray(cloud.heights(height_attr))
    return _core.pad_profile_voxel(cloud.xyz, h, voxel_size, max_height, clumping)


def gap_fraction_zenith(shots: Shots, echo_heights: np.ndarray, min_height: float = 0.0,
                        zenith_edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Directional gap fraction by zenith ring from pulses that include misses.

    A pulse is a gap when none of its echoes is above ``min_height``. Needs
    every fired pulse, so use shots from a ray cloud or after
    :meth:`sylva.Shots.fill_missing`; for raw RiVLib streams use
    :func:`gap_fraction_pattern` or :class:`GapProfile`.

    Parameters
    ----------
    shots
        Pulses of one scan position.
    echo_heights
        Height above ground of each echo (length ``shots.n_echoes``).
    min_height
        Echoes at or below this height (m) do not block the pulse.
    zenith_edges
        Ring edges (degrees from up); 0-90 in 5 degree rings if None.

    Returns
    -------
    centres_deg, gap : numpy.ndarray
        Ring centres and gap fraction; NaN for rings without pulses.
    """
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

    Parameters
    ----------
    shots
        Pulses of one scan position.
    echo_heights
        Height above ground of each echo.
    pattern
        Scan pattern from :attr:`sylva.riscan.ScanPosition.pattern`.
    min_height
        Echoes at or below this height (m) do not block the pulse.
    zenith_edges
        Ring edges (degrees); 0-90 in 5 degree rings if None.
    pulses_per_line
        Pulses fired per zenith line; estimated from ``shots`` if None.

    Returns
    -------
    centres_deg, gap : numpy.ndarray
        Rings outside the scanned zenith range are NaN.

    See Also
    --------
    GapProfile : the same pooled over scans and resolved by height.
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

    Parameters
    ----------
    zenith_deg
        Ring centres (degrees).
    gap_fraction
        Gap fraction per ring.
    method : {"hinge", "miller"}
        ``"hinge"`` uses the ring nearest 57.5 degrees, where the result is
        almost independent of leaf angle; ``"miller"`` needs rings covering
        0-90 degrees.

    Returns
    -------
    float
        Effective PAI (m² m⁻²), not corrected for clumping.

    Raises
    ------
    ValueError
        For an unknown method or no valid rings.
    """
    return _core.lai_from_gap_fraction(np.ascontiguousarray(zenith_deg, dtype=float),
                                       np.ascontiguousarray(gap_fraction, dtype=float), method)


def canopy_cover(chm_data: np.ndarray, threshold: float = 2.0) -> float:
    """Canopy cover from a canopy height model.

    Parameters
    ----------
    chm_data
        CHM values, e.g. ``make_chm(...).data``; NaN cells are ignored.
    threshold
        Height (m) at or above which a cell counts as canopy.

    Returns
    -------
    float
        Fraction of valid cells that are canopy; NaN if none are valid.
    """
    return _core.canopy_cover(np.ascontiguousarray(chm_data, dtype=float), threshold)


@dataclass
class DensityGrid:
    """Ray-traced voxel statistics (raycloudtools / Lowe et al. 2021 style).

    Arrays are ``[k, j, i]`` (z, y, x). ``density`` is
    ``2 (n-1)/n · hits / path_length`` per voxel (spherical leaf angles),
    NaN where fewer than ``min_hits`` hits. Build with :func:`density_grid`.

    For most work :func:`sylva.voxels.ray_voxelize` is the fuller model
    (beam width, several estimators, occlusion); this grid is the simple,
    fast one.

    Attributes
    ----------
    n_rays
        Pulses entering each voxel.
    n_hits
        Echoes inside each voxel.
    path_length
        Summed beam path length through each voxel (m).
    density
        Plant area density (m² m⁻³).
    profile
        Mean density per layer, bottom first.
    origin
        Minimum corner ``(x, y, z)``.
    voxel_size
        Voxel edge (m).
    """

    n_rays: np.ndarray
    n_hits: np.ndarray
    path_length: np.ndarray
    density: np.ndarray
    profile: np.ndarray
    origin: np.ndarray
    voxel_size: float

    def z_levels(self) -> np.ndarray:
        """Bottom z of each voxel layer.

        Returns
        -------
        numpy.ndarray
            Length ``nz``, bottom first (m).
        """
        return self.origin[2] + np.arange(self.n_rays.shape[0]) * self.voxel_size

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxel-centre coordinates.

        Returns
        -------
        X, Y, Z : numpy.ndarray
            Each shaped like ``density``, ``(nz, ny, nx)``.
        """
        nz, ny, nx = self.density.shape
        xs = self.origin[0] + (np.arange(nx) + 0.5) * self.voxel_size
        ys = self.origin[1] + (np.arange(ny) + 0.5) * self.voxel_size
        zs = self.origin[2] + (np.arange(nz) + 0.5) * self.voxel_size
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        return X, Y, Z

    def height_above(self, dtm: Raster) -> np.ndarray:
        """Height of each voxel centre above the terrain.

        Parameters
        ----------
        dtm
            Terrain in the grid's frame.

        Returns
        -------
        numpy.ndarray
            Shaped like ``density`` (m).
        """
        X, Y, Z = self.centers()
        return Z - dtm.sample(X.ravel(), Y.ravel()).reshape(Z.shape)

    @property
    def pai(self) -> float:
        """Plant area index: the layer means summed over height (m² m⁻²).

        Includes ground layers unless :meth:`mask_ground` was applied.
        """
        return float(np.nansum(self.profile) * self.voxel_size)

    def mask_ground(self, dtm: Raster, margin: float | None = None) -> DensityGrid:
        """Remove ground voxels so terrain returns do not count as plant material.

        Parameters
        ----------
        dtm
            Terrain in the grid's frame.
        margin
            Voxels whose centre is less than this height (m) above the
            terrain are set to NaN; one voxel if None.

        Returns
        -------
        DensityGrid
            A copy with ``density`` masked and ``profile`` recomputed.
        """
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

        Parameters
        ----------
        dtm
            Terrain in the grid's frame.
        bin_size
            Height bin (m); the voxel size if None.
        max_height
            Top of the profile; the highest sampled voxel if None.
        margin
            Minimum voxel-centre height (m); one voxel if None.
        pooled
            Pool hits and path lengths per bin (recommended).

        Returns
        -------
        bin_bottoms, pad : numpy.ndarray
            PAD in m² m⁻³; ``np.nansum(pad) * bin_size`` is PAI.
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
    """Ray-trace pulses through a voxel grid and estimate plant area density.

    Pulses without echoes run to the grid edge.

    Parameters
    ----------
    shots
        Pulses from one or more scan positions in a common frame.
    voxel_size
        Voxel edge (m).
    origin, shape
        Grid minimum corner and ``(nx, ny, nz)``; both None covers the echoes.
    min_hits
        Voxels with fewer echoes get NaN density.

    Returns
    -------
    DensityGrid
    """
    d = _core.density_grid(shots._to_core(), voxel_size,
                           None if origin is None else tuple(float(v) for v in origin),
                           None if shape is None else tuple(int(v) for v in shape), min_hits)
    return DensityGrid(d["n_rays"], d["n_hits"], d["path_length"], d["density"], d["profile"],
                       d["origin"], d["voxel_size"])


def fit_ground_plane(points, cell: float = 1.0, centre=None, radius: float | None = None,
                     iterations: int = 20) -> np.ndarray:
    """Ground plane ``z = a x + b y + c`` through the lowest point of every
    ``cell`` (m) grid cell, optionally within ``radius`` of ``centre`` (xy),
    fitted with Huber-weighted least squares so tree bases and pits pull
    little (after Calders et al. 2014).

    Used for the single-scan gap profiles, where a plane around the scanner
    is enough and a DTM may not exist.

    Parameters
    ----------
    points
        A :class:`~sylva.PointCloud` or ``(N, 3)`` array; for a scan,
        downward echoes work best.
    cell
        Grid cell size (m) for the lowest-point selection.
    centre, radius
        Only use points within ``radius`` (m) of ``centre`` (xy).
    iterations
        Reweighting iterations.

    Returns
    -------
    numpy.ndarray
        ``[a, b, c]``; ground height at (x, y) is ``a * x + b * y + c``.

    Raises
    ------
    ValueError
        With fewer than 3 points.
    """
    xyz = np.asarray(points.xyz if isinstance(points, PointCloud) else points, dtype=float)
    if centre is not None and radius is not None:
        xyz = xyz[np.hypot(xyz[:, 0] - centre[0], xyz[:, 1] - centre[1]) <= radius]
    if len(xyz) < 3:
        raise ValueError("too few points for a ground plane")
    key = np.floor(xyz[:, :2] / cell).astype(np.int64)
    order = np.lexsort((xyz[:, 2], key[:, 1], key[:, 0]))
    k = key[order]
    first = np.r_[True, np.any(k[1:] != k[:-1], axis=1)]
    low = xyz[order][first]
    A = np.c_[low[:, 0], low[:, 1], np.ones(len(low))]
    w = np.ones(len(low))
    coef = np.zeros(3)
    for _ in range(iterations):
        coef = np.linalg.lstsq(A * w[:, None], low[:, 2] * w, rcond=None)[0]
        r = low[:, 2] - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = np.abs(r) / (1.345 * s)
        w = np.sqrt(np.where(u <= 1, 1.0, 1.0 / u))
    return coef


def fired_pulses_per_ring(shots_scanner: Shots, pattern: dict, zenith_edges,
                          shot_stride: int = 1, ground_zenith=(100.0, 125.0)) -> np.ndarray:
    """Pulses a RIEGL scan fired into each zenith ring, for streams without
    the pulses that returned nothing (RiVLib). ``shots_scanner`` in the
    scanner frame, read with ``shot_stride``.

    Every azimuth step fires one pulse per zenith line, so a ring gets
    (lines in the ring) x (pulses per line). Pulses per line are counted on
    the downward lines in ``ground_zenith``, where every pulse hits the
    ground and so every fired pulse is in the stream: the median there. The
    nominal ``phi_count`` of the pattern is not used, because the scanner
    fires about 1 % more pulses than nominal and a percentile over all lines
    overshoots, since the mirror's zenith angles do not sit exactly on the
    nominal lines. In dense canopy, where nearly every pulse returns, a few
    per cent too many fired pulses read as gaps and cap the PAI.

    Parameters
    ----------
    shots_scanner
        Pulses of one scan in the scanner frame.
    pattern
        Scan pattern from :attr:`sylva.riscan.ScanPosition.pattern`.
    zenith_edges
        Ring edges (degrees).
    shot_stride
        The ``shot_stride`` used when reading (for the fallback estimate).
    ground_zenith
        Zenith range (degrees) of the lines used to count pulses per line.
        If fewer than 10 lines fall in it, the estimate from
        :meth:`sylva.Shots.pulses_per_line` is used instead.

    Returns
    -------
    numpy.ndarray
        Pulses fired per ring (float, length ``len(zenith_edges) - 1``), for
        :meth:`GapProfile.add_scan`.
    """
    edges = np.asarray(zenith_edges, dtype=float)
    theta, line_edges = shots_scanner._zenith_lines(pattern)
    zen, _ = shots_scanner.zenith_azimuth()
    observed, _ = np.histogram(zen, bins=line_edges)
    ground = (theta >= ground_zenith[0]) & (theta <= ground_zenith[1])
    if ground.sum() >= 10 and np.median(observed[ground]) > 0:
        ppl = float(np.median(observed[ground]))
    else:
        ppl = float(shots_scanner.pulses_per_line(pattern, shot_stride=shot_stride))
    # Each line spans theta_delta; share it among the rings it overlaps, so a
    # line on a ring edge counts half on each side, like its pulses do.
    half = 0.5 * float(pattern["theta_delta"])
    lo, hi = theta - half, theta + half
    overlap = np.clip(np.minimum(hi[:, None], edges[None, 1:]) - np.maximum(lo[:, None], edges[None, :-1]), 0, None)
    lines = overlap.sum(axis=0) / (2 * half)
    return lines * ppl


@dataclass
class GapProfile:
    """Gap probability by zenith ring, azimuth sector and height above ground,
    pooled over scans (Jupp et al. 2009), and what follows from it.

    Build it with :meth:`empty` and :meth:`add_scan` (one call per scan
    position), then read :meth:`report`. Returns are weighted ``1 / n`` per
    echo of an ``n``-echo pulse (equal weighting, Armston et al. 2013);
    ``shots`` are pulses fired.

    Validated against pylidar on TERN plots (within 4 % where unsaturated)
    and against hemispherical photographs; see *Benchmarks > Canopy gap
    profiles* and the *Pulse data* guide for the full workflow.

    Examples
    --------
    >>> prof = GapProfile.empty()
    >>> for pos in project.with_scans():
    ...     s = io.read_rxp_shots(pos.rxp, shot_stride=4)
    ...     fired = fired_pulses_per_ring(s, pos.pattern, prof.zenith_edges, shot_stride=4)
    ...     s = s.transform(pos.sop)
    ...     xyz = s.echo_xyz()
    ...     a, b, c = fit_ground_plane(xyz, centre=pos.sop[:2, 3], radius=25)
    ...     prof.add_scan(s, xyz[:, 2] - (a * xyz[:, 0] + b * xyz[:, 1] + c), fired_per_ring=fired)
    >>> prof.report()["pai_hinge"]
    """

    zenith_edges: np.ndarray  #: deg
    n_azimuth: int
    height_bin: float  #: m
    hits: np.ndarray  #: (rings, sectors, heights)
    shots: np.ndarray  #: (rings, sectors)
    scan_hits: list = field(default_factory=list)  #: per scan: (rings, sectors) total returns
    scan_shots: list = field(default_factory=list)  #: per scan: (rings, sectors)
    scan_low: list = field(default_factory=list)  #: per scan: (rings, sectors) returns in bins below ``min_height``
    #: Plant area is counted from this height up (m). Upward pulses from a
    #: tripod hit little below it but terrain rising on slopes, and returns
    #: below the ground model are kept in the lowest bin.
    min_height: float = 0.5

    @classmethod
    def empty(cls, zenith_edges=np.arange(5.0, 75.0, 5.0), n_azimuth: int = 36,
              height_bin: float = 0.5, max_height: float = 80.0) -> "GapProfile":
        """Start an empty profile.

        Parameters
        ----------
        zenith_edges
            Ring edges (degrees from up). The default, 5-75 degrees in 5 degree
            rings, avoids the near-horizontal rings that see mostly stems.
        n_azimuth
            Azimuth sectors per ring, used for the clumping index.
        height_bin
            Height resolution (m).
        max_height
            Top of the profile (m); returns above it go in the top bin.

        Returns
        -------
        GapProfile
        """
        edges = np.asarray(zenith_edges, dtype=float)
        nh = int(np.ceil(max_height / height_bin))
        return cls(edges, int(n_azimuth), float(height_bin), np.zeros((len(edges) - 1, n_azimuth, nh)),
                   np.zeros((len(edges) - 1, n_azimuth)))

    def add_scan(self, shots: Shots, echo_heights, fired_per_ring=None, min_height: float = -np.inf) -> None:
        """Add one scan position's pulses; ``echo_heights`` above ground, one
        per echo. Without ``fired_per_ring`` the shots must include the pulses
        that returned nothing (e.g. :meth:`Shots.fill_missing` or a ray cloud).

        Returns below ``min_height`` are not counted. By default all count, and
        those below zero go in the lowest bin: a pulse fired upwards cannot
        hit the ground, so a negative height means the ground model is off
        there, not that the return is not vegetation.

        Parameters
        ----------
        shots
            One scan position's pulses. Only their directions and echo
            counts are used, so the scanner or project frame both work.
        echo_heights
            Height above ground of each echo (length ``shots.n_echoes``),
            e.g. from a DTM or :func:`fit_ground_plane`.
        fired_per_ring
            Pulses fired into each ring (:func:`fired_pulses_per_ring`) for
            RiVLib streams, which lack the misses.
        min_height
            Echoes below this height (m) are dropped entirely.

        Notes
        -----
        The profile is updated in place; add every scan of the plot before
        reading it.
        """
        nr, na, nh = self.hits.shape
        hits, shots_n = _core.pgap_histogram(
            shots._to_core(), np.ascontiguousarray(echo_heights, dtype=float), [float(e) for e in self.zenith_edges],
            na, self.height_bin, nh, float(min_height),
            None if fired_per_ring is None else [float(v) for v in fired_per_ring])
        hits = hits.reshape(nr, na, nh)
        shots_n = shots_n.reshape(nr, na)
        self.hits += hits
        self.shots += shots_n
        self.scan_hits.append(hits.sum(axis=2))
        self.scan_shots.append(shots_n)
        self.scan_low.append(hits[:, :, : self._first_bin()].sum(axis=2))

    def _first_bin(self) -> int:
        return int(np.floor(self.min_height / self.height_bin + 1e-9))

    @property
    def heights(self) -> np.ndarray:
        """Top of each height bin (m): the profile at ``z`` counts returns below it."""
        return (np.arange(self.hits.shape[2]) + 1) * self.height_bin

    @property
    def zenith(self) -> np.ndarray:
        """Ring centres (deg)."""
        return 0.5 * (self.zenith_edges[:-1] + self.zenith_edges[1:])

    def pgap(self) -> np.ndarray:
        """Gap probability by ring and height.

        Returns
        -------
        numpy.ndarray
            ``(rings, heights)``: 1 minus the returns between
            :attr:`min_height` and the top of each height bin, over the
            pulses fired, clipped to 0-1. NaN for rings without pulses.
        """
        fired = self.shots.sum(axis=1)[:, None]
        counted = self.hits.sum(axis=1).copy()
        counted[:, : self._first_bin()] = 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            p = 1.0 - np.cumsum(counted, axis=1) / fired
        p[fired[:, 0] <= 0] = np.nan
        return np.clip(p, 0.0, 1.0)

    def _floor(self) -> np.ndarray:
        # A ring with no gap left is floored at one pulse's worth of gap.
        return 1.0 / np.maximum(self.shots.sum(axis=1), 1.0)

    def pai_profile(self, method: str = "hinge") -> np.ndarray:
        """Cumulative plant area index below each height (effective, not
        corrected for clumping).

        ``hinge``: ``-1.1 ln P(57.5 deg)`` from the ring holding 57.5 deg.
        ``linear``: Jupp et al. (2009): ``-ln P(theta) = PAI_h + (2 / pi)
        tan(theta) PAI_v`` (horizontal and vertical foliage) fitted over the
        rings, ``PAI = PAI_h + PAI_v``; the mean leaf angle is
        ``atan(PAI_v / PAI_h)``.
        ``weighted``: ``2 cos(theta) (-ln P)`` averaged over rings with weights
        ``sin(theta)`` (Miller 1967 over the rings measured, spherical leaves).

        Parameters
        ----------
        method : {"hinge", "linear", "weighted"}
            Estimator. Hinge is the most robust; linear also gives leaf angle.

        Returns
        -------
        numpy.ndarray
            PAI below the top of each height bin (m² m⁻²), non-decreasing.

        Raises
        ------
        ValueError
            For an unknown method, or ``"hinge"`` with no pulses in the ring
            holding 57.5 degrees.
        """
        p = self.pgap()
        lp = -np.log(np.maximum(p, self._floor()[:, None]))
        th = np.radians(self.zenith)
        ok = np.isfinite(p).all(axis=1)
        if method == "hinge":
            ring = np.searchsorted(self.zenith_edges, 57.5, side="right") - 1
            if not (0 <= ring < len(th)) or not ok[ring]:
                raise ValueError("no pulses in the ring holding 57.5 deg")
            return 1.1 * lp[ring]
        if method == "weighted":
            w = (np.sin(th) * ok)[:, None]
            return (2.0 * np.cos(th)[:, None] * np.nan_to_num(lp) * w).sum(axis=0) / w.sum()
        if method == "linear":
            return self._linear(lp, th, ok)[0]
        raise ValueError("method must be 'hinge', 'linear' or 'weighted'")

    def _linear(self, lp, th, ok):
        x = np.tan(th[ok])
        y = lp[ok]
        A = np.c_[np.ones_like(x), x]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pai_h, pai_v = coef[0], coef[1] * np.pi / 2
        pai = np.maximum(pai_h + pai_v, 0.0)
        # Either term can fit slightly negative where there is little
        # foliage; the angle is only defined for the parts that are not.
        h, v = np.maximum(pai_h, 0.0), np.maximum(pai_v, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mla = np.where(h + v > 0, np.degrees(np.arctan2(v, h)), np.nan)
        return pai, mla

    def pavd_profile(self, method: str = "hinge") -> np.ndarray:
        """Plant area volume density: the height derivative of :meth:`pai_profile`.

        Parameters
        ----------
        method : {"hinge", "linear", "weighted"}
            Estimator, as for :meth:`pai_profile`.

        Returns
        -------
        numpy.ndarray
            PAVD per height bin (m² m⁻³).
        """
        return np.gradient(self.pai_profile(method), self.height_bin)

    def clumping(self, zenith: float = 57.5) -> float:
        """Lang & Xiang (1986) clumping index at the ring holding ``zenith``:
        ``ln(mean P) / mean(ln P)`` over every (scan, azimuth sector) segment.
        1 is random foliage; below 1 clumped.

        Parameters
        ----------
        zenith
            Zenith angle (degrees) whose ring is used.

        Returns
        -------
        float
            Clumping index; NaN without scans or without any gap. Divide
            effective PAI by it for true PAI.
        """
        ring = np.searchsorted(self.zenith_edges, zenith, side="right") - 1
        seg_p, seg_n = [], []
        lows = self.scan_low if len(self.scan_low) == len(self.scan_hits) else [0.0] * len(self.scan_hits)
        for hits, shots, low in zip(self.scan_hits, self.scan_shots, lows):
            ok = shots[ring] > 0
            h = hits[ring] - (low[ring] if np.ndim(low) else 0.0)
            seg_p.append(1.0 - h[ok] / shots[ring][ok])
            seg_n.append(shots[ring][ok])
        if not seg_p:
            return float("nan")
        p = np.clip(np.concatenate(seg_p), 0.0, 1.0)
        n = np.concatenate(seg_n)
        p = np.maximum(p, 1.0 / n)
        mean_p = np.average(p, weights=n)
        mean_lnp = np.average(np.log(p), weights=n)
        return float(np.log(mean_p) / mean_lnp) if mean_lnp < 0 else float("nan")

    def report(self, top_fraction: float = 0.99, saturation_gap: float = 0.005) -> dict:
        """Plot summary: effective PAI (hinge, linear, weighted), mean leaf
        angle from the linear fit, clumping index and clumping-corrected hinge
        PAI, canopy height (where the hinge PAI profile reaches
        ``top_fraction`` of its total), canopy closure at 57.5 deg, cover at
        the steepest ring measured, and the profiles.

        ``saturated`` is set when less than ``saturation_gap`` of the pulses
        at 57.5 deg got through the canopy (hinge PAI above about 5.8): the
        PAI is then bounded by the pulse count, not measured. Dense
        rainforest seen from the ground reaches it.

        Parameters
        ----------
        top_fraction
            Fraction of total PAI that defines canopy height.
        saturation_gap
            Gap fraction at 57.5 degrees below which ``saturated`` is set.

        Returns
        -------
        dict
            Scalars ``saturated``, ``gap_57``, ``pai_hinge``, ``pai_linear``,
            ``pai_weighted``, ``mla_linear`` (degrees), ``clumping``,
            ``pai_hinge_corrected``, ``canopy_height`` (m), ``closure_57``,
            ``cover``, ``cover_zenith``, ``n_scans``, ``pulses``; and arrays
            ``height``, ``pai_hinge_profile``, ``pavd_hinge``,
            ``pai_linear_profile``, ``pavd_linear``. Check ``saturated``
            before reporting PAI.
        """
        p = self.pgap()
        th = np.radians(self.zenith)
        ok = np.isfinite(p).all(axis=1)
        hinge = self.pai_profile("hinge")
        lp = -np.log(np.maximum(p, self._floor()[:, None]))
        linear, mla = self._linear(lp, th, ok)
        weighted = self.pai_profile("weighted")
        omega = self.clumping()
        total = hinge[-1]
        top = self.heights[np.searchsorted(hinge, top_fraction * total)] if total > 0 else 0.0
        ring = np.searchsorted(self.zenith_edges, 57.5, side="right") - 1
        steep = int(np.flatnonzero(ok)[0]) if ok.any() else 0
        # Gap left in the hinge ring above the canopy: below `saturation_gap`
        # the PAI is set by how few pulses got through, not by the canopy.
        gap57 = float(p[ring, -1])
        return {
            "saturated": bool(gap57 < saturation_gap), "gap_57": gap57,
            "pai_hinge": float(total), "pai_linear": float(linear[-1]), "pai_weighted": float(weighted[-1]),
            "mla_linear": float(mla[-1]), "clumping": omega,
            "pai_hinge_corrected": float(total / omega) if omega and np.isfinite(omega) else float("nan"),
            "canopy_height": float(top), "closure_57": float(1 - p[ring, -1]),
            "cover": float(1 - p[steep, -1]), "cover_zenith": float(self.zenith[steep]),
            "n_scans": len(self.scan_hits), "pulses": float(self.shots.sum()),
            "height": self.heights, "pai_hinge_profile": hinge, "pavd_hinge": np.gradient(hinge, self.height_bin),
            "pai_linear_profile": linear, "pavd_linear": np.gradient(linear, self.height_bin),
        }
