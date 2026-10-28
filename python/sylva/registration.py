# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Rigid registration of scan positions."""

from __future__ import annotations

import warnings

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = ["kabsch", "icp", "merge_scans", "plot_overlay", "plot_sections", "rotation_z",
           "translation", "estimate_normals"]


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best rigid transform between paired points (Kabsch 1976; Umeyama 1991, no scale).

    Use with matched targets (reflectors, tie points) to register scans.

    Parameters
    ----------
    source, target
        ``(N, 3)`` corresponding points, N >= 3 and not collinear.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` matrix mapping ``source`` onto ``target`` in the least
        squares sense.
    """
    return _core.kabsch(np.ascontiguousarray(source, dtype=float),
                        np.ascontiguousarray(target, dtype=float))


def icp(source: PointCloud, target: PointCloud, init: np.ndarray | None = None,
        max_correspondence_distance: float = 0.5, max_iterations: int = 50,
        tolerance: float = 1e-6, method: str = "point", trim: float = 1.0,
        normal_k: int = 12) -> tuple[np.ndarray, dict]:
    """Iterative closest point.

    ``method`` is ``"point"`` (point-to-point, Besl & McKay 1992) or
    ``"plane"`` (point-to-plane, Chen & Medioni 1992, linearised as in Low
    2004; normals from ``target.attrs['nx','ny','nz']`` if present, else
    PCA). ``trim`` keeps that fraction of closest correspondences per
    iteration (trimmed ICP for partial overlap, Chetverikov et al. 2002).

    Parameters
    ----------
    source
        Cloud to move.
    target
        Fixed reference cloud.
    init
        Starting ``(4, 4)`` transform (e.g. from the SOP or :func:`kabsch`);
        ICP only converges from a start within about
        ``max_correspondence_distance``.
    max_correspondence_distance
        Pairs further apart (m) are ignored.
    max_iterations
        Iteration limit.
    tolerance
        Stop when the RMSE changes less than this.
    method : {"point", "plane"}
        Error metric; point-to-plane converges faster on stems and ground.
    trim
        Fraction of pairs kept (0-1).
    normal_k
        Neighbours for PCA normals with ``"plane"``.

    Returns
    -------
    transform : numpy.ndarray
        ``(4, 4)`` matrix mapping ``source`` onto ``target``.
    info : dict
        ``rmse`` (m), ``iterations`` and ``n_correspondences``.

    Notes
    -----
    Thin both clouds to 2-5 cm first; ICP on full-density TLS is slow and
    no more accurate. Check the result with :func:`sylva.quality.stem_noise`.
    """
    init = None if init is None else np.ascontiguousarray(init, dtype=float)
    return _core.icp(source.xyz, target.xyz, init, max_correspondence_distance, max_iterations,
                     tolerance, method, trim, normal_k)


def estimate_normals(cloud: PointCloud, k: int = 12) -> np.ndarray:
    """PCA normals; the same as :func:`sylva.filters.estimate_normals`.

    Parameters
    ----------
    cloud
        Input points.
    k
        Neighbours per point.

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` unit normals with arbitrary sign.
    """
    return _core.estimate_normals(cloud.xyz, k)


def merge_scans(clouds: list[PointCloud], transforms: list[np.ndarray] | None = None,
                scan_ids: bool = True) -> PointCloud:
    """Put several scans in one frame and merge them.

    Parameters
    ----------
    clouds
        One cloud per scan position.
    transforms
        One ``(4, 4)`` matrix per cloud (SOPs or ICP results); None if the
        clouds are already registered.
    scan_ids
        Add a ``scan_id`` attribute (int32, the index in ``clouds``), which
        :func:`sylva.quality.stem_noise` can read.

    Returns
    -------
    PointCloud
        Only attributes present in every cloud are kept.
    """
    out = []
    for i, c in enumerate(clouds):
        if transforms is not None:
            c = c.transform(transforms[i])
        if scan_ids:
            c = c.with_attrs(scan_id=np.full(len(c), i, dtype=np.int32))
        out.append(c)
    return PointCloud.concatenate(out)


def plot_overlay(source: PointCloud, target: PointCloud, transform: np.ndarray | None = None, *,
                 slab_z: float | None = None, thickness: float = 0.5,
                 centre: tuple[float, float] | None = None, radius: float | None = None,
                 max_distance: float = 0.2, max_points: int = 200_000, title: str | None = None):
    """Draw two scans over each other, to judge a registration by eye.

    A small RMSE says the clouds are close on average; it does not show a
    systematic offset in z, a slight rotation, or doubled stems. This plots a
    thin horizontal slab (plan view, ``x`` against ``y``) and a thin vertical
    slab (side view, ``x`` against ``z``) with the source in vermilion and the
    target in blue, so a misfit shows as two copies of every stem and of the
    ground. With ``transform`` there is a column before and one after it, and a
    last column colours the source by its distance to the nearest target point.

    Requires matplotlib, which Sylva does not otherwise depend on.

    Parameters
    ----------
    source
        Cloud that was moved.
    target
        Fixed reference cloud.
    transform
        ``(4, 4)`` matrix that registers ``source`` on ``target`` (an
        :func:`icp` result, a RIEGL SOP, or one from :mod:`sylva.coreg`). None
        if ``source`` is already in the frame of ``target``, which leaves out
        the "before" column.
    slab_z
        Height (m, in the frame of ``target``) of the plan-view slab. Default
        1.3 m above the lowest 2 % of the target, about breast height on level
        ground. Pick a height with stems or a flat floor for the clearest
        picture.
    thickness
        Thickness (m) of both slabs.
    centre, radius
        ``(x, y)`` (m) and half-width (m) of a square window to zoom into,
        which is how a centimetre of misfit becomes visible. Default the whole
        extent, with the side slab at the ``y`` where the target has most
        points.
    max_distance
        Upper end (m) of the distance colour scale. The scale goes with the
        square root of the distance, so the first centimetres, where a good
        registration lives, take more of the colour range than the tail.
    max_points
        Most points drawn per cloud and panel; a larger slab is thinned at
        random (seeded, so the figure repeats).
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Two rows (plan, side) by two or three columns. Each column title gives
        the median distance from a random sample of the source to the target
        over the whole cloud, not just the slab.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``transform`` is not 4x4, or a slab holds no points.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm
    except ImportError as exc:
        raise ImportError("plot_overlay needs matplotlib: pip install matplotlib") from exc
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("transform must be 4x4")
    rng = np.random.default_rng(0)
    tgt = np.ascontiguousarray(target.xyz)
    registered = source.xyz if transform is None else source.transform(transform).xyz
    if transform is None:
        stages = [("registered", registered)]
    else:
        stages = [("before", source.xyz), ("after", registered)]
    if slab_z is None:
        slab_z = float(np.percentile(tgt[:, 2], 2) + 1.3)
    half = thickness / 2
    if centre is None:
        centre = (float(np.median(tgt[:, 0])), float(np.median(tgt[:, 1])))
    box = None
    if radius is not None:
        box = (centre[0] - radius, centre[0] + radius, centre[1] - radius, centre[1] + radius)
        y0 = centre[1]
    else:
        # the side slab goes where the target has most points, so it shows stems and not a gap
        counts, edges = np.histogram(tgt[:, 1], bins=max(1, int(np.ptp(tgt[:, 1]) / thickness)))
        y0 = float(0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]))

    def cut(xyz, view):
        """Indices of the points in the plan slab (a z band) or the side slab (a y band)."""
        band = xyz[:, 2] - slab_z if view == "plan" else xyz[:, 1] - y0
        keep = np.abs(band) < half
        if box is not None:
            keep &= (xyz[:, 0] >= box[0]) & (xyz[:, 0] <= box[1])
            if view == "plan":
                keep &= (xyz[:, 1] >= box[2]) & (xyz[:, 1] <= box[3])
        return np.flatnonzero(keep)

    def thin(idx):
        if len(idx) <= max_points:
            return idx
        return np.sort(rng.choice(idx, max_points, replace=False))

    def nearest(xyz):
        """Distance from each point to the nearest target point."""
        return _core.knn(tgt, np.ascontiguousarray(xyz), 1)[0][:, 0]

    ncols = len(stages) + 1
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 6.4), squeeze=False,
                             layout="constrained")
    colours = {"source": "#D55E00", "target": "#0072B2"}     # Okabe-Ito vermilion and blue
    lo, hi = tgt.min(axis=0), tgt.max(axis=0)
    drawn = 0
    for view, row in (("plan", axes[0]), ("side", axes[1])):
        yi = 1 if view == "plan" else 2                      # plotted against x
        tidx = thin(cut(tgt, view))
        for ax, (name, xyz) in zip(row, stages, strict=False):
            sidx = thin(cut(xyz, view))
            drawn += len(sidx) + len(tidx)
            for pts, idx, who in ((tgt, tidx, "target"), (xyz, sidx, "source")):
                ax.scatter(pts[idx, 0], pts[idx, yi], s=0.6, alpha=0.6, c=colours[who],
                           rasterized=True, label=who)
            if view == "plan":
                sample = xyz[rng.choice(len(xyz), min(len(xyz), 20_000), replace=False)]
                ax.set_title(f"{name}: median {np.median(nearest(sample)):.3f} m", fontsize=10)
        # distance map: the registered source in the slab, coloured by the nearest target point
        sidx = thin(cut(registered, view))
        if len(sidx):
            pts = registered[sidx]
            sc = row[-1].scatter(pts[:, 0], pts[:, yi], s=0.8, c=nearest(pts), cmap="viridis",
                                 norm=PowerNorm(0.5, 0.0, max_distance), rasterized=True)
            if view == "plan":
                row[-1].set_title("distance to target", fontsize=10)
                bar = fig.colorbar(sc, ax=axes[:, -1].tolist(), label="m", extend="max", shrink=0.8)
                ticks = [t for t in (0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
                         if t <= max_distance]
                bar.set_ticks(ticks + ([] if max_distance in ticks else [max_distance]))
        for ax in row:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (m)")
            if ax is not row[0]:
                ax.sharex(row[0])
                ax.sharey(row[0])
        row[0].set_ylabel("y (m)" if view == "plan" else "z (m)")
        # limits from the whole cloud (or the window), not the slab, so a sparse slab is no sliver
        row[0].set_xlim(box[:2] if box else (lo[0], hi[0]))
        row[0].set_ylim(box[2:] if box and view == "plan" else (lo[yi], hi[yi]))
        if len(tidx) == 0:
            warnings.warn(f"no target points in the {view} slab: set slab_z, thickness, centre "
                          "or radius", stacklevel=2)
    if drawn == 0:
        plt.close(fig)
        raise ValueError("no points in the slabs: set slab_z, thickness, centre or radius")
    axes[0, 0].legend(markerscale=10, loc="upper left", fontsize=8)
    if title:
        fig.suptitle(title)
    return fig


def plot_sections(source: PointCloud, target: PointCloud, transform: np.ndarray | None = None,
                  centres: list[tuple[float, float]] | None = None, *, n: int = 3,
                  height: float = 1.3, thickness: float = 0.1, radius: float = 1.0,
                  max_points: int = 50_000, title: str | None = None):
    """Look at single stems, or other compact objects, in cross section.

    A registration can look right at plot scale and still put every stem a few
    centimetres out. For each object this draws a horizontal cross section (a
    slab ``thickness`` thick at ``height`` above the local ground, ``x``
    against ``y``) and a vertical one (``x`` against ``z`` through the centre),
    with the source in vermilion over the target in blue. A stem that
    registers shows as one ring and one column; a misregistered one shows two,
    shifted. With ``transform`` each section is drawn before and after it.

    Requires matplotlib, which Sylva does not otherwise depend on.

    Parameters
    ----------
    source
        Cloud that was moved.
    target
        Fixed reference cloud.
    transform
        ``(4, 4)`` matrix that registers ``source`` on ``target``; None if it
        already is, which leaves out the "before" panels.
    centres
        ``(x, y)`` (m) of each object, in the frame of ``target``. Default the
        ``n`` places where the target has the most points in the slab at
        ``height``, at least ``2 * radius`` apart, and the source has some
        within a panel's reach. The source need not match the target there, so
        a bad registration still gets sections. These are the best-seen
        objects, stems only if the plot has stems there: look at the sections.
    n
        Number of objects to pick when ``centres`` is None.
    height
        Height (m) of the horizontal section above the local ground, which is
        the lowest 2 % of the target within ``3 * radius`` of the centre.
        1.3 m is breast height.
    thickness
        Thickness (m) of both sections. Thinner shows a stem as a clean ring
        but needs a dense scan.
    radius
        Half-width (m) of each panel.
    max_points
        Most points drawn per cloud and panel; thinned at random (seeded).
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        One row per object. The title of each horizontal section gives the
        median distance (m) from the source points in it to the whole target.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If ``transform`` is not 4x4, or no object, or no ground, is found.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_sections needs matplotlib: pip install matplotlib") from exc
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("transform must be 4x4")
    rng = np.random.default_rng(0)
    tgt = np.ascontiguousarray(target.xyz)
    registered = source.xyz if transform is None else source.transform(transform).xyz
    if transform is None:
        stages = [("registered", registered)]
    else:
        stages = [("before", source.xyz), ("after", registered)]

    def ground_at(x, y):
        near = np.hypot(tgt[:, 0] - x, tgt[:, 1] - y) < 3 * radius
        if not near.any():
            raise ValueError(f"no target points within {3 * radius:g} m of ({x:.1f}, {y:.1f})")
        return float(np.percentile(tgt[near, 2], 2))

    if centres is None:
        # grid cells of 0.25 m; an object is where both clouds fill the same cell at height
        cell = 0.25
        lo = tgt[:, :2].min(axis=0)
        shape = np.ceil((tgt[:, :2].max(axis=0) - lo) / cell).astype(int) + 1
        coarse = 2.0
        cg = np.ceil((tgt[:, :2].max(axis=0) - lo) / coarse).astype(int) + 1
        ground = np.full(cg, np.nan)
        ci = np.floor((tgt[:, :2] - lo) / coarse).astype(int)
        flat = ci[:, 0] * cg[1] + ci[:, 1]
        order = np.lexsort((tgt[:, 2], flat))
        fs, zs = flat[order], tgt[order, 2]
        starts = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]])
        counts = np.diff(np.r_[starts, len(fs)])
        ground.flat[fs[starts]] = zs[starts + (0.02 * counts).astype(int)]

        def counts_in_slab(xyz):
            ij = np.clip(np.floor((xyz[:, :2] - lo) / coarse).astype(int), 0, cg - 1)
            g = ground[ij[:, 0], ij[:, 1]]
            keep = np.abs(xyz[:, 2] - g - height) < thickness / 2 + 0.05
            ij = np.floor((xyz[keep, :2] - lo) / cell).astype(int)
            ij = ij[((ij >= 0) & (ij < shape)).all(axis=1)]
            out = np.zeros(shape)
            np.add.at(out, (ij[:, 0], ij[:, 1]), 1)
            return out

        k = max(1, int(round(radius / cell)))
        padded = np.pad(counts_in_slab(registered), ((k + 1, k), (k + 1, k)))
        cum = padded.cumsum(axis=0).cumsum(axis=1)
        w = 2 * k + 1
        near_source = (cum[w:w + shape[0], w:w + shape[1]] - cum[:shape[0], w:w + shape[1]]
                       - cum[w:w + shape[0], :shape[1]] + cum[:shape[0], :shape[1]])
        both = np.where(near_source >= 5, counts_in_slab(tgt), 0.0)
        k = max(1, int(round(2 * radius / cell)))
        centres = []
        for _ in range(n):
            i, j = np.unravel_index(np.argmax(both), both.shape)
            if both[i, j] == 0:
                break
            centres.append((float(lo[0] + (i + 0.5) * cell), float(lo[1] + (j + 0.5) * cell)))
            both[max(0, i - k):i + k + 1, max(0, j - k):j + k + 1] = 0
        if not centres:
            raise ValueError("no object seen by both clouds at this height: set centres or height")
    centres = [(float(x), float(y)) for x, y in centres]

    ncols = 2 * len(stages)
    fig, axes = plt.subplots(len(centres), ncols, figsize=(2.9 * ncols, 3.0 * len(centres)),
                             squeeze=False, layout="constrained")
    colours = {"source": "#D55E00", "target": "#0072B2"}     # Okabe-Ito vermilion and blue

    def thin(idx):
        if len(idx) <= max_points:
            return idx
        return np.sort(rng.choice(idx, max_points, replace=False))

    half = thickness / 2
    for r, (cx, cy) in enumerate(centres):
        z0 = ground_at(cx, cy)
        zc = z0 + height
        for c, (name, xyz) in enumerate(stages):
            for off, view in ((0, "plan"), (1, "side")):
                ax = axes[r, 2 * c + off]
                if view == "plan":
                    band = np.abs(xyz[:, 2] - zc) < half
                    tband = np.abs(tgt[:, 2] - zc) < half
                    a, b = 0, 1
                else:
                    band = np.abs(xyz[:, 1] - cy) < half
                    tband = np.abs(tgt[:, 1] - cy) < half
                    a, b = 0, 2
                win = np.abs(xyz[:, 0] - cx) < radius
                twin = np.abs(tgt[:, 0] - cx) < radius
                if view == "plan":
                    win &= np.abs(xyz[:, 1] - cy) < radius
                    twin &= np.abs(tgt[:, 1] - cy) < radius
                else:
                    win &= (xyz[:, 2] > z0 - 0.2) & (xyz[:, 2] < z0 + 3.0)
                    twin &= (tgt[:, 2] > z0 - 0.2) & (tgt[:, 2] < z0 + 3.0)
                sidx = thin(np.flatnonzero(band & win))
                tidx = thin(np.flatnonzero(tband & twin))
                for pts, idx, who in ((tgt, tidx, "target"), (xyz, sidx, "source")):
                    ax.scatter(pts[idx, a], pts[idx, b], s=2.0, alpha=0.6, c=colours[who],
                               rasterized=True, label=who)
                ax.set_xlim(cx - radius, cx + radius)
                ax.set_ylim((cy - radius, cy + radius) if view == "plan" else (z0 - 0.2, z0 + 3.0))
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)" if view == "plan" else "z (m)")
                if view == "plan":
                    if len(sidx):
                        d = _core.knn(tgt, np.ascontiguousarray(xyz[sidx]), 1)[0][:, 0]
                        ax.set_title(f"({cx:.1f}, {cy:.1f}) {name}: median {np.median(d):.3f} m",
                                     fontsize=9)
                    else:
                        ax.set_title(f"({cx:.1f}, {cy:.1f}) {name}: no source points", fontsize=9)
                else:
                    ax.set_title(f"{name}, side", fontsize=9)
    axes[0, 0].legend(markerscale=6, loc="upper left", fontsize=8)
    if title:
        fig.suptitle(title)
    return fig


def rotation_z(angle_deg: float) -> np.ndarray:
    """Rotation about the z axis.

    Parameters
    ----------
    angle_deg
        Angle in degrees, counter-clockwise seen from above.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` matrix.
    """
    return _core.rotation_z(float(angle_deg))


def translation(dx: float, dy: float, dz: float) -> np.ndarray:
    """Translation matrix.

    Parameters
    ----------
    dx, dy, dz
        Shift (m).

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` matrix.
    """
    return _core.translation(float(dx), float(dy), float(dz))
