# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Terrestrial and airborne lidar together.

A TLS plot measures stems, diameters and the lower canopy in detail; an
airborne survey measures the upper canopy and the terrain over the whole
landscape. This module puts the two in one frame and combines them:

- :func:`register` places a TLS plot (in a local frame, or georeferenced
  with a GNSS error of a few metres) on the ALS survey, and reports the
  residuals and their uncertainty;
- :func:`link_trees` links TLS stems to ALS trees, reports the stems
  under another tree's crown, and gives one table of TLS diameters and ALS
  heights;
- :func:`merge_clouds` and :func:`fuse_profiles` give one point cloud and
  one plant area density profile from both instruments;
- :func:`plot_values`, :func:`fit_model` and :func:`upscale` carry plot
  values from the TLS over the ALS survey with a regression on area-based
  metrics;
- :func:`synthetic_scan` is a terrestrial scanner for synthetic scenes that
  sees them as :func:`sylva.synthetic.als_flight` does, so both instruments
  can be checked against one known forest.

The computations are in the Rust core (``sylva_rs::fusion``).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .raster import Raster
from .shots import Shots

__all__ = ["Registration", "register", "TreeLinks", "link_trees", "merge_clouds", "FusedProfile",
           "fuse_profiles", "plot_values", "Model", "fit_model", "Upscaling", "upscale",
           "synthetic_scan"]

GROUND = 2


# ------------------------------------------------------------------ helpers

def _matrix(transform) -> np.ndarray | None:
    if transform is None:
        return None
    if isinstance(transform, Registration):
        return transform.transform
    m = np.asarray(transform, dtype=float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError(f"transform must be a finite (4, 4) matrix or a Registration, got shape {m.shape}")
    return m


def _ground(cloud: PointCloud, mask, what: str) -> np.ndarray:
    if mask is not None:
        g = np.ascontiguousarray(mask, dtype=bool).ravel()
        if len(g) != len(cloud):
            raise ValueError(f"{what}_ground has {len(g)} values for {len(cloud)} points")
        return g
    if "classification" not in cloud.attrs:
        raise ValueError(f"the {what} cloud has no 'classification' attribute; classify its ground first "
                         f"(sylva.ground.classify_ground_csf) or pass {what}_ground")
    return np.ascontiguousarray(np.asarray(cloud.attrs["classification"]) == GROUND)


def _transform_xyz(m: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return xyz @ m[:3, :3].T + m[:3, 3]


# ------------------------------------------------------------- registration

@dataclass
class Registration:
    """A TLS plot placed on an ALS survey, from :func:`register`.

    Attributes
    ----------
    transform
        ``(4, 4)`` matrix taking TLS coordinates into the ALS frame
        (``initial`` included).
    search_transform
        The same after the search, before the ICP refinement: the best
        peak.
    pivot
        Centre of the TLS ground after ``initial`` (ALS frame): the point the
        heading turns about, and to which ``shift`` and the uncertainties
        refer.
    heading
        Rotation about the vertical (degrees, counter-clockwise) added to
        ``initial``.
    shift
        Translation (m) of the pivot added to ``initial``.
    candidates
        The peaks of the search, best first, refined: ``heading``, ``dx``,
        ``dy``, ``score`` (the combined correlation), ``chm_r`` and
        ``dtm_r`` (Pearson's r of the canopy and terrain models) and
        ``overlap`` (TLS canopy cells on ALS data).
    ambiguity
        Misfit (1 - score) of the best peak over that of the second best: 0
        for a clear answer, near 1 when another pose fits about as well (NaN
        with one peak).
    icp
        Summary of the kept ICP run (``fitness``, ``rmse``, ``n``,
        ``iterations``, ``converged``, ``accepted``, ``shift`` and ``turn``
        from its peak, and ``start``, the index of that peak in
        ``candidates``), or None without refinement.
    covariance
        Formal ``(6, 6)`` covariance of ``[rx, ry, rz, tx, ty, tz]``
        (radians, m) at the pivot, from the ICP's point-to-plane information.
        It treats every residual as independent, so it is a lower bound.
    jackknife
        Jackknife standard errors of the pivot's x, y, z (m) and the heading
        (degrees) over the four quadrants of the plot, or None.
    residuals
        ``ground_n``, ``ground_median``, ``ground_nmad``, ``ground_rmse``
        (TLS ground minus the ALS DTM, m) and ``canopy_n``,
        ``canopy_median``, ``canopy_p90`` (distance of TLS canopy points to
        the nearest ALS canopy return, m).
    n_tls, n_als
        Points used (the ALS within reach of the plot).
    resolution, coarse_resolution
        Cell sizes (m) of the fine and coarse search, after widening for a
        sparse survey.
    als_density
        ALS returns per m² near the plot.
    settings
        The settings of the call.
    """

    transform: np.ndarray
    search_transform: np.ndarray
    pivot: np.ndarray
    heading: float
    shift: np.ndarray
    candidates: dict
    ambiguity: float
    icp: dict | None
    covariance: np.ndarray | None
    jackknife: np.ndarray | None
    residuals: dict
    n_tls: int = 0
    n_als: int = 0
    resolution: float = float("nan")
    coarse_resolution: float = float("nan")
    als_density: float = float("nan")
    settings: dict = field(default_factory=dict)

    @property
    def std(self) -> dict:
        """Standard errors of the pivot's ``x``, ``y``, ``z`` (m) and the
        ``heading`` (degrees): the larger of the formal and the jackknife
        ones, NaN where neither exists."""
        formal = np.full(4, np.nan)
        if self.covariance is not None:
            d = np.sqrt(np.clip(np.diag(self.covariance), 0, None))
            formal = np.array([d[3], d[4], d[5], np.degrees(d[2])])
        jk = np.full(4, np.nan) if self.jackknife is None else np.asarray(self.jackknife)
        both = np.fmax(formal, jk)
        return dict(zip(("x", "y", "z", "heading"), both.tolist(), strict=True))

    def apply(self, cloud: PointCloud) -> PointCloud:
        """The TLS cloud moved into the ALS frame.

        Parameters
        ----------
        cloud
            Points in the TLS frame given to :func:`register`.

        Returns
        -------
        PointCloud
        """
        return cloud.transform(self.transform)

    def to_dict(self) -> dict:
        """Everything as plain lists and numbers, for JSON."""
        def plain(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            return v
        return {k: plain(getattr(self, k)) for k in self.__dataclass_fields__}

    def save(self, path: str | Path) -> None:
        """Write :meth:`to_dict` as JSON (NaN written as null).

        Parameters
        ----------
        path
            Output file.
        """
        def clean(v):
            if isinstance(v, float) and not np.isfinite(v):
                return None
            if isinstance(v, list):
                return [clean(x) for x in v]
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            return v
        Path(path).write_text(json.dumps(clean(self.to_dict()), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Registration:
        """Read a registration written by :meth:`save`.

        Parameters
        ----------
        path
            JSON file.

        Returns
        -------
        Registration
        """
        d = json.loads(Path(path).read_text())

        def arr(v):
            return None if v is None else np.array([np.nan if x is None else x for x in np.ravel(v)],
                                                   dtype=float).reshape(np.shape(v))
        nan = float("nan")
        return cls(
            transform=arr(d["transform"]), search_transform=arr(d["search_transform"]),
            pivot=arr(d["pivot"]), heading=nan if d["heading"] is None else d["heading"],
            shift=arr(d["shift"]), candidates={k: arr(v) for k, v in d["candidates"].items()},
            ambiguity=nan if d["ambiguity"] is None else d["ambiguity"], icp=d["icp"],
            covariance=arr(d["covariance"]), jackknife=arr(d["jackknife"]),
            residuals={k: nan if v is None else v for k, v in d["residuals"].items()},
            n_tls=d.get("n_tls", 0), n_als=d.get("n_als", 0),
            resolution=nan if d.get("resolution") is None else d["resolution"],
            coarse_resolution=nan if d.get("coarse_resolution") is None else d["coarse_resolution"],
            als_density=nan if d.get("als_density") is None else d["als_density"], settings=d.get("settings", {}))

    def report(self) -> str:
        """A short text summary of the pose, its checks and its uncertainty."""
        c = self.candidates
        s = self.std
        r = self.residuals
        lines = [
            "TLS to ALS registration",
            f"  pose       heading {self.heading:+.2f} deg, shift x {self.shift[0]:+.3f} y {self.shift[1]:+.3f} "
            f"z {self.shift[2]:+.3f} m about ({self.pivot[0]:.2f}, {self.pivot[1]:.2f}, {self.pivot[2]:.2f})",
            f"  search     score {c['score'][0]:.3f} (CHM r {c['chm_r'][0]:.3f}, DTM r {c['dtm_r'][0]:.3f}), "
            f"{int(c['overlap'][0])} cells of {self.resolution:g} m, ambiguity {self.ambiguity:.2f}",
        ]
        if self.icp is None:
            lines.append("  icp        not run")
        else:
            i = self.icp
            lines.append(f"  icp        {'accepted' if i['accepted'] else 'REJECTED'}: fitness {i['fitness']:.3f}, "
                         f"rmse {i['rmse']:.3f} m, {i['n']} pairs, moved {i['shift']:.3f} m and {i['turn']:.2f} deg "
                         f"from peak {i.get('start', 0) + 1}")
        lines.append(f"  ground     TLS minus ALS DTM: median {r['ground_median']:+.3f} m, NMAD {r['ground_nmad']:.3f} m "
                     f"({r['ground_n']} points)")
        lines.append(f"  canopy     to the nearest ALS return: median {r['canopy_median']:.3f} m, 90 % "
                     f"{r['canopy_p90']:.3f} m ({r['canopy_n']} points)")
        lines.append(f"  std error  x {s['x']:.3f} m, y {s['y']:.3f} m, z {s['z']:.3f} m, heading {s['heading']:.3f} deg")
        if np.isfinite(self.ambiguity) and self.ambiguity > 0.8:
            lines.append("  warning    a second pose scores almost as well; check the result")
        return "\n".join(lines)


def _als_points(als, reach_box):
    from .als import Catalog

    if isinstance(als, PointCloud):
        return als
    if isinstance(als, (str, Path, list, tuple)) and not isinstance(als, Catalog):
        from .als import catalog
        als = catalog(als)
    if isinstance(als, Catalog):
        return als.read(reach_box)
    raise ValueError(f"als must be a PointCloud or a catalogue, got {type(als).__name__}")


def register(tls: PointCloud, als, initial=None, *, resolution: float = 0.5,
             returns_per_cell: float = 10.0, coarse_resolution: float = 2.0, search_radius: float = 10.0,
             heading_range: float = 180.0, heading_step: float = 3.0, dtm_weight: float = 0.5,
             min_height: float = 2.0, min_overlap: float = 0.5, n_candidates: int = 3,
             refine: str = "all", icp_voxel_sizes=(1.0, 0.5, 0.25),
             icp_max_distances=(2.0, 1.0, 0.5), min_planarity: float = 0.3,
             max_refine_shift: float = 3.0, max_refine_turn: float = 8.0, jackknife: bool = True,
             tls_ground=None,
             als_ground=None) -> Registration:
    """Register a TLS plot onto an ALS survey.

    **Search.** The canopy height model (CHM) and terrain model (DTM) of
    each cloud are made on one grid spacing, from its ground points
    (classification 2). The TLS cells are turned by a heading and shifted
    over the ALS rasters, and each pose is scored by the correlation
    (Pearson's r) of the two CHMs plus ``dtm_weight`` times that of the two
    DTMs, over the cells both measured. A correlation is blind to a
    vertical offset between the terrains and to a TLS canopy that reads low
    where the scanner saw less of the crowns. Every heading within
    ``heading_range`` of the initial one (``heading_step`` apart) and every
    shift within ``search_radius`` (``coarse_resolution`` apart) is scored;
    the ``n_candidates`` best separated peaks are searched again at
    ``resolution`` with bilinear sampling and a parabola through the best
    score in heading, x and y, and the vertical offset is the median
    difference of the two DTMs. For a sparse survey the fine cells are
    widened to hold about ``returns_per_cell`` returns each (1 m at 10
    returns per m²): a CHM of cells with one or two returns is too ragged
    to place a plot by.

    **Refinement.** A robust point-to-plane ICP (Huber weights, trimmed
    tail; the ICP of :mod:`sylva.coreg`) moves the TLS points onto the ALS
    points (``refine="all"``: ground, stems and crowns; ``"ground"``: the
    ground alone; ``"none"``: no refinement), over the voxel pyramid
    ``icp_voxel_sizes``, from each refined peak. A run counts only if it
    moves less than ``max_refine_shift`` and ``max_refine_turn`` from its
    peak; of those the best fit is kept (the most TLS points within the
    last cut-off of an ALS point; the higher peak when two fit within half
    a percent). The 3-D fit of ground and crowns tells apart poses that the
    canopy models cannot: the TLS sees the crowns from below and to the
    side, so the best correlation of the canopy models can lie a few
    degrees or metres off, most of all with one or two scan positions.

    **Uncertainty.** The formal covariance of the ICP (``sigma² H⁻¹`` of
    its point-to-plane information: a lower bound, since the residuals are
    not independent) and a jackknife over the four quadrants of the plot
    (each left out and the ICP run again). The residuals of the TLS ground
    against the ALS DTM and of the TLS canopy against the nearest ALS
    returns check the result without any reference.

    Everything is computed about a pivot near the plot, so map coordinates
    of any size lose no precision.

    Parameters
    ----------
    tls
        The TLS plot, with ground classified (or ``tls_ground``).
    als
        The ALS returns with ground classified (or ``als_ground``), as a
        :class:`~sylva.PointCloud`, or a catalogue (anything
        :func:`sylva.als.catalog` accepts), of which only the box around the
        plot is read.
    initial
        ``(4, 4)`` matrix placing the TLS roughly in the ALS frame: a GNSS
        position and a compass heading, or the identity (the default) for a
        plot already georeferenced.
    resolution
        Smallest cell size (m) of the fine search and the residual DTM.
    returns_per_cell
        ALS returns a fine cell should hold on average; 0 keeps
        ``resolution`` whatever the density.
    coarse_resolution
        Cell size (m) of the search over the whole window.
    search_radius
        Largest horizontal shift (m) searched, from the initial position.
        A few metres for a GNSS position; the uncertainty of a hand-held
        position otherwise.
    heading_range
        Headings searched either side of the initial one (degrees). 180 (the
        default) searches the full circle, for a plot in a scanner frame; a
        few degrees for a georeferenced one.
    heading_step
        Heading step (degrees) of the coarse search.
    dtm_weight
        Weight of the terrain correlation against the canopy's. Terrain
        relief constrains a pose where the canopy is uniform.
    min_height
        CHM heights below this (m) count as 0.
    min_overlap
        Share of the TLS canopy cells that must fall on ALS data for a pose
        to be scored.
    n_candidates
        Peaks of the coarse search refined.
    refine : {"all", "ground", "none"}
        Points the ICP uses.
    icp_voxel_sizes, icp_max_distances
        ICP voxel pyramid and correspondence cut-offs (m), coarse to fine.
    min_planarity
        Planarity an ALS point needs to be an ICP target (0 to 1).
    max_refine_shift, max_refine_turn
        Largest horizontal move (m) and turn (degrees) the ICP may make from
        the search result.
    jackknife
        Run the quadrant jackknife (four more ICPs).
    tls_ground, als_ground
        Boolean ground masks, in place of ``classification == 2``
        (``als_ground`` only with an ALS point cloud).

    Returns
    -------
    Registration

    Raises
    ------
    ValueError
        For bad settings, clouds without ground points, or no pose that puts
        enough of the TLS canopy on ALS data.

    Examples
    --------
    >>> reg = fusion.register(tls, als_catalog, initial=gnss_pose, search_radius=5.0,
    ...                       heading_range=5.0)                        # doctest: +SKIP
    >>> print(reg.report())                                           # doctest: +SKIP
    >>> tls_map = reg.apply(tls)                                      # doctest: +SKIP
    """
    if not isinstance(tls, PointCloud):
        raise ValueError(f"tls must be a PointCloud, got {type(tls).__name__}")
    if len(tls) == 0:
        raise ValueError("the TLS cloud is empty")
    init = np.eye(4) if initial is None else _matrix(initial)
    tg = _ground(tls, tls_ground, "tls")
    # The box of the ALS that can matter: the corners of the TLS box, placed.
    b0, b1 = tls.xyz.min(axis=0), tls.xyz.max(axis=0)
    corners = np.array([[x, y, z] for x in (b0[0], b1[0]) for y in (b0[1], b1[1]) for z in (b0[2], b1[2])])
    placed = _transform_xyz(init, corners)
    lo, hi = placed[:, :2].min(axis=0), placed[:, :2].max(axis=0)
    centre = (lo + hi) / 2
    reach = float(np.max(hi - lo)) * np.sqrt(2) / 2 + float(search_radius) + 2 * float(coarse_resolution) + 10.0
    box = (centre[0] - reach, centre[1] - reach, centre[0] + reach, centre[1] + reach)
    if als_ground is not None and not isinstance(als, PointCloud):
        raise ValueError("als_ground needs the ALS as a PointCloud; a catalogue's tiles carry their classes")
    cloud = _als_points(als, box)
    if len(cloud) == 0:
        raise ValueError("no ALS points near the TLS plot; check `initial`")
    ag = _ground(cloud, als_ground, "als")
    vox = [float(v) for v in icp_voxel_sizes]
    dist = [float(v) for v in icp_max_distances]
    settings = dict(resolution=resolution, returns_per_cell=returns_per_cell,
                    coarse_resolution=coarse_resolution,
                    search_radius=search_radius, heading_range=heading_range,
                    heading_step=heading_step, dtm_weight=dtm_weight, min_height=min_height,
                    min_overlap=min_overlap, n_candidates=int(n_candidates), refine=refine,
                    icp_voxel_sizes=vox, icp_max_distances=dist, min_planarity=min_planarity,
                    max_refine_shift=max_refine_shift, max_refine_turn=max_refine_turn,
                    jackknife=bool(jackknife))
    d = _core.fusion_register(np.ascontiguousarray(tls.xyz), tg, np.ascontiguousarray(cloud.xyz), ag,
                              np.ascontiguousarray(init), float(resolution), float(returns_per_cell),
                              float(coarse_resolution),
                              float(search_radius), float(heading_range), float(heading_step),
                              float(dtm_weight), float(min_height), float(min_overlap),
                              int(n_candidates), str(refine), vox, dist, float(min_planarity),
                              float(max_refine_shift), float(max_refine_turn), bool(jackknife))
    return Registration(d["transform"], d["search_transform"], np.asarray(d["pivot"]), float(d["heading"]),
                        np.asarray(d["shift"]), dict(d["candidates"]), float(d["ambiguity"]), d["icp"],
                        d["covariance"], d["jackknife"], dict(d["residuals"]), int(d["n_tls"]),
                        int(d["n_als"]), float(d["resolution"]), float(d["coarse_resolution"]),
                        float(d["als_density"]), settings)


# ------------------------------------------------------------------- trees

def _table(trees, what: str, columns: tuple) -> dict:
    """A dict of arrays from a table, a list of objects or an ``als.Trees``."""
    if isinstance(trees, Mapping):
        if "x" not in trees or "y" not in trees:
            raise ValueError(f"{what} needs 'x' and 'y' columns")
        n = len(np.asarray(trees["x"]))
        out = {k: np.asarray(v) for k, v in trees.items()}
        for k, v in out.items():
            if k != "crowns" and v.ndim == 1 and len(v) != n:
                raise ValueError(f"column {k!r} of {what} has {len(v)} values for {n} trees")
        return out
    if hasattr(trees, "table") and hasattr(trees, "crowns"):       # als.Trees
        t = {k: np.asarray(v) for k, v in trees.table().items()}
        t["crowns"] = list(trees.crowns)
        return t
    rows = list(trees)
    out = {}
    for c in columns:
        vals = [getattr(t, c, None) for t in rows]
        if c in ("x", "y") and any(v is None for v in vals):
            raise ValueError(f"{what} must be trees with x and y")
        if all(v is not None for v in vals):
            out[c] = np.asarray(vals)
    if "x" not in out:
        out["x"], out["y"] = np.zeros(0), np.zeros(0)
    return out


def _volumes(volumes, ids: np.ndarray, n: int) -> np.ndarray:
    if volumes is None:
        return np.full(n, np.nan)
    if hasattr(volumes, "volume") and hasattr(volumes, "models"):    # qsm.PlotQSMs
        return np.array([volumes.volume(int(i)) if int(i) in volumes.models else np.nan for i in ids])
    if isinstance(volumes, Mapping):
        return np.array([float(volumes.get(int(i), np.nan)) for i in ids])
    v = np.asarray(volumes, dtype=float).ravel()
    if len(v) != n:
        raise ValueError(f"{len(v)} volumes for {n} TLS trees")
    return v


@dataclass
class TreeLinks:
    """TLS trees linked to ALS trees, from :func:`link_trees`.

    Attributes
    ----------
    tls
        The TLS trees as a table: ``tree_id``, ``x``, ``y`` (in the ALS
        frame), ``dbh``, ``height`` (TLS), ``volume``.
    als
        The ALS trees as a table: ``id``, ``x``, ``y``, ``height`` and
        ``crowns`` when given.
    status
        Per TLS tree: ``matched`` (it is the ALS tree), ``suppressed``
        (under the crown of a taller ALS tree, which the ALS cannot see
        past), ``codominant`` (under an ALS crown whose tree is another
        stem of about the same height: the ALS merged two canopy trees) or
        ``unlinked`` (under no ALS crown and near no top).
    als_index
        Index into ``als`` of the tree each TLS tree is linked to or stands
        under; -1 for none.
    distance
        Stem to that tree's top (m).
    inside
        Whether the stem is inside that crown.
    als_height
        That ALS tree's height.
    height
        Combined height: for a matched tree the ALS height, unless the TLS
        is known to have seen the top and measured it taller; the TLS
        height otherwise.
    height_source
        ``"als"``, ``"tls"`` or ``"none"``.
    top_seen
        Whether the TLS saw the top: 1, 0 or NaN (unknown). Given, or for a
        matched tree inferred from the TLS height reaching the ALS height.
    flag
        ``top_seen``; ``als_height`` (TLS top not seen, ALS height used);
        ``top_not_seen`` (not seen and no ALS height: the height is a lower
        bound); ``top_unknown``.
    cost
        Assignment cost of each matched tree.
    als_tls
        Per ALS tree, the index of its matched TLS tree (-1 for none).
    als_stems
        Per ALS tree, the indices of every TLS tree linked to it or under it.
    settings
        The settings of the call.
    """

    tls: dict
    als: dict
    status: np.ndarray
    als_index: np.ndarray
    distance: np.ndarray
    inside: np.ndarray
    als_height: np.ndarray
    height: np.ndarray
    height_source: np.ndarray
    top_seen: np.ndarray
    flag: np.ndarray
    cost: np.ndarray
    als_tls: np.ndarray
    als_stems: list
    settings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        c = self.counts()
        return (f"TreeLinks({len(self.status)} TLS trees: {c['matched']} matched, {c['suppressed']} "
                f"suppressed, {c['codominant']} codominant, {c['unlinked']} unlinked)")

    def counts(self) -> dict:
        """Numbers of TLS trees by status, of ALS trees with a matched stem
        and without, and of ALS crowns holding two stems or more."""
        out = {s: int(np.sum(self.status == s)) for s in ("matched", "suppressed", "codominant", "unlinked")}
        out["als_matched"] = int(np.sum(self.als_tls >= 0))
        out["als_unmatched"] = int(np.sum(self.als_tls < 0))
        out["one_to_many"] = int(sum(len(s) >= 2 for s in self.als_stems))
        return out

    def one_to_many(self) -> dict:
        """ALS trees with more than one TLS stem under them.

        Returns
        -------
        dict
            ALS tree id to the TLS tree ids under it, the matched one first.
        """
        out = {}
        for j, stems in enumerate(self.als_stems):
            if len(stems) >= 2:
                main = int(self.als_tls[j])
                order = [main] + [int(i) for i in stems if int(i) != main] if main >= 0 else [int(i) for i in stems]
                out[int(self.als["id"][j])] = [int(self.tls["tree_id"][i]) for i in order]
        return out

    def table(self) -> dict:
        """One row per TLS tree: its TLS measurements, its link and the
        combined height.

        Returns
        -------
        dict
            ``tree_id``, ``x``, ``y``, ``dbh``, ``height_tls``, ``volume``,
            ``status``, ``als_id`` (-1 for none), ``als_height``,
            ``distance``, ``inside``, ``height``, ``height_source``,
            ``top_seen``, ``flag``.
        """
        ids = np.asarray(self.als["id"])
        return {
            "tree_id": self.tls["tree_id"], "x": self.tls["x"], "y": self.tls["y"],
            "dbh": self.tls["dbh"], "height_tls": self.tls["height"], "volume": self.tls["volume"],
            "status": self.status,
            "als_id": np.where(self.als_index >= 0, ids[np.clip(self.als_index, 0, None)] if len(ids) else -1, -1),
            "als_height": self.als_height, "distance": self.distance, "inside": self.inside,
            "height": self.height, "height_source": self.height_source, "top_seen": self.top_seen,
            "flag": self.flag,
        }

    def als_table(self) -> dict:
        """One row per ALS tree: ``id``, ``x``, ``y``, ``height``,
        ``tls_id`` (its matched TLS tree, -1 for none), ``n_stems`` (TLS
        stems linked to it or under it) and ``stems`` (their ids)."""
        tid = np.asarray(self.tls["tree_id"])
        return {"id": self.als["id"], "x": self.als["x"], "y": self.als["y"], "height": self.als["height"],
                "tls_id": np.array([tid[i] if i >= 0 else -1 for i in self.als_tls], dtype=np.int64),
                "n_stems": np.array([len(s) for s in self.als_stems], dtype=np.int64),
                "stems": [tid[np.asarray(s, dtype=int)].tolist() for s in self.als_stems]}

    def to_csv(self, path: str | Path) -> None:
        """Write :meth:`table` as CSV (empty cells for NaN).

        Parameters
        ----------
        path
            Output file.
        """
        t = self.table()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(t))
            for row in zip(*t.values(), strict=True):
                w.writerow(["" if isinstance(v, float) and not np.isfinite(v) else
                            (v.item() if isinstance(v, np.generic) else v) for v in row])


def link_trees(tls_trees, als_trees, transform=None, *, volumes=None, top_seen=None, sampling=None,
               min_above_observed: float = 0.5, max_distance: float = 3.0,
               crown_buffer: float = 0.5, height_weight: float = 2.0, dbh_weight: float = 2.0,
               max_cost: float = 2.0, top_tolerance: float = 1.0) -> TreeLinks:
    """Link TLS stems to ALS trees and combine their measurements.

    A TLS tree and an ALS tree can be one tree when the stem lies inside the
    ALS crown (or within ``crown_buffer`` of its outline) or within
    ``max_distance`` of its top. Among these pairs an optimal assignment
    (Kuhn 1955; Munkres 1957) minimises the total cost, a stem left without
    an ALS tree costing ``max_cost``. A pair costs::

        (d / D)² + height_weight * dh² + dbh_weight * (1 - dbh / dbh_max)²

    with ``d`` the stem-to-top distance, ``D`` the larger of
    ``max_distance`` and the crown's equivalent radius, ``dh = (h_als -
    h_tls) / h_als`` (0.3 for a stem without a TLS height) and ``dbh_max``
    the largest DBH among the stems that could be that ALS tree. The stem
    that is tallest by the TLS and thickest under a crown is therefore its
    tree, as the dominant tree of a crown usually is both. Each ALS tree
    gets at most one stem, and the others under its crown are reported with
    it as ``suppressed`` (shorter by more than ``top_tolerance`` and 10 %)
    or ``codominant`` (about as tall: the ALS saw two canopy trees as one).
    :meth:`TreeLinks.one_to_many` lists these crowns.

    The combined table keeps the TLS diameter and takes the height from the
    instrument that saw the top. Whether the TLS saw a tree's top comes from
    ``top_seen``, from ``sampling`` (:func:`sylva.voxels.tree_sampling`:
    seen when ``above_observed_fraction`` is at least
    ``min_above_observed``), or, for a matched tree, from its TLS height
    reaching the ALS height within ``top_tolerance``. A matched tree takes
    the ALS height unless the TLS is known (from ``top_seen`` or
    ``sampling``) to have seen the top and measured it taller: both are
    lower bounds, the ALS for missing the apex and the TLS for occlusion,
    but a TLS height can also be too tall where its segmentation gave the
    tree part of a neighbour's crown. Other trees keep the TLS height,
    flagged ``top_not_seen`` when that is a lower bound.

    Parameters
    ----------
    tls_trees
        :class:`sylva.trees.Tree` objects (after
        :func:`sylva.trees.tree_heights`), or a table (dict of arrays) with
        ``x``, ``y`` and optionally ``tree_id``, ``dbh``, ``height``,
        ``volume``, ``z``, in the TLS frame.
    als_trees
        :class:`sylva.als.Trees` (with crowns), or a table with ``x``,
        ``y``, ``height`` and optionally ``id`` and ``crowns`` (a list of
        ``(k, 2)`` outlines).
    transform
        :class:`Registration` or ``(4, 4)`` matrix taking the TLS into the
        ALS frame; the trees are taken to share a frame if None. Stem
        positions are moved at ``z`` (0 if not given).
    volumes
        Wood volume (m³) per TLS tree: a :class:`sylva.qsm.PlotQSMs`, a
        mapping of tree id to volume, or an array in the order of the trees.
    top_seen
        Per TLS tree, whether its top was seen (booleans, or 1, 0 and NaN).
    sampling
        Output of :func:`sylva.voxels.tree_sampling` (matched by
        ``tree_id``), in place of ``top_seen``.
    min_above_observed
        ``above_observed_fraction`` from which a top counts as seen.
    max_distance
        Farthest a stem can be from an ALS top (m) to be its tree outside
        the crown outline, and the scale of the distance cost.
    crown_buffer
        Distance (m) outside a crown outline that still counts as under it.
    height_weight
        Weight of the relative height difference in the cost.
    dbh_weight
        Weight of the DBH shortfall from the thickest candidate stem.
    max_cost
        Cost of leaving a stem without an ALS tree; no pair costing more is
        made.
    top_tolerance
        Height difference (m) within which a TLS tree reaches the ALS
        height.

    Returns
    -------
    TreeLinks

    Raises
    ------
    ValueError
        For a non-finite position, mismatched columns or a setting out of
        range.

    Examples
    --------
    >>> links = fusion.link_trees(stems, als_trees, transform=reg, volumes=qsms)  # doctest: +SKIP
    >>> links.one_to_many()                          # {als id: [tls ids]}         # doctest: +SKIP
    >>> links.to_csv("trees_fused.csv")                                           # doctest: +SKIP
    """
    t = _table(tls_trees, "tls_trees", ("tree_id", "x", "y", "dbh", "height", "volume", "z"))
    a = _table(als_trees, "als_trees", ("id", "x", "y", "height", "crowns"))
    n, m = len(np.asarray(t["x"])), len(np.asarray(a["x"]))
    col = lambda d, k, size: np.asarray(d[k], dtype=float) if k in d else np.full(size, np.nan)  # noqa: E731
    ids = np.asarray(t["tree_id"]) if "tree_id" in t else np.arange(1, n + 1)
    x, y = col(t, "x", n), col(t, "y", n)
    mat = _matrix(transform)
    if mat is not None and n:
        z = col(t, "z", n)
        xyz = _transform_xyz(mat, np.column_stack([x, y, np.nan_to_num(z)]))
        x, y = xyz[:, 0], xyz[:, 1]
    vol = col(t, "volume", n) if "volume" in t and volumes is None else _volumes(volumes, ids, n)
    if sampling is not None and top_seen is not None:
        raise ValueError("give top_seen or sampling, not both")
    if sampling is not None:
        frac = dict(zip(np.asarray(sampling["tree_id"]).tolist(),
                        np.asarray(sampling["above_observed_fraction"], dtype=float).tolist(), strict=True))
        f = np.array([frac.get(int(i), np.nan) for i in ids])
        seen = np.where(np.isfinite(f), (f >= min_above_observed).astype(float), np.nan)
    elif top_seen is not None:
        seen = np.asarray(top_seen, dtype=float).ravel()
        if len(seen) != n:
            raise ValueError(f"top_seen has {len(seen)} values for {n} TLS trees")
    else:
        seen = np.full(n, np.nan)
    tls_arr = np.column_stack([x, y, col(t, "dbh", n), col(t, "height", n), vol, seen]).reshape(-1, 6)
    als_ids = np.asarray(a["id"]) if "id" in a else np.arange(1, m + 1)
    als_arr = np.column_stack([col(a, "x", m), col(a, "y", m), col(a, "height", m)]).reshape(-1, 3)
    crowns = [np.ascontiguousarray(np.asarray(c, dtype=float).reshape(-1, 2)) for c in a["crowns"]] \
        if "crowns" in a else [np.zeros((0, 2)) for _ in range(m)]
    d = _core.fusion_link_trees(np.ascontiguousarray(tls_arr), np.ascontiguousarray(als_arr), crowns,
                                float(max_distance), float(crown_buffer), float(height_weight),
                                float(dbh_weight), float(max_cost), float(top_tolerance))
    tls_table = {"tree_id": ids, "x": x, "y": y, "dbh": col(t, "dbh", n), "height": col(t, "height", n),
                 "volume": vol}
    als_table = {"id": als_ids, "x": als_arr[:, 0], "y": als_arr[:, 1], "height": als_arr[:, 2]}
    if "crowns" in a:
        als_table["crowns"] = list(a["crowns"])
    return TreeLinks(
        tls=tls_table, als=als_table, status=np.array(d["status"], dtype=object).astype(str),
        als_index=d["als"], distance=d["distance"], inside=d["inside"], als_height=d["als_height"],
        height=d["height"], height_source=np.array(d["height_source"], dtype=object).astype(str),
        top_seen=d["top_seen"], flag=np.array(d["flag"], dtype=object).astype(str), cost=d["cost"],
        als_tls=d["als_tls"], als_stems=list(d["als_stems"]),
        settings={"max_distance": max_distance, "crown_buffer": crown_buffer,
                  "height_weight": height_weight, "dbh_weight": dbh_weight, "max_cost": max_cost,
                  "top_tolerance": top_tolerance,
                  "min_above_observed": min_above_observed})


# ---------------------------------------------------------- clouds, profiles

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


# ---------------------------------------------------------------- upscaling

def plot_values(trees, area: float, wood_density: float, *, volumes=None, min_dbh: float = 0.0) -> dict:
    """Plot totals per hectare from a plot's TLS trees.

    Parameters
    ----------
    trees
        :class:`sylva.trees.Tree` objects, a table (dict of arrays) with
        ``dbh`` and optionally ``volume`` and ``tree_id``, or a
        :class:`TreeLinks` (its TLS trees).
    area
        Plot area (m²). Give only the trees inside the plot.
    wood_density
        Basic wood density (kg/m³), e.g. 500 to 700 for many hardwoods;
        there is no default, since it depends on the species.
    volumes
        Wood volume (m³) per tree, as for :func:`link_trees`, if the trees
        do not carry a ``volume``.
    min_dbh
        Inventory threshold (m): smaller trees are left out.

    Returns
    -------
    dict
        ``agb`` (Mg/ha: volume times wood density), ``volume`` (m³/ha),
        ``basal_area`` (m²/ha), ``stems`` (per ha), ``n_trees`` and
        ``missing_volume`` (trees counted without a volume).

    Raises
    ------
    ValueError
        For a non-positive area or wood density.
    """
    if isinstance(trees, TreeLinks):
        t = trees.tls
    elif isinstance(trees, Mapping):
        t = {k: np.asarray(v) for k, v in trees.items()}
    else:
        t = _table(trees, "trees", ("tree_id", "x", "y", "dbh", "volume"))
    if "dbh" not in t:
        raise ValueError("the trees need a 'dbh' (m)")
    dbh = np.asarray(t["dbh"], dtype=float).ravel()
    n = len(dbh)
    ids = np.asarray(t["tree_id"]) if "tree_id" in t else np.arange(1, n + 1)
    vol = _volumes(volumes, ids, n) if volumes is not None else \
        (np.asarray(t["volume"], dtype=float) if "volume" in t else np.full(n, np.nan))
    return dict(_core.fusion_plot_summary(np.ascontiguousarray(dbh), np.ascontiguousarray(vol), float(area),
                                          float(wood_density), float(min_dbh)))


def _predictor_matrix(predictors, names) -> tuple[np.ndarray, list[str], tuple | None]:
    """Stack predictors (arrays or Rasters on one grid) as columns."""
    if isinstance(predictors, Mapping) or hasattr(predictors, "columns"):
        src = predictors.columns if hasattr(predictors, "columns") and not isinstance(predictors, Mapping) \
            else predictors
        names = list(src) if names is None else list(names)
        missing = [n for n in names if n not in src]
        if missing:
            raise ValueError(f"no predictor {missing[0]!r}; have {sorted(src)}")
        cols = [src[n] for n in names]
    else:
        a = np.asarray(predictors, dtype=float)
        a = a.reshape(-1, 1) if a.ndim == 1 else a
        names = [f"x{k + 1}" for k in range(a.shape[1])] if names is None else list(names)
        if len(names) != a.shape[1]:
            raise ValueError(f"{len(names)} names for {a.shape[1]} predictors")
        cols = [a[:, k] for k in range(a.shape[1])]
    grid = None
    if cols and all(isinstance(c, Raster) for c in cols):
        r0 = cols[0]
        for c in cols[1:]:
            if c.shape != r0.shape or c.xmin != r0.xmin or c.ymin != r0.ymin or c.resolution != r0.resolution:
                raise ValueError("predictor rasters must share one grid")
        grid = (r0.shape, r0.xmin, r0.ymin, r0.resolution, r0.crs)
        cols = [c.data.ravel() for c in cols]
    elif any(isinstance(c, Raster) for c in cols):
        raise ValueError("give every predictor as a Raster or none")
    x = np.column_stack([np.asarray(c, dtype=float).ravel() for c in cols]) if cols else np.zeros((0, 0))
    return np.ascontiguousarray(x), names, grid


@dataclass
class Model:
    """A regression of plot values on ALS metrics, from :func:`fit_model`.

    Attributes
    ----------
    form
        ``"linear"`` (``y = b0 + Σ bk xk``) or ``"loglog"``
        (``ln y = b0 + Σ bk ln xk``).
    names
        Predictor names, in the order of ``coef[1:]``.
    coef, se
        Coefficients (intercept first) and their standard errors, on the
        model's scale.
    sigma, df
        Residual standard error (model scale) and its degrees of freedom.
    n
        Plots.
    r2, adj_r2
        Coefficient of determination on the model's scale, and adjusted.
    xtx_inv
        ``(XᵀX)⁻¹`` for prediction variances.
    correction
        Back-transformation factor ``exp(σ²/2)`` of a loglog model
        (Baskerville 1972); 1 for linear.
    y, fitted, loo
        Plot values, fitted values and leave-one-out predictions (scale of y).
    loo_rmse, loo_bias, loo_r2, loo_rrmse
        Leave-one-out RMSE, mean error (prediction minus observation),
        ``1 - PRESS / SS`` and RMSE as a percentage of the mean of y.
    x_min, x_max
        Range of each predictor over the plots.
    """

    form: str
    names: list
    coef: np.ndarray
    se: np.ndarray
    sigma: float
    df: int
    n: int
    r2: float
    adj_r2: float
    xtx_inv: np.ndarray
    correction: float
    y: np.ndarray
    fitted: np.ndarray
    loo: np.ndarray
    loo_rmse: float
    loo_bias: float
    loo_r2: float
    loo_rrmse: float
    x_min: np.ndarray
    x_max: np.ndarray

    def equation(self) -> str:
        """The fitted model as text."""
        if self.form == "loglog":
            terms = " + ".join(f"{b:.4g} ln({n})" for b, n in zip(self.coef[1:], self.names, strict=True))
            return f"ln y = {self.coef[0]:.4g} + {terms}"
        terms = " + ".join(f"{b:.4g} {n}" for b, n in zip(self.coef[1:], self.names, strict=True))
        return f"y = {self.coef[0]:.4g} + {terms}"

    def summary(self) -> str:
        """Coefficients, fit and leave-one-out statistics as text."""
        t = _core.fusion_t_quantile(0.975, float(self.df))
        lines = [f"{self.form} model, {self.n} plots: {self.equation()}",
                 "  term            coef        se    95 % interval"]
        for name, b, s in zip(["intercept"] + list(self.names), self.coef, self.se, strict=True):
            lines.append(f"  {name:<12} {b:>9.4g} {s:>9.3g}    [{b - t * s:.4g}, {b + t * s:.4g}]")
        lines.append(f"  R² {self.r2:.3f} (adjusted {self.adj_r2:.3f}), residual SE {self.sigma:.4g} "
                     f"on {self.df} df")
        lines.append(f"  leave-one-out: RMSE {self.loo_rmse:.4g} ({self.loo_rrmse:.1f} %), bias "
                     f"{self.loo_bias:+.4g}, R² {self.loo_r2:.3f}")
        return "\n".join(lines)

    def predict(self, predictors, level: float = 0.95) -> dict:
        """Predict at new predictor values.

        Parameters
        ----------
        predictors
            A dict of arrays or of :class:`~sylva.Raster` (one grid) named
            as the model's predictors, or a ``(n, k)`` array in their order.
        level
            Coverage of the prediction interval.

        Returns
        -------
        dict
            ``mean`` (for loglog, ``exp(ŷ + σ²/2)``), ``se`` (standard error
            of a new observation; for loglog that of the log-normal
            distribution), ``lower``, ``upper`` (the t interval, for loglog
            taken back from the log scale) and ``extrapolated`` (a predictor
            outside the plots' range). Rasters when the predictors are
            rasters (``extrapolated`` as 0 and 1, NaN where there is no
            prediction), arrays otherwise. Missing or, for loglog,
            non-positive predictors give NaN.
        """
        if isinstance(predictors, Mapping) or hasattr(predictors, "columns"):
            x, _, grid = _predictor_matrix(predictors, self.names)
        else:
            x, _, grid = _predictor_matrix(predictors, list(self.names))
        d = _core.fusion_predict(self.form, list(map(float, self.coef)), float(self.sigma), int(self.df),
                                 np.ascontiguousarray(self.xtx_inv), list(map(float, self.x_min)),
                                 list(map(float, self.x_max)), x, float(level))
        if grid is None:
            return dict(d)
        shape, xmin, ymin, res, crs = grid
        out = {k: Raster(np.asarray(d[k]).reshape(shape), xmin, ymin, res, crs)
               for k in ("mean", "se", "lower", "upper")}
        ext = np.where(np.isfinite(out["mean"].data), np.asarray(d["extrapolated"], float).reshape(shape), np.nan)
        out["extrapolated"] = Raster(ext, xmin, ymin, res, crs)
        return out


def fit_model(y, predictors, model: str = "loglog", names=None) -> Model:
    """Regress plot values on ALS metrics.

    Ordinary least squares, in one of two forms:

    - ``"loglog"`` (the default): ``ln y = b0 + Σ bk ln xk``, the usual form
      for biomass against canopy height (a power law), with predictions
      taken back as ``exp(ŷ + σ²/2)`` (Baskerville 1972);
    - ``"linear"``: ``y = b0 + Σ bk xk``.

    Leave-one-out predictions come from the hat matrix in closed form (they
    equal refitting without each plot), and with them the RMSE, bias and R²
    a new plot can expect. Keep the predictors few (one or two for ten
    plots): each costs a degree of freedom, and correlated metrics (``zq95``
    and ``zmax``) add little.

    Parameters
    ----------
    y
        Plot values, e.g. the ``agb`` of :func:`plot_values` per plot.
    predictors
        Per plot, the metrics: a dict of arrays, a
        :class:`sylva.als.PlotMetrics`, or an ``(n, k)`` array.
    model : {"loglog", "linear"}
        Model form.
    names
        The predictors to use (keys of ``predictors``), or names for the
        columns of an array; every column if None.

    Returns
    -------
    Model

    Raises
    ------
    ValueError
        With fewer than ``k + 2`` plots for ``k`` coefficients, non-finite
        values, values that are not positive for ``loglog``, or collinear
        predictors.

    Examples
    --------
    >>> m = fusion.fit_model(agb, plot_metrics, "loglog", names=["zq95", "cover"])  # doctest: +SKIP
    >>> print(m.summary())                                                          # doctest: +SKIP
    """
    yv = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    x, nm, grid = _predictor_matrix(predictors, names)
    if grid is not None:
        raise ValueError("plot predictors must be arrays, not rasters")
    if x.shape[0] != len(yv):
        raise ValueError(f"{x.shape[0]} rows of predictors for {len(yv)} plot values")
    d = _core.fusion_fit(yv, x, str(model))
    return Model(d["form"], nm, d["coef"], d["se"], float(d["sigma"]), int(d["df"]), int(d["n"]),
                 float(d["r2"]), float(d["adj_r2"]), d["xtx_inv"], float(d["correction"]), yv, d["fitted"],
                 d["loo"], float(d["loo_rmse"]), float(d["loo_bias"]), float(d["loo_r2"]),
                 float(d["loo_rrmse"]), d["x_min"], d["x_max"])


@dataclass
class Upscaling:
    """A model and its wall-to-wall prediction, from :func:`upscale`.

    Attributes
    ----------
    model
        The fitted :class:`Model`.
    mean, se, lower, upper
        Rasters of the prediction, the standard error of a new observation
        and the interval at ``level``.
    extrapolated
        Raster: 1 where a predictor lies outside the range of the plots, 0
        elsewhere, NaN without a prediction.
    level
        Coverage of the interval.
    """

    model: Model
    mean: Raster
    se: Raster
    lower: Raster
    upper: Raster
    extrapolated: Raster
    level: float = 0.95

    def to_ascii_grids(self, directory: str | Path, prefix: str = "") -> list[str]:
        """Write the rasters as ESRI ASCII grids.

        Parameters
        ----------
        directory
            Output directory (created if needed).
        prefix
            Prepended to ``mean.asc``, ``se.asc``, ``lower.asc``,
            ``upper.asc`` and ``extrapolated.asc``.

        Returns
        -------
        list of str
            The files written.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in ("mean", "se", "lower", "upper", "extrapolated"):
            p = out / f"{prefix}{name}.asc"
            getattr(self, name).to_ascii_grid(p)
            paths.append(str(p))
        return paths


def upscale(y, plot_metrics, grid: Mapping, predictors, model: str = "loglog",
            level: float = 0.95) -> Upscaling:
    """Fit plot values on ALS metrics and predict them over the ALS grid.

    :func:`fit_model` on the plots, then :meth:`Model.predict` on every cell
    of ``grid``. The grid's cells should have the plots' area (e.g. 25 m
    cells for 0.06 ha plots), since metrics such as the height percentiles
    depend on the area they are computed over.

    Parameters
    ----------
    y
        Plot values (e.g. ``agb`` from :func:`plot_values`), one per plot.
    plot_metrics
        The plots' ALS metrics: :class:`sylva.als.PlotMetrics` (from
        :func:`sylva.als.plot_metrics`) or a dict of arrays.
    grid
        The same metrics as rasters on one grid (from
        :func:`sylva.als.grid_metrics`), by name.
    predictors
        Names of the metrics to use.
    model : {"loglog", "linear"}
        Model form.
    level
        Coverage of the prediction interval.

    Returns
    -------
    Upscaling

    Raises
    ------
    ValueError
        As :func:`fit_model`, or for a grid without a predictor.

    Examples
    --------
    >>> plots = als.plot_metrics(cat, centres, radius=15.0, metrics=["zq95", "cover"])  # doctest: +SKIP
    >>> grid = als.grid_metrics(cat, 25.0, ["zq95", "cover"])                           # doctest: +SKIP
    >>> up = fusion.upscale(agb, plots, grid, ["zq95"], "loglog")                       # doctest: +SKIP
    >>> up.mean.to_geotiff("agb.tif")                                                   # doctest: +SKIP
    """
    names = [predictors] if isinstance(predictors, str) else list(predictors)
    missing = [n for n in names if n not in grid]
    if missing:
        raise ValueError(f"the grid has no {missing[0]!r} raster")
    m = fit_model(y, plot_metrics, model, names)
    p = m.predict({n: grid[n] for n in names}, level)
    if not isinstance(p["mean"], Raster):
        raise ValueError("grid must hold Rasters")
    return Upscaling(m, p["mean"], p["se"], p["lower"], p["upper"], p["extrapolated"], float(level))


# ---------------------------------------------------------------- synthetic

def synthetic_scan(scene: PointCloud, origins=(10.0, 10.0, 1.5), resolution_deg: float = 0.25,
                   max_zenith_deg: float = 130.0, target_radius: float = 0.03,
                   terrain_slope: float = 0.05, min_range: float = 0.1, max_range: float = 200.0):
    """Scan a synthetic scene from the ground as :func:`sylva.synthetic.als_flight` flies it.

    Every scene point is a sphere of ``target_radius`` (points of class 2,
    the scene's ground, are ignored), and the ground is the analytic
    terrain of :func:`sylva.synthetic.terrain_height`. Pulses leave the
    scanner on a regular zenith and azimuth grid as thin rays; a ray stops
    at the first sphere or the terrain, whichever it meets first, and gives
    one echo there, or none if it leaves the scene. A layer of spheres of
    ``n`` per m³ is then a turbid medium of plant area density ``2 π r² n``
    (spherical leaf angles) for both scanners, so their estimates can be
    checked against the same foliage. (:func:`sylva.synthetic.scan` instead
    lets the points in each pulse's angular cell be hit, which suits
    geometry but not densities.)

    Parameters
    ----------
    scene
        From :func:`sylva.synthetic.forest` or
        :func:`sylva.synthetic.crown_forest`.
    origins
        One scanner position ``(x, y, z)`` or an ``(n, 3)`` array.
    resolution_deg
        Angular step in zenith and azimuth (degrees).
    max_zenith_deg
        Pulses are fired from straight up to this zenith.
    target_radius
        Sphere radius (m); use the ``target_radius`` of the flight.
    terrain_slope
        Slope of the terrain (``terrain_slope`` of the flight).
    min_range, max_range
        Echoes nearer than ``min_range`` are ignored; rays are followed to
        ``max_range`` (m).

    Returns
    -------
    Shots or list of Shots
        One per origin, every pulse fired (misses included), with the
        echo attributes ``classification`` (2 for the terrain) and
        ``tree_id`` when the scene has it.

    Raises
    ------
    ValueError
        For settings out of range or an origin below the terrain.
    """
    o = np.asarray(origins, dtype=float)
    single = o.ndim == 1
    o = np.ascontiguousarray(o.reshape(-1, 3))
    out = _core.fusion_scan_spheres(np.ascontiguousarray(scene.xyz), scene.attrs, o, float(resolution_deg),
                                    float(max_zenith_deg), float(target_radius), float(terrain_slope),
                                    float(min_range), float(max_range))
    shots = [Shots._from_core(d) for d in out]
    return shots[0] if single else shots


# ------------------------------------------------------------ command line

def _add_commands(sub, fmt: dict) -> None:
    """The ``fusion-register``, ``fusion-trees`` and ``fusion-upscale`` commands."""
    s = sub.add_parser("fusion-register", help="register a TLS plot onto ALS tiles or an ALS cloud", **fmt)
    s.add_argument("tls", help="TLS point cloud with ground classified (class 2)")
    s.add_argument("als", help="ALS directory of tiles, or one file, with ground classified")
    s.add_argument("output", help="JSON file for the registration (transform, residuals, uncertainty)")
    s.add_argument("--initial", help="text file with the 4x4 matrix placing the TLS roughly on the ALS")
    s.add_argument("--transformed", help="also write the TLS cloud moved into the ALS frame here")
    s.add_argument("--search-radius", type=float, default=10.0, help="largest horizontal shift searched (m)")
    s.add_argument("--heading-range", type=float, default=180.0,
                   help="headings searched either side of the initial one (degrees; 180 for all)")
    s.add_argument("--heading-step", type=float, default=3.0, help="coarse heading step (degrees)")
    s.add_argument("--resolution", type=float, default=0.5, help="fine search cell size (m)")
    s.add_argument("--coarse-resolution", type=float, default=2.0, help="coarse search cell size (m)")
    s.add_argument("--dtm-weight", type=float, default=0.5, help="weight of the terrain correlation")
    s.add_argument("--refine", choices=["all", "ground", "none"], default="all",
                   help="points the ICP refinement uses")
    s.add_argument("--no-jackknife", action="store_true", help="skip the quadrant jackknife")
    s.set_defaults(func=_cmd_register)

    s = sub.add_parser("fusion-trees", help="link TLS trees to ALS trees and combine DBH and height", **fmt)
    s.add_argument("tls_trees", help="CSV of TLS trees: x, y and optionally tree_id, dbh, height, volume")
    s.add_argument("als_trees", help="CSV of ALS trees (from als-trees): id, x, y, height")
    s.add_argument("output", help="CSV of the linked TLS trees")
    s.add_argument("--crowns", help="GeoJSON of the ALS crowns (from als-trees --crowns)")
    s.add_argument("--registration", help="JSON from fusion-register, applied to the TLS positions")
    s.add_argument("--als-output", help="also write the ALS trees with their stems as CSV")
    s.add_argument("--max-distance", type=float, default=3.0, help="stem to top distance allowed (m)")
    s.add_argument("--crown-buffer", type=float, default=0.5, help="distance outside a crown still under it (m)")
    s.add_argument("--height-weight", type=float, default=2.0, help="weight of the height difference")
    s.add_argument("--dbh-weight", type=float, default=2.0, help="weight of the DBH shortfall")
    s.add_argument("--max-cost", type=float, default=2.0, help="cost of leaving a stem unmatched")
    s.add_argument("--top-tolerance", type=float, default=1.0, help="height within which a top is reached (m)")
    s.set_defaults(func=_cmd_trees)

    s = sub.add_parser("fusion-upscale", help="regress plot values on ALS metrics and predict them "
                       "wall to wall", **fmt)
    s.add_argument("plots", help="CSV with one row per plot: the value and the ALS metrics (e.g. "
                   "als-plot-metrics output with a column added)")
    s.add_argument("metrics", help="directory of metric rasters <name>.asc (from als-metrics)")
    s.add_argument("output", help="directory for mean.asc, se.asc, lower.asc, upper.asc, extrapolated.asc")
    s.add_argument("--response", required=True, help="column of the plot value (e.g. agb)")
    s.add_argument("--predictors", required=True, help="comma-separated metric names")
    s.add_argument("--model", choices=["loglog", "linear"], default="loglog", help="model form")
    s.add_argument("--level", type=float, default=0.95, help="coverage of the prediction interval")
    s.set_defaults(func=_cmd_upscale)


def _read_csv(path: str) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            out[k] = np.array(vals)
    return out


def _cmd_register(args) -> None:
    from .io import read, write

    tls = read(args.tls)
    init = None if args.initial is None else np.loadtxt(args.initial).reshape(4, 4)
    als = args.als if Path(args.als).is_dir() else read(args.als)
    reg = register(tls, als, init, resolution=args.resolution, coarse_resolution=args.coarse_resolution,
                   search_radius=args.search_radius, heading_range=args.heading_range,
                   heading_step=args.heading_step, dtm_weight=args.dtm_weight, refine=args.refine,
                   jackknife=not args.no_jackknife)
    reg.save(args.output)
    print(reg.report())
    if args.transformed:
        write(reg.apply(tls), args.transformed)


def _cmd_trees(args) -> None:
    tls = _read_csv(args.tls_trees)
    als = _read_csv(args.als_trees)
    if args.crowns:
        gj = json.loads(Path(args.crowns).read_text())
        by_id = {}
        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            if geom.get("type") == "Polygon":
                ring = np.asarray(geom["coordinates"][0], dtype=float)
                by_id[int(feat["properties"]["id"])] = ring[:-1] if len(ring) > 1 and np.allclose(ring[0], ring[-1]) else ring
        ids = als["id"].astype(int) if "id" in als else np.arange(1, len(als["x"]) + 1)
        als["crowns"] = [by_id.get(int(i), np.zeros((0, 2))) for i in ids]
    reg = None if args.registration is None else Registration.load(args.registration)
    links = link_trees(tls, als, reg, max_distance=args.max_distance, crown_buffer=args.crown_buffer,
                       height_weight=args.height_weight, dbh_weight=args.dbh_weight,
                       max_cost=args.max_cost, top_tolerance=args.top_tolerance)
    links.to_csv(args.output)
    if args.als_output:
        t = links.als_table()
        with open(args.als_output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "x", "y", "height", "tls_id", "n_stems", "stems"])
            for k in range(len(t["id"])):
                w.writerow([t["id"][k].item() if hasattr(t["id"][k], "item") else t["id"][k], float(t["x"][k]),
                            float(t["y"][k]), float(t["height"][k]), int(t["tls_id"][k]), int(t["n_stems"][k]),
                            " ".join(str(int(v)) for v in t["stems"][k])])
    print(links)


def _cmd_upscale(args) -> None:
    plots = _read_csv(args.plots)
    if args.response not in plots:
        raise ValueError(f"{args.plots} has no column {args.response!r}")
    names = [n.strip() for n in args.predictors.split(",") if n.strip()]
    grid = {}
    for n in names:
        p = Path(args.metrics) / f"{n}.asc"
        if not p.exists():
            raise ValueError(f"no raster {p}")
        grid[n] = Raster.from_ascii_grid(p)
    up = upscale(plots[args.response], plots, grid, names, args.model, args.level)
    up.to_ascii_grids(args.output)
    print(up.model.summary())
