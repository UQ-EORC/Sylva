# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Registration of a terrestrial cloud to an airborne survey."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ._common import _ground, _matrix, _transform_xyz


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
    from ..als import Catalog

    if isinstance(als, PointCloud):
        return als
    if isinstance(als, (str, Path, list, tuple)) and not isinstance(als, Catalog):
        from ..als import catalog
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
