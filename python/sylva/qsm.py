"""Quantitative structure models: skeletonisation and cylinder fitting.

A QSM is a set of connected cylinders. :func:`build_qsm` bins geodesic
distance from the base over a kNN graph, splits bins into connected segments,
fits a RANSAC cylinder to each and links parents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = ["QSM", "fit_cylinder", "fit_cylinder_ransac", "skeletonize", "build_qsm", "wood_points",
           "write_obj", "write_ply_mesh"]

COLUMNS = ("sx", "sy", "sz", "ax", "ay", "az", "length", "radius", "parent", "branch_order",
           "branch_id", "n_points")


@dataclass
class QSM:
    """Cylinder model. ``cylinders`` is ``(n, 12)`` with columns :data:`COLUMNS`."""

    cylinders: np.ndarray

    def __post_init__(self) -> None:
        self.cylinders = np.ascontiguousarray(self.cylinders, dtype=np.float64).reshape(-1, 12)

    def __len__(self) -> int:
        return len(self.cylinders)

    def column(self, name: str) -> np.ndarray:
        return self.cylinders[:, COLUMNS.index(name)]

    @property
    def start(self) -> np.ndarray:
        return self.cylinders[:, 0:3]

    @property
    def axis(self) -> np.ndarray:
        return self.cylinders[:, 3:6]

    @property
    def end(self) -> np.ndarray:
        return self.start + self.axis * self.column("length")[:, None]

    @property
    def volumes(self) -> np.ndarray:
        return np.pi * self.column("radius") ** 2 * self.column("length")

    @property
    def total_volume(self) -> float:
        return float(self.volumes.sum())

    @property
    def stem_volume(self) -> float:
        return float(self.volumes[self.column("branch_order") == 0].sum())

    @property
    def branch_volume(self) -> float:
        return self.total_volume - self.stem_volume

    @property
    def total_length(self) -> float:
        return float(self.column("length").sum())

    @property
    def max_branch_order(self) -> int:
        return int(self.column("branch_order").max()) if len(self) else 0

    @property
    def dbh(self) -> float:
        return _core.qsm_summary(self.cylinders)["dbh"]

    def summary(self) -> dict:
        s = _core.qsm_summary(self.cylinders)
        return {
            "n_cylinders": len(self),
            "total_volume_m3": s["total_volume"],
            "stem_volume_m3": s["stem_volume"],
            "branch_volume_m3": s["branch_volume"],
            "total_length_m": s["total_length"],
            "max_branch_order": s["max_branch_order"],
            "dbh_m": s["dbh"],
        }

    def to_csv(self, path: str | Path) -> None:
        _core.qsm_write_csv(self.cylinders, str(path))

    def to_treefile(self, path: str | Path) -> None:
        """Write in the raycloudtools/treetools ``_trees.txt`` format."""
        _core.qsm_write_treefile(self.cylinders, str(path))

    @classmethod
    def from_csv(cls, path: str | Path) -> QSM:
        return cls(np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2))

    def mesh(self, sides: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Triangle mesh of the cylinders: ``(vertices (n,3), faces (m,3), cylinder index per face)``."""
        return _core.qsm_mesh(self.cylinders, sides)

    def to_obj(self, path: str | Path, sides: int = 12) -> None:
        """Write the cylinder mesh as a Wavefront OBJ (Blender, MeshLab, CloudCompare)."""
        v, f, _ = self.mesh(sides)
        write_obj(path, [(v, f)])

    def to_ply(self, path: str | Path, sides: int = 12, color=None) -> None:
        """Write the cylinder mesh as a binary PLY; ``color`` is an optional RGB triple,
        by default faces are coloured by branch order."""
        v, f, owner = self.mesh(sides)
        if color is None:
            order = self.column("branch_order").astype(int)
            face_rgb = _ORDER_COLORS[np.minimum(order[owner], len(_ORDER_COLORS) - 1)]
        else:
            face_rgb = np.tile(np.asarray(color, dtype=np.uint8), (len(f), 1))
        write_ply_mesh(path, v, f, face_rgb)


_ORDER_COLORS = np.array([[139, 90, 43], [205, 133, 63], [222, 184, 135], [60, 179, 113],
                          [46, 139, 87], [34, 139, 34]], dtype=np.uint8)


def write_obj(path: str | Path, meshes: list[tuple[np.ndarray, np.ndarray]],
              names: list[str] | None = None) -> None:
    """Write one or more ``(vertices, faces)`` meshes to a single OBJ, one object each."""
    with open(path, "w") as fh:
        fh.write("# sylva QSM mesh\n")
        offset = 1
        for i, (v, f) in enumerate(meshes):
            fh.write(f"o {names[i] if names else f'tree_{i + 1}'}\n")
            np.savetxt(fh, v, fmt="v %.4f %.4f %.4f")
            np.savetxt(fh, np.asarray(f, dtype=np.int64) + offset, fmt="f %d %d %d")
            offset += len(v)


def write_ply_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray,
                   face_rgb: np.ndarray | None = None) -> None:
    """Write a binary little-endian PLY triangle mesh with optional per-face colours."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    header = ["ply", "format binary_little_endian 1.0", "comment sylva QSM mesh",
              f"element vertex {len(vertices)}", "property float x", "property float y",
              "property float z", f"element face {len(faces)}",
              "property list uchar int vertex_indices"]
    if face_rgb is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(vertices.tobytes())
        if face_rgb is None:
            rec = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
            arr = np.empty(len(faces), dtype=rec)
        else:
            rec = np.dtype([("n", "u1"), ("i", "<i4", (3,)), ("rgb", "u1", (3,))])
            arr = np.empty(len(faces), dtype=rec)
            arr["rgb"] = np.asarray(face_rgb, dtype=np.uint8)
        arr["n"] = 3
        arr["i"] = faces
        fh.write(arr.tobytes())


def fit_cylinder(xyz: np.ndarray, axis_init=None) -> dict:
    """Least-squares cylinder fit: ``{"point", "axis", "radius", "rmse"}``."""
    return _core.fit_cylinder(np.ascontiguousarray(xyz, dtype=float),
                              None if axis_init is None else tuple(float(v) for v in axis_init))


def fit_cylinder_ransac(xyz: np.ndarray, threshold: float = 0.02, iterations: int = 100,
                        sample_size: int = 12, seed: int = 0) -> dict:
    """RANSAC cylinder fit; adds an ``"inliers"`` mask to :func:`fit_cylinder`'s result."""
    return _core.fit_cylinder_ransac(np.ascontiguousarray(xyz, dtype=float), threshold,
                                     iterations, sample_size, seed)


def skeletonize(cloud: PointCloud, base_xy=None, k: int = 15, max_edge: float = 1.0,
                bin_length: float = 0.1) -> dict:
    """Graph skeleton of one tree: ``segment_id`` per point, ``geodesic``
    distance from the base, segment ``centres`` and ``(child, parent)`` ``edges``."""
    return _core.skeletonize(cloud.xyz, None if base_xy is None else tuple(base_xy), k,
                             max_edge, bin_length)


def build_qsm(cloud: PointCloud, base_xy=None, k: int = 15, max_edge: float = 1.0,
              bin_length: float = 0.1, min_points: int = 1, ransac_threshold: float = 0.02,
              max_radius: float = 1.0, taper_limit: float = 1.1, max_rmse: float = 0.03,
              smooth_steps: int = 10, apex_radius: float = 0.0025, min_arc_deg: float = 90.0,
              min_inlier_fraction: float = 0.05, prune_points: int = 5, fit_min_points: int = 50,
              crop_length: float = 0.0, butt_height: float = 0.6,
              relative_tolerance: float = 0.08, base_radius: float = 0.0,
              allometry_tolerance: float = 0.3, buttress_equivalent_area: bool = True,
              buttress_max_inlier_fraction: float = 0.3, pipe_slack: float = 1.2,
              branch_min_inlier_fraction: float = 0.3, cluster_eps: float = 0.1,
              centre_fit_points: int = 100, radius_smooth_steps: int = 15,
              butt_swell: float = 1.1, butt_vertical_run: int = 4,
              butt_max_lean_deg: float = 50.0, chain_max_d: float = 0.1,
              fourier_min_radius: float = 0.15) -> QSM:
    """Skeletonise then fit cylinders for a single segmented tree.

    Skeleton nodes (geodesic bins of ``bin_length``) are Taubin-smoothed and
    a RANSAC circle is fitted to each node's points in the plane
    perpendicular to the skeleton; accepted fits need a contiguous arc of
    ``min_arc_deg`` and ``min_inlier_fraction`` inliers. Radii are then
    regularised along every root-to-tip path: an allometric prior anchored
    on ``base_radius`` (pass the measured DBH / 2) replaces weak fits more
    than ``allometry_tolerance`` away, chains are made non-increasing, gaps
    are interpolated and unmeasured branches take the prior. Leafy tips
    shorter than ``crop_length`` are not reconstructed. Run
    :func:`wood_points` first on leafy trees.
    """
    d = _core.build_qsm(cloud.xyz, None if base_xy is None else tuple(base_xy), k, max_edge,
                        bin_length, min_points, ransac_threshold, max_radius, taper_limit,
                        max_rmse, smooth_steps, apex_radius, min_arc_deg, min_inlier_fraction,
                        prune_points, fit_min_points, crop_length, butt_height,
                        relative_tolerance, base_radius, allometry_tolerance,
                        buttress_equivalent_area, buttress_max_inlier_fraction, pipe_slack,
                        branch_min_inlier_fraction, cluster_eps, centre_fit_points,
                        radius_smooth_steps, butt_swell, butt_vertical_run, butt_max_lean_deg, chain_max_d, fourier_min_radius)
    return QSM(d["cylinders"])


def wood_points(cloud: PointCloud, k: int = 20, threshold: float = 0.85,
                voxel_size: float | None = 0.02, medium_threshold: float = 0.75,
                scale_radius: float = 0.0, passage: bool = True, min_passage: int = 3,
                target_res: float = 0.2,
                graph_k: int = 10, max_edge: float = 1.0, base_height: float = 0.25,
                assign_dist: float = 0.05, assign_scale: float = 0.0,
                component_res: float = 0.05,
                component_min: int = 200, sor_k: int = 50, sor_std: float = 1.0,
                dilate_dist: float = 0.03, method: str = "passage") -> PointCloud:
    """Leaf / wood separation for one tree's points, returning the wood
    thinned to ``voxel_size``.

    Two cues are combined. Local anisotropy -- planarity + linearity over
    ``k`` neighbours (and, with ``scale_radius`` > 0, again over that wider
    neighbourhood, the lower score counting: better leaf labels, but it
    removes wood a QSM needs, see :func:`sylva.leaves.classify_leaf_wood`)
    -- marks bark and branch surfaces
    above ``threshold`` (high likelihood) and ``medium_threshold`` (kept
    only after statistical outlier removal, ``sor_k`` / ``sor_std``, and
    next to wood already found; set it at or above ``threshold`` to skip
    the step, which on small leafy crowns pulls foliage in around the
    stem). Topology recovers what anisotropy
    misses: shortest paths from the base over a kNN graph are traced to one
    target per ``target_res`` cell, and any point that at least
    ``min_passage`` of those paths run through is wood, with its neighbours
    within ``assign_dist`` -- a roughly or thinly scanned trunk is neither
    planar nor linear locally but every path to the crown crosses it. A
    path follows one side of a stem, so this keeps the points on and beside
    the paths rather than the whole section; ``assign_scale`` (try 0.03)
    widens the reach with the share of the tree a point carries, up to that
    fraction of the tree height on the trunk. High-likelihood points survive in connected
    components (``component_res``) that touch passage wood or hold
    ``component_min`` points; the result is dilated by ``dilate_dist`` for
    thin branches. ``passage=False`` gives the anisotropy-only filter.
    """
    from .filters import voxel_downsample

    thin = voxel_downsample(cloud, voxel_size) if voxel_size else cloud
    if method == "gbs":
        from .leaves import classify_leaf_wood

        return thin[classify_leaf_wood(thin, voxel_size=0.0, method="gbs")]
    mask = _core.wood_mask(thin.xyz, k, threshold, medium_threshold, scale_radius, graph_k, max_edge,
                           base_height, target_res, min_passage, assign_dist, assign_scale,
                           component_res,
                           component_min, sor_k, sor_std, dilate_dist, passage)
    return thin[mask]
