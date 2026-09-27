"""Parity cases for sylva.leaves (leaf/wood labels, angle distributions, leaf
area grids, leaf shapes, leaf insertion and the OBJ reader and writers)."""

import tempfile
from pathlib import Path

import numpy as np

from sylva import PointCloud, leaves, voxels
from sylva.qsm import QSM
from sylva.shots import Shots


def _leaf_points(seed, n_leaves=120, mean_deg=35.0, centre=(0.0, 0.0, 5.0), spread=0.8, size=0.06, per=60):
    """Square leaves sampled as points, with normals around ``mean_deg``."""
    rng = np.random.default_rng(seed)
    pts = []
    for theta in np.radians(np.clip(rng.normal(mean_deg, 12.0, n_leaves), 0, 90)):
        phi = rng.uniform(0, 2 * np.pi)
        n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        u = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        c = np.asarray(centre) + rng.uniform(-spread, spread, 3)
        ab = rng.uniform(-size / 2, size / 2, (per, 2))
        pts.append(c + ab[:, :1] * u + ab[:, 1:] * v + rng.normal(0, 0.0005, (per, 3)))
    return np.vstack(pts)


def _tree(seed, height=6.0):
    """A leaning stem with one branch and a blob of leaves."""
    rng = np.random.default_rng(seed)
    parts = []
    for base, axis, length, r, n in [((0, 0, 0), (0.05, 0, 1), height, 0.12, 5000),
                                     ((0.1, 0, 0.5 * height), (1, 0, 0.6), 1.5, 0.05, 1200)]:
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        u = np.cross(a, [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        t = rng.uniform(0, length, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        rr = r + rng.normal(0, 0.002, n)
        parts.append(np.asarray(base) + t[:, None] * a + rr[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v))
    parts.append(_leaf_points(seed + 1, n_leaves=80, centre=(1.2, 0.0, 0.5 * height + 0.8), spread=0.4))
    return np.vstack(parts)


def _model():
    cyl = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.12, -1, 0, 0, 100],
                    [0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 2.5, 0.09, 0, 0, 0, 80],
                    [0.0, 0.0, 3.0, 0.8, 0.0, 0.6, 1.5, 0.05, 0, 1, 1, 40]], float)
    cyl[2, 3:6] /= np.linalg.norm(cyl[2, 3:6])
    return QSM(cyl)


def classify_leaf_wood():
    short = PointCloud(_tree(1))
    tall = _tree(2, height=16.0)
    out = {"gbs": leaves.classify_leaf_wood(short),
           "gbs_tall": leaves.classify_leaf_wood(tall),
           "gbs_intervals": leaves.classify_leaf_wood(short, intervals=[0.2, 0.4, 0.8]),
           "gbs_angle": leaves.classify_leaf_wood(tall, max_angle=0.3),
           "gbs_empty": leaves.classify_leaf_wood(np.zeros((0, 3))),
           "passage": leaves.classify_leaf_wood(short, method="passage"),
           "passage_threshold": leaves.classify_leaf_wood(short, method="passage", threshold=0.9, voxel_size=0.03)}
    try:
        leaves.classify_leaf_wood(short, method="nope")
    except ValueError as e:
        out["bad_method"] = str(e)
    return out


def _lad(prefix, d):
    out = {f"{prefix}{k}": getattr(d, k) for k in ("bin_centres", "density", "mean", "std", "beta_a", "beta_b", "chi")}
    out[f"{prefix}de_wit"] = str(d.de_wit)
    out[f"{prefix}mean_deg"] = d.mean_deg
    out[f"{prefix}g"] = d.g([0.0, 0.3, 0.9, 1.4])
    return out


def angle_distributions():
    out = {}
    for name in ("spherical", "uniform", "planophile", "erectophile", "plagiophile", "extremophile"):
        out.update(_lad(f"{name}_", leaves.LeafAngleDistribution.from_type(name)))
    out.update(_lad("sph7_", leaves.LeafAngleDistribution.from_type("spherical", n_bins=7)))
    try:
        leaves.LeafAngleDistribution.from_type("round")
    except KeyError as e:
        out["bad_type"] = str(e)
    pts = _leaf_points(3)
    out.update(_lad("res0_", leaves.leaf_angle_distribution(pts)))
    out.update(_lad("res_", leaves.leaf_angle_distribution(PointCloud(pts), k=8, n_bins=9, res=0.02)))
    out.update(_lad("none_", leaves.leaf_angle_distribution(pts, res=None)))
    w = np.random.default_rng(4).uniform(0, 2, len(pts))
    out.update(_lad("none_w_", leaves.leaf_angle_distribution(pts, res=None, weights=w)))
    incl = np.random.default_rng(5).uniform(0, np.pi / 2, 500)
    out.update(_lad("incl_", leaves.leaf_angle_distribution(incl, inclinations=True, n_bins=12)))
    out.update(_lad("incl_w_", leaves.leaf_angle_distribution(incl, inclinations=True, weights=incl ** 2)))
    return out


def _grid(prefix, g):
    z, a = g.profile()
    c, ca = g.cells()
    return {f"{prefix}origin": g.origin, f"{prefix}voxel_size": g.voxel_size, f"{prefix}density": g.density,
            f"{prefix}area": g.area, f"{prefix}total_area": g.total_area, f"{prefix}profile_z": z,
            f"{prefix}profile_area": a, f"{prefix}cells": c, f"{prefix}cell_area": ca}


def leaf_area_grids():
    pts = _leaf_points(6, n_leaves=200, spread=1.2)
    out = _grid("default_", leaves.leaf_area_density(pts))
    out.update(_grid("fine_", leaves.leaf_area_density(PointCloud(pts), voxel_size=0.1, res=0.015, k=9)))
    out.update(_grid("scaled_", leaves.leaf_area_density(pts, voxel_size=0.3).scaled_to(7.5)))
    out.update(_grid("empty_", leaves.leaf_area_density(np.zeros((0, 3)))))
    out.update(_grid("empty_scaled_", leaves.leaf_area_density(np.zeros((0, 3))).scaled_to(3.0)))
    d = np.random.default_rng(7).uniform(0, 2, (3, 4, 5))
    d[0, 1, 2] = np.nan
    d[2, 0, 0] = 0.0
    d[1, 3, 4] = -1.0
    g = leaves.LeafAreaGrid(np.array([-1.0, 2.0, 0.5]), 0.4, d)
    out.update(_grid("nan_", g))
    out.update(_grid("nan_scaled_", g.scaled_to(4.0)))
    return out


def from_voxels():
    rng = np.random.default_rng(8)
    n = 5000
    xy = rng.uniform(0, 2, (n, 2))
    free = rng.exponential(1 / 0.8, n)
    ranges = np.where(free < 1.0, 8.0 + free, np.nan)
    hit = np.isfinite(ranges)
    count = hit.astype(np.int64)
    shots = Shots(np.column_stack([xy, np.full(n, 10.0)]), np.tile([0.0, 0.0, -1.0], (n, 1)),
                  np.r_[0, np.cumsum(count)[:-1]], count, ranges[hit])
    shots.echo_attrs["classification"] = rng.choice([4, 6], hit.sum()).astype(np.uint8)
    grid = voxels.ray_voxelize(shots, 0.5, ((0, 0, 0), (2, 2, 3)), leaf_classes=[4], wood_classes=[6])
    out = _grid("pad_", leaves.LeafAreaGrid.from_voxels(grid))
    out.update(_grid("lad_", leaves.LeafAreaGrid.from_voxels(grid, "lad_fpl")))
    return out


def _shape(prefix, s):
    return {f"{prefix}vertices": s.vertices, f"{prefix}faces": s.faces, f"{prefix}length": s.length,
            f"{prefix}width": s.width, f"{prefix}area": s.area}


def _error(fn):
    try:
        fn()
    except (ValueError, KeyError) as e:
        return f"{type(e).__name__}: {e}"
    return "no error"


def leaf_shapes():
    out = _shape("default_", leaves.LeafShape())
    out.update(_shape("unit_", leaves.LeafShape(*leaves._unit_blade())))
    out.update(_shape("resized_", leaves.LeafShape().resized(0.15)))
    out.update(_shape("resized2_", leaves.LeafShape().resized(width=0.02)))
    out.update(_shape("scaled_", leaves.LeafShape(length=0.1, width=0.03).scaled_to(0.004)))
    v = np.array([[0, 0, 0], [0.1, 0.03, 0.01], [0.2, 0, 0], [0.1, -0.03, 0.01]], float) + [0.3, -0.2, 1.0]
    f = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    out.update(_shape("mesh_", leaves.LeafShape.from_mesh(v, f)))
    out.update(_shape("mesh_sized_", leaves.LeafShape.from_mesh(v, f, length=0.05)))
    out.update(_shape("mesh_raw_", leaves.LeafShape.from_mesh(v, f, normalise=False)))
    out.update(_shape("mesh_raw_sized_", leaves.LeafShape.from_mesh(v, f, length=0.3, width=0.1, normalise=False)))
    out.update(_shape("mesh_2d_", leaves.LeafShape.from_mesh(v[:, :2], f.astype(np.int64))))
    out["single_default"] = leaves.single_leaf_area(0.06, 0.03)
    out["single_custom"] = leaves.single_leaf_area(0.06, 0.03, leaves.LeafShape.from_mesh(v, f))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "leaf.obj"
        p.write_text("# a leaf\no leaf\n" + "".join(f"v {x} {y} {z}\n" for x, y, z in v) + "vt 0 0\n\nf 1/1 2/1 3/1 4/1\n"
                     "f -4 -2 -1\n")
        out.update(_shape("obj_", leaves.LeafShape.from_obj(p)))
        out.update(_shape("obj_sized_", leaves.LeafShape.from_obj(str(p), length=0.2, width=0.05)))
        out.update(_shape("obj_raw_", leaves.LeafShape.from_obj(p, normalise=False)))
        empty = Path(d) / "empty.obj"
        empty.write_text("v 0 0 0\nv 1 0 0\n")
        out["obj_empty"] = _error(lambda: leaves.LeafShape.from_obj(empty)).replace(d, "DIR")
    out["bad_shape_v"] = _error(lambda: leaves.LeafShape(vertices=v[:, :2], faces=f))
    out["bad_faces"] = _error(lambda: leaves.LeafShape(faces=np.zeros((0, 3), np.uint32)))
    out["bad_index"] = _error(lambda: leaves.LeafShape(v, np.array([[0, 1, 4]])))
    out["bad_length"] = _error(lambda: leaves.LeafShape(length=0.0))
    out["bad_area"] = _error(lambda: leaves.LeafShape().scaled_to(0.0))
    out["bad_mesh"] = _error(lambda: leaves.LeafShape.from_mesh(np.zeros((0, 3)), f))
    out["flat_mesh"] = _error(lambda: leaves.LeafShape.from_mesh(np.c_[np.zeros(4), v[:, 1:]], f))
    return out


def default_leaf():
    out = _shape("initial_", leaves.default_leaf())
    was = leaves.set_default_leaf(length=0.12)
    try:
        out.update(_shape("set_", leaves.default_leaf()))
        out.update(_shape("was_", was))
        leaves.set_default_leaf(leaves.LeafShape(length=0.2, width=0.1), width=0.05)
        out.update(_shape("shape_", leaves.default_leaf()))
    finally:
        leaves.set_default_leaf(was)
    out.update(_shape("restored_", leaves.default_leaf()))
    return out


def _mesh(prefix, m):
    return {f"{prefix}vertices": m.vertices, f"{prefix}faces": m.faces, f"{prefix}centres": m.centres,
            f"{prefix}normals": m.normals, f"{prefix}inclination": m.inclination, f"{prefix}cylinder": m.cylinder,
            f"{prefix}leaf_area": m.leaf_area, f"{prefix}n": len(m), f"{prefix}total_area": m.total_area}


def add_leaves():
    pts = _leaf_points(9, n_leaves=100, centre=(0.6, 0.0, 3.8), spread=0.5)
    model = _model()
    angles = leaves.leaf_angle_distribution(pts)
    grid = leaves.leaf_area_density(pts, voxel_size=0.25)
    out = _mesh("grid_", leaves.add_leaves(model, grid, angles, leaf_points=pts, leaf_length=0.06, leaf_width=0.03,
                                           max_branch_distance=1.5))
    out.update(_mesh("total_", leaves.add_leaves(model, 1.5, "planophile", leaf_points=PointCloud(pts), seed=4)))
    out.update(_mesh("nowood_", leaves.add_leaves(None, grid, "spherical", jitter=0.0)))
    v = np.array([[0, 0, 0], [0.1, 0.03, 0.01], [0.2, 0, 0], [0.1, -0.03, 0.01]], float)
    shape = leaves.LeafShape.from_mesh(v, np.array([[0, 1, 2], [0, 2, 3]]))
    out.update(_mesh("shape_", leaves.add_leaves(model, grid.scaled_to(3.0), angles, leaf_points=pts, shape=shape,
                                                 leaf_width=0.05, seed=11)))
    out["no_points"] = _error(lambda: leaves.add_leaves(model, 2.0))
    return out


def obj_writers():
    pts = _leaf_points(10, n_leaves=30, centre=(0.3, 0.0, 3.5), spread=0.3)
    model = _model()
    mesh = leaves.add_leaves(model, leaves.leaf_area_density(pts), "spherical", leaf_points=pts)
    out = {}
    with tempfile.TemporaryDirectory() as d:
        mesh.to_obj(Path(d) / "leaves.obj")
        out["leaves_obj"] = (Path(d) / "leaves.obj").read_text()
        leaves.write_tree_obj(Path(d) / "tree.obj", model, mesh)
        out["tree_obj"] = (Path(d) / "tree.obj").read_text()
        leaves.write_tree_obj(str(Path(d) / "tree8.obj"), model, mesh, sides=8, contiguous=True)
        out["tree8_obj"] = (Path(d) / "tree8.obj").read_text()
        # Values that round at the fourth decimal, negative zero and ties.
        odd = leaves.LeafMesh(np.array([[0.00005, -0.00004, 1.23455], [-0.0, 2.5e-5, 12345.678949999],
                                        [0.03125, -0.03125, 1e9]]), np.array([[0, 1, 2]]), np.zeros((1, 3)),
                              np.zeros((1, 3)), np.zeros(1), np.zeros(1, np.int64), 0.1)
        odd.to_obj(Path(d) / "odd.obj")
        out["odd_obj"] = (Path(d) / "odd.obj").read_text()
        empty = leaves.LeafMesh(np.zeros((0, 3)), np.zeros((0, 3), np.uint32), np.zeros((0, 3)), np.zeros((0, 3)),
                                np.zeros(0), np.zeros(0, np.int64), 0.1)
        empty.to_obj(Path(d) / "empty.obj")
        out["empty_obj"] = (Path(d) / "empty.obj").read_text()
    return out


CASES = {"classify_leaf_wood": classify_leaf_wood, "angle_distributions": angle_distributions,
         "leaf_area_grids": leaf_area_grids, "from_voxels": from_voxels, "leaf_shapes": leaf_shapes,
         "default_leaf": default_leaf, "add_leaves": add_leaves, "obj_writers": obj_writers}
