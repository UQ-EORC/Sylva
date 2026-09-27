"""Write inputs and Python outputs for the R package's leaf and voxel tests.

    python tests/parity/export_r_leaves.py

The R tests (r/sylva/tests/testthat/test-leaves-python.R and
test-voxels-python.R) read the same inputs, call the R API and require the
Python results to 1e-9 (and OBJ and voxel files to match byte for byte).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity import cases_leaves as cl  # noqa: E402
from parity import cases_voxels as cv  # noqa: E402
from parity.export_r_fixtures import OUT, arr, num, save_shots  # noqa: E402
from sylva import leaves, voxels  # noqa: E402
from sylva.qsm import QSM  # noqa: E402
from sylva.raster import Raster  # noqa: E402


def r_value(v):
    """R source for nested dicts of arrays, numbers, strings and None."""
    if isinstance(v, dict):
        return "list(" + ", ".join(f"`{k}` = {r_value(x)}" for k, x in v.items()) + ")"
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, np.ndarray):
        return arr(v)
    return num(v)


def save(d, name, a):
    a = np.asarray(a, dtype=float)
    np.savetxt(d / f"{name}.csv", a.reshape(len(a), -1), delimiter=",", fmt="%.17g")


def lad(d):
    return {k: getattr(d, k) for k in ("bin_centres", "density", "mean", "std", "beta_a", "beta_b", "chi",
                                        "de_wit", "mean_deg")}


def grid(g):
    z, a = g.profile()
    c, ca = g.cells()
    return {"origin": np.asarray(g.origin, float), "voxel_size": g.voxel_size, "density": g.density,
            "total_area": g.total_area, "profile_z": z, "profile_area": a, "cells": c, "cell_area": ca}


def shape(s):
    return {"vertices": s.vertices, "faces": s.faces.astype(float), "length": s.length, "width": s.width,
            "area": s.area}


def mesh(d, name, m):
    save(d, f"{name}_vertices", m.vertices)
    save(d, f"{name}_faces", m.faces)
    save(d, f"{name}_centres", m.centres)
    save(d, f"{name}_normals", m.normals)
    return {"inclination": m.inclination, "cylinder": m.cylinder.astype(float), "leaf_area": m.leaf_area,
            "n": float(len(m)), "total_area": m.total_area}


def leaves_fixtures():
    d = OUT / "leaves"
    d.mkdir(parents=True, exist_ok=True)
    exp = {}
    tree = cl._tree(1)
    save(d, "tree_xyz", tree)
    save(d, "gbs", leaves.classify_leaf_wood(tree))
    save(d, "gbs_intervals", leaves.classify_leaf_wood(tree, intervals=[0.2, 0.4, 0.8], max_angle=0.6))
    save(d, "passage", leaves.classify_leaf_wood(tree, method="passage", threshold=0.9, voxel_size=0.03))
    pts = cl._leaf_points(3)
    save(d, "leaf_xyz", pts)
    exp["lad_res0"] = lad(leaves.leaf_angle_distribution(pts))
    exp["lad_none"] = lad(leaves.leaf_angle_distribution(pts, k=8, res=None, n_bins=9))
    incl = np.random.default_rng(5).uniform(0, np.pi / 2, 500)
    save(d, "incl", incl)
    exp["lad_incl"] = lad(leaves.leaf_angle_distribution(incl, inclinations=True, weights=incl ** 2, n_bins=12))
    for name in ("spherical", "planophile", "extremophile"):
        exp[f"type_{name}"] = lad(leaves.LeafAngleDistribution.from_type(name, n_bins=15))
    exp["g_planophile"] = leaves.LeafAngleDistribution.from_type("planophile").g([0.0, 0.3, 0.9, 1.4])
    dense = cl._leaf_points(6, n_leaves=200, spread=1.2)
    save(d, "dense_xyz", dense)
    g = leaves.leaf_area_density(dense)
    exp["grid"] = grid(g)
    exp["grid_fine"] = grid(leaves.leaf_area_density(dense, voxel_size=0.1, res=0.015, k=9))
    exp["grid_scaled"] = grid(g.scaled_to(7.5))
    rng = np.random.default_rng(7)
    dn = rng.uniform(0, 2, (3, 4, 5))
    dn[0, 1, 2] = np.nan
    dn[1, 3, 4] = -1.0
    exp["nan_density"] = dn
    ng = leaves.LeafAreaGrid(np.array([-1.0, 2.0, 0.5]), 0.4, dn)
    exp["nan"] = grid(ng)
    exp["nan"]["area"] = ng.area
    exp["nan_scaled"] = grid(ng.scaled_to(4.0))
    # Shapes.
    v = np.array([[0, 0, 0], [0.1, 0.03, 0.01], [0.2, 0, 0], [0.1, -0.03, 0.01]], float) + [0.3, -0.2, 1.0]
    f = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    exp["mesh_v"] = v
    exp["shape_default"] = shape(leaves.LeafShape())
    exp["shape_mesh"] = shape(leaves.LeafShape.from_mesh(v, f))
    exp["shape_mesh_raw"] = shape(leaves.LeafShape.from_mesh(v, f, length=0.3, normalise=False))
    exp["shape_mesh_2d"] = shape(leaves.LeafShape.from_mesh(v[:, :2], f, width=0.05))
    exp["shape_scaled"] = shape(leaves.LeafShape(length=0.1, width=0.03).scaled_to(0.004))
    exp["shape_resized"] = shape(leaves.LeafShape().resized(0.15))
    (d / "leaf.obj").write_text("# a leaf\no leaf\n" + "".join(f"v {x} {y} {z}\n" for x, y, z in v)
                                + "vt 0 0\n\nf 1/1 2/1 3/1 4/1\nf -4 -2 -1\n")
    exp["shape_obj"] = shape(leaves.LeafShape.from_obj(d / "leaf.obj", width=0.05))
    exp["single_default"] = leaves.single_leaf_area(0.06, 0.03)
    exp["single_custom"] = leaves.single_leaf_area(0.06, 0.03, leaves.LeafShape.from_mesh(v, f))
    # Leaves on a model, and the OBJ files.
    model = cl._model()
    exp["cylinders"] = model.cylinders
    seeds = cl._leaf_points(9, n_leaves=100, centre=(0.6, 0.0, 3.8), spread=0.5)
    save(d, "seeds", seeds)
    angles = leaves.leaf_angle_distribution(seeds)
    sg = leaves.leaf_area_density(seeds, voxel_size=0.25)
    exp["mesh_grid"] = mesh(d, "mesh_grid", leaves.add_leaves(model, sg, angles, leaf_points=seeds, leaf_length=0.06,
                                                              leaf_width=0.03, max_branch_distance=1.5))
    exp["mesh_total"] = mesh(d, "mesh_total", leaves.add_leaves(model, 1.5, "planophile", leaf_points=seeds, seed=4))
    custom = leaves.LeafShape.from_mesh(v, f)
    m = leaves.add_leaves(None, sg.scaled_to(3.0), angles, leaf_points=seeds, shape=custom, leaf_width=0.05, seed=11)
    exp["mesh_shape"] = mesh(d, "mesh_shape", m)
    few = leaves.add_leaves(model, 0.05, "spherical", leaf_points=seeds, seed=2)
    exp["mesh_few"] = mesh(d, "mesh_few", few)
    with tempfile.TemporaryDirectory() as t:
        few.to_obj(Path(t) / "l.obj")
        shutil.copy(Path(t) / "l.obj", d / "few_leaves.obj")
        leaves.write_tree_obj(Path(t) / "t.obj", model, few, sides=8, contiguous=True)
        shutil.copy(Path(t) / "t.obj", d / "few_tree.obj")
    (d / "expected.R").write_text("expected <- " + r_value(exp) + "\n")
    print(f"wrote {d}")


def voxels_fixtures():
    d = OUT / "voxels"
    d.mkdir(parents=True, exist_ok=True)
    s = cv._canopy(1, n=3000)
    save_shots(d, "canopy", s)
    save(d, "canopy_attrs", np.column_stack([s.echo_attrs[k] for k in ("classification", "tree_id", "intensity")]))
    rng = np.random.default_rng(2)
    foliage = rng.integers(0, 4, s.n_echoes).astype(np.uint8)
    save(d, "foliage", foliage)
    dtm = Raster(np.full((4, 4), 0.0) + np.arange(4)[:, None] * 0.01, -0.5, -0.5, 1.0)
    exp = {"dtm_data": dtm.data}
    grids = {
        "classes": voxels.ray_voxelize(s, 0.5, cv.BOUNDS, ground_class=2, leaf_classes=[4], wood_classes=[6],
                                       occlusion=True, attenuation=["fpl", "transmittance"], laser="VZ-400"),
        "dtm": voxels.ray_voxelize(s, 0.5, cv.BOUNDS, dtm=dtm, ground_distance=0.15, wood_classes=[6], occlusion=True,
                                   attenuation="transmittance", beam=(0.005, 0.0003)),
        "arrays": voxels.ray_voxelize(s, 0.75, None, ground=s.echo_attrs["classification"] == 2, foliage=foliage,
                                      weighting="relative", tree_attr=None),
        "iad": voxels.ray_voxelize(s, 0.5, cv.BOUNDS, ground_class=2, weighting="strongest", inclination=True,
                                   leaf_classes=[4], wood_classes=[6]),
    }
    for name, g in grids.items():
        e = {"origin": np.asarray(g.origin), "voxel_size": g.voxel_size, "shape": np.array(g.shape, float),
             "fields": ",".join(g.fields), "z_levels": g.z_levels()}
        for k in ("num_hits", "num_beams", "path_length", "state", "pad_fpl", "transmittance", "distance_from_ground"):
            if k in g.fields or k in g.metrics:
                e[k] = np.asarray(g[k], float)
        e["X"], e["Y"], e["Z"] = g.centers()
        e["profile_pad"] = g.profile()
        e["profile_fpl"] = g.profile("free_path_length", min_beams=3)
        occ = g.occlusion_profile(0.0, 6.5)
        e["occ"] = {k: (np.asarray(v, float) if k != "total" else v) for k, v in occ.items()}
        e["occ_default"] = {k: (np.asarray(v, float) if k != "total" else v) for k, v in g.occlusion_profile(0.5).items()}
        e["map"] = g.observed_map(0.0, 6.5)
        exp[name] = e
    iad = grids["iad"].tree_iad
    exp["tree_iad"] = {str(t): {k: (np.asarray(v, float) if isinstance(v, np.ndarray) else v) for k, v in x.items()}
                       for t, x in iad.items()}
    lg = leaves.LeafAreaGrid.from_voxels(grids["classes"], "lad_fpl")
    exp["lad_grid"] = {"origin": lg.origin, "density": lg.density}
    pts = np.vstack([np.c_[rng.uniform(0.2, 1.3, (80, 2)), rng.uniform(0.5, 2.5, 80)],
                     np.c_[rng.uniform(1.7, 2.8, (80, 2)), rng.uniform(0.5, 3.5, 80)]])
    lab = np.r_[np.zeros(80, int), np.full(80, 3)]
    save(d, "sample_xyz", pts)
    save(d, "sample_labels", lab)
    exp["sampling"] = {k: np.asarray(v, float) for k, v in
                       voxels.tree_sampling(grids["classes"], pts, lab, min_beams=3.0, above=1.0).items()}
    g = grids["iad"]
    cyl = np.array([[1.0, 1.0, 0.1, 0, 0, 1, 2.8, 0.25, -1, 0, 0, 0]], float)
    exp["wood_cyl"] = cyl
    g.add_wood_volume(QSM(cyl))
    exp["wood_volume"] = np.asarray(g.wood_volume, float)
    exp["wood_density"] = g.wood_volume_density
    with tempfile.TemporaryDirectory() as t:
        exp["n_vox"] = float(g.write(Path(t) / "g.vox"))
        shutil.copy(Path(t) / "g.vox", d / "iad.vox")
        exp["n_txt"] = float(g.write(Path(t) / "g.txt", filled_only=True))
        shutil.copy(Path(t) / "g.txt", d / "iad.txt")
        g.write_iad_csv(Path(t) / "iad.csv")
        shutil.copy(Path(t) / "iad.csv", d / "iad.csv")
    p = d / "plot.shots"
    cv._canopy(3, n=1500).save(p)
    fg = voxels.ray_voxelize(p, 0.5, cv.BOUNDS, ground_class=2, leaf_classes=[4], wood_classes=[6], occlusion=True)
    exp["file"] = {"num_hits": np.asarray(fg.num_hits, float), "pad_fpl": fg.pad_fpl,
                   "occ": {k: (np.asarray(v, float) if k != "total" else v) for k, v in fg.occlusion_profile().items()}}
    exp["laser"] = np.array(voxels.laser_spec("LMS-Q780"))
    theta = np.linspace(0, np.pi / 2, 11)
    exp["g_ellipsoidal"] = voxels.leaf_projection(theta, "ellipsoidal", [1.7])
    exp["g_erectophile"] = voxels.leaf_projection(theta, "erectophile")
    (d / "expected.R").write_text("expected <- " + r_value(exp) + "\n")
    print(f"wrote {d}")


if __name__ == "__main__":
    leaves_fixtures()
    voxels_fixtures()
