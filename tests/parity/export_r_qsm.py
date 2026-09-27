"""Write inputs and Python outputs for the R package's QSM tests.

    python tests/parity/export_r_qsm.py

The R test (r/sylva/tests/testthat/test-qsm-python.R) reads the same
inputs, calls the R API and requires the Python results to 1e-9, and files
written by R to match the Python ones byte for byte (by MD5).
"""

import hashlib
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity import cases_qsm as cq  # noqa: E402
from parity.export_r_fixtures import OUT  # noqa: E402
from parity.export_r_leaves import r_value  # noqa: E402
from sylva import PointCloud, qsm, trees  # noqa: E402


def save(d, name, a):
    """``name.csv``, one row per row of ``a``; an empty file for no rows."""
    a = np.asarray(a, dtype=float)
    if not len(a):
        (d / f"{name}.csv").write_text("")
        return
    np.savetxt(d / f"{name}.csv", a.reshape(len(a), -1), delimiter=",", fmt="%.17g")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def model_values(m):
    return {"end": m.end, "volumes": m.volumes, "total_volume": m.total_volume, "stem_volume": m.stem_volume,
            "branch_volume": m.branch_volume, "total_length": m.total_length,
            "max_branch_order": float(m.max_branch_order), "dbh": m.dbh}


def metrics(m):
    out = {}
    for k, v in m.metrics().items():
        if isinstance(v, dict):
            out.update({f"crown_{c}": x for c, x in v.items()})
        else:
            out[k] = np.asarray(v, dtype=float)
    return out


def plot_cloud(seed):
    """Two branchy trees, a flanged base, a clump and some noise."""
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for tid, (x, y, r, h) in enumerate([(0.0, 0.0, 0.15, 4.0), (5.0, 1.0, 0.1, 3.0)], start=1):
        p = cq._branchy(rng, x, y, r, h, density=1000)
        parts.append(p)
        labels.append(np.full(len(p), tid))
    flanged = cq._stem(rng, 9.0, -3.0, 0.25, 2.5, density=4000, flanges=True)
    parts.append(flanged)
    labels.append(np.full(len(flanged), 4))
    clump = cq._stem(rng, 14.0, 0.0, 0.05, 0.4, density=300)
    parts.append(clump)
    labels.append(np.full(len(clump), 6))
    noise = rng.uniform(-2, 16, (200, 3)) * [1, 1, 0.3]
    parts.append(noise)
    labels.append(np.full(len(noise), -1))
    return np.vstack(parts), np.concatenate(labels)


def qsm_fixtures():
    d = OUT / "qsm"
    d.mkdir(parents=True, exist_ok=True)
    exp = {}
    # A model's values, cuts and metrics.
    rows = cq._valid_rows(1, 30)
    save(d, "rows", rows)
    m = qsm.QSM(rows)
    exp["model"] = model_values(m)
    zs = [-1.0, 0.8, 1.7, 2.9, 100.0]
    exp["zs"] = np.array(zs)
    exp["volume_above"] = np.array([m.volume_above(z) for z in zs])
    for k, z in enumerate(zs):
        save(d, f"above_{k}", m.above(z).cylinders)
    exp["metrics"] = metrics(m)
    exp["branches"] = {k: np.asarray(v, dtype=float) for k, v in m.branches().items()}
    exp["summary"] = {k: float(v) for k, v in m.summary().items()}
    for name, kw in [("mesh", {}), ("mesh_cont", {"sides": 8, "contiguous": True})]:
        v, f, o = m.mesh(**kw)
        save(d, f"{name}_vertices", v)
        save(d, f"{name}_faces", f)
        exp[f"{name}_owner"] = o.astype(float)
    # Files, compared byte for byte.
    files = {}
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        m.to_csv(t / "m.csv")
        (d / "model.csv").write_bytes((t / "m.csv").read_bytes())
        files["csv"] = md5(t / "m.csv")
        m.to_treefile(t / "m.txt")
        files["treefile"] = md5(t / "m.txt")
        m.to_obj(t / "m.obj", sides=8)
        files["obj"] = md5(t / "m.obj")
        m.to_ply(t / "m.ply", contiguous=True)
        files["ply"] = md5(t / "m.ply")
        m.to_ply(t / "c.ply", color=(10, 200, 30))
        files["ply_color"] = md5(t / "c.ply")
        v, f, _ = m.mesh(sides=6)
        qsm.write_obj(t / "w.obj", [(v, f), (v[:3], [[0, 1, 2]])], names=["a", "b"])
        files["write_obj"] = md5(t / "w.obj")
        qsm.write_ply_mesh(t / "w.ply", v, f, (5, 6, 7))
        files["write_ply"] = md5(t / "w.ply")
    exp["from_csv"] = qsm.QSM.from_csv(d / "model.csv").cylinders
    # Fits, skeleton and model on a stem with two branches.
    rng = np.random.default_rng(12)
    section = cq._stem(rng, 1.0, -2.0, 0.2, 0.5, density=600, noise=0.004)
    section = np.vstack([section, rng.uniform(0.5, 1.5, (60, 3)) + [0, -2.5, 0]])
    save(d, "section", section)
    f = qsm.fit_cylinder(section[:300])
    exp["fit"] = np.r_[f["point"], f["axis"], f["radius"], f["rmse"]]
    f = qsm.fit_cylinder(section[:300], axis_init=(0.1, 0.0, 1.0))
    exp["fit_axis"] = np.r_[f["point"], f["axis"], f["radius"], f["rmse"]]
    f = qsm.fit_cylinder_ransac(section, threshold=0.01, iterations=50, seed=4)
    exp["ransac"] = np.r_[f["point"], f["axis"], f["radius"], f["rmse"]]
    exp["ransac_inliers"] = f["inliers"].astype(float)
    tree = cq._branchy(np.random.default_rng(13), 0.0, 0.0, 0.12, 3.0, density=1500)
    save(d, "tree", tree)
    cloud = PointCloud(tree)
    s = qsm.skeletonize(cloud, bin_length=0.2)
    exp["skeleton"] = {"segment_id": s["segment_id"].astype(float), "geodesic": s["geodesic"],
                       "centres": s["centres"], "edges": s["edges"].astype(float)}
    save(d, "build_qsm", qsm.build_qsm(cloud).cylinders)
    save(d, "build_qsm_set", qsm.build_qsm(cloud, base_xy=(0.01, 0.0), bin_length=0.15, radius_power=0.25,
                                            buttress_equivalent_area=False).cylinders)
    save(d, "wood", qsm.wood_points(cloud).xyz)
    save(d, "wood_set", qsm.wood_points(cloud, voxel_size=0.03, threshold=0.8, passage=False).xyz)
    # A buttress, joined to a model.
    base = cq._stem(np.random.default_rng(14), 0.0, 0.0, 0.25, 2.5, density=3500, flanges=True)
    save(d, "base", base)
    bc = PointCloud(base, {"height": base[:, 2].copy()})
    b = qsm.buttress_mesh(bc, (0.0, 0.0), ground_z=0.0, top=1.2, resolution=0.04)
    auto = qsm.buttress_mesh(bc, (0.01, 0.02), resolution=0.05, slice_height=0.1)
    for name, x in [("buttress", b), ("buttress_auto", auto)]:
        save(d, f"{name}_vertices", x.vertices)
        save(d, f"{name}_faces", x.faces)
        exp[name] = {"volume": x.volume, "top": x.top, "top_z": x.top_z, "heights": x.heights, "areas": x.areas,
                     "solidities": x.solidities, "open": x.open.astype(float)}
    stem = qsm.QSM(np.array([[0, 0, 0, 0, 0, 1, 2.0, 0.24, -1, 0, 0, 10],
                             [0, 0, 2.0, 0.6, 0, 0.8, 1.0, 0.08, 0, 1, 1, 10],
                             [0, 0, 2.0, 0, 0, 1, 1.0, 0.2, 0, 0, 0, 10]], float))
    save(d, "stem", stem.cylinders)
    exp["buttress_total_volume"] = b.total_volume(stem)
    for name, kw in [("fused", {}), ("fused_flat", {"sides": 8, "contiguous": False, "overlap": 0.0})]:
        t = b.fuse(stem, **kw)
        save(d, f"{name}_vertices", t.vertices[len(b.vertices):])     # after the buttress's own
        save(d, f"{name}_faces", t.faces[len(b.faces):])
        exp[name] = {"part": t.part.astype(float), "buttress_volume": t.buttress_volume,
                     "wood_volume": t.wood_volume, "volume": t.volume, "top_z": t.top_z, "offset": t.offset,
                     "overhang": t.overhang}
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        b.to_obj(t / "b.obj")
        files["buttress_obj"] = md5(t / "b.obj")
        b.to_ply(t / "b.ply")
        files["buttress_ply"] = md5(t / "b.ply")
        fused = b.fuse(stem)
        fused.to_obj(t / "f.obj")
        files["fused_obj"] = md5(t / "f.obj")
        fused.to_ply(t / "f.ply")
        files["fused_ply"] = md5(t / "f.ply")
        fused.to_ply(t / "g.ply", color=(1, 2, 3))
        files["fused_ply_color"] = md5(t / "g.ply")
    # A plot.
    xyz, labels = plot_cloud(15)
    save(d, "plot_xyz", xyz)
    save(d, "plot_labels", labels)
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    stems = [trees.Tree(4, 9.0, -3.0, 0.5)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plots = {"plain": qsm.build_plot(cloud, labels, wood=False, min_points=1000),
                 "full": qsm.build_plot(cloud, labels, stems, wood=True, buttress=True, min_points=1000,
                                        bin_length=0.15)}
    for name, p in plots.items():
        e = {"ids": np.array(sorted(p.models), dtype=float), "total_volume": p.total_volume,
             "skipped_ids": np.array(list(p.skipped), dtype=float), "skipped": {str(k): v for k, v in p.skipped.items()},
             "buttress_ids": np.array(sorted(p.buttresses), dtype=float),
             "volumes": np.array([p.volume(t) for t in sorted(p.models)])}
        rows = p.table()
        for k in rows[0]:
            e[f"table_{k}"] = np.array([np.nan if r[k] == "" else float(r[k]) for r in rows])
        for t, mm in p.models.items():
            save(d, f"{name}_model_{t}", mm.cylinders)
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            p.to_csv(t / "trees.csv")
            files[f"{name}_csv"] = md5(t / "trees.csv")
            for fmt in ("ply", "obj"):
                for f in p.write_meshes(t / fmt, fmt=fmt, sides=8):
                    files[f"{name}_{fmt}_{f.name}"] = md5(f)
            p.write_cylinders(t / "cyl", prefix="c")
            for f in sorted((t / "cyl").iterdir()):
                files[f"{name}_cyl_{f.name}"] = md5(f)
        exp[f"plot_{name}"] = e
    exp["files"] = files
    (d / "expected.R").write_text("expected <- " + r_value(exp) + "\n")
    print(f"wrote {d}")


if __name__ == "__main__":
    qsm_fixtures()
