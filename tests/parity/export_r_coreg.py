"""Write inputs and Python outputs for the R coregistration tests.

    python tests/parity/export_r_coreg.py

r/sylva/tests/testthat/test-coreg-python.R rebuilds the same inputs from
these files, calls the R API and requires the Python results to 1e-9.
Indices are written as Python gives them (from 0).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity import cases_coreg_refine as cr  # noqa: E402
from parity.export_r_fixtures import OUT, num, r_value  # noqa: E402
from sylva import coreg  # noqa: E402
from sylva.coreg import posegraph as pg  # noqa: E402
from sylva.coreg import transforms as tf  # noqa: E402

D = OUT / "coreg"


def value(v):
    """``r_value`` with strings, lists and integers."""
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "list(" + ", ".join(value(x) for x in v) + ")"
    if isinstance(v, dict):
        return "list(" + ", ".join(f"{k} = {value(x)}" for k, x in v.items()) + ")"
    if isinstance(v, np.ndarray) and v.dtype.kind in "US":
        return "c(" + ", ".join(json.dumps(str(x)) for x in v) + ")"
    if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
        return num(v)
    return r_value(v)


def csv(name, a):
    np.savetxt(D / name, np.asarray(a, dtype=float).reshape(len(a), -1), delimiter=",", fmt="%.17g")


def match(m):
    return {"transform": m.transform, "n_inliers": m.n_inliers, "rmse": m.rmse,
            "correspondences": np.asarray(m.correspondences, float).reshape(-1, 2), "success": m.success}


def transforms(rng):
    xi = np.vstack([rng.normal(0, 1, (4, 6)), [[0, 0, np.pi - 1e-7, 1, 2, 3]], [[1e-10, 0, 0, 1, 0, 0]]])
    src = rng.normal(0, 10, (30, 3))
    T = tf.se3_exp(rng.normal(0, 1, 6))
    dst = tf.transform_points(T, src) + rng.normal(0, 0.01, (30, 3))
    w = rng.uniform(0.1, 2.0, 30)
    return {
        "xi": xi, "se3_exp": np.array([tf.se3_exp(x) for x in xi]), "se3_log": np.array([tf.se3_log(tf.se3_exp(x)) for x in xi]),
        "so3_log": np.array([tf.so3_log(tf.so3_exp(x[:3])) for x in xi]),
        "inverse": np.array([tf.invert(tf.se3_exp(x)) for x in xi]),
        "src": src, "dst": dst, "w": w, "T": T,
        "kabsch": tf.kabsch(src, dst), "kabsch_w": tf.kabsch(src, dst, w),
        "yaw": tf.kabsch_2d_yaw(src, dst), "yaw_w": tf.kabsch_2d_yaw(src, dst, w),
        "moved": tf.transform_points(T, src), "rotated": tf.transform_vectors(T, src),
        "difference": np.array(tf.transform_difference(T, tf.se3_exp(xi[0]))),
        "angle": tf.rotation_angle(T), "yaw_transform": tf.yaw_transform(0.7, 1.0, -2.0, 0.5),
        "skew": tf.skew([1.0, 2.0, 3.0]),
    }


def reflectors(rng):
    out = {}
    tpl = D / "scan.tpl"
    tpl.write_text(json.dumps([
        {"name": "TP00", "reflectance": 27.3, "diameter": 0.057, "pointcount": 415,
         "positionCartesian": {"x": 4.52, "y": -1.08, "z": -1.28}},
        {"name": "broken"},
        {"name": 7, "pointcount": 12.9, "positionCartesian": {"x": "1.5", "y": " -2 ", "z": 3}},
    ]))
    rfl = D / "ScanPos002.rfl"
    rfl.write_text(
        "RieglRflID=1.1\n"
        "ReflectorIdx=name,index,status,x,y,z,r,theta,phi,reflectance,diameter,points,linkname\n"
        "Reflector0=ScanPos002/a.rxp,0,5,18.406487,1.751866,-1.807602,18.58,95.58,5.44,14.69,0.0997,19060,$Nolink\n"
        "Reflector1=ScanPos002/b.rxp,1,5,23.672012,2.619638,-2.476889,23.94,95.94,6.31,18.98,0.0840,14397,$Nolink\n"
        "Reflector2=broken\n"
    )
    for key, found in (("tpl", coreg.read_tiepoint_list(tpl)), ("rfl", coreg.read_reflector_list(rfl))):
        out[key] = {"xyz": np.array([r.position for r in found]), "reflectance": np.array([r.reflectance for r in found]),
                    "n_points": np.array([r.n_points for r in found], float),
                    "name": np.array([r.name for r in found], dtype="U64")}
    scene = rng.uniform(-10, 10, (3000, 3))
    refl = rng.normal(-10, 3, 3000)
    blobs = [c + rng.normal(0, 0.03, (30, 3)) for c in rng.uniform(-8, 8, (4, 3))]
    xyz = np.vstack([scene, *blobs])
    refl = np.concatenate([refl, rng.uniform(6, 20, 120)])
    csv("reflectance_cloud.csv", np.column_stack([xyz, refl]))
    found = coreg.detect_reflectors(xyz, refl)
    out["detected"] = {"xyz": np.array([r.position for r in found]), "reflectance": np.array([r.reflectance for r in found]),
                       "diameter": np.array([r.diameter for r in found]), "n_points": np.array([r.n_points for r in found], float)}
    positions = rng.uniform(-20, 20, (7, 3))
    T = tf.se3_exp(np.array([0.05, -0.03, 1.2, 4.0, -3.0, 0.5]))
    moved = tf.transform_points(tf.invert(T), positions) + rng.normal(0, 0.002, (7, 3))
    out["source"], out["target"] = moved[:6], positions[1:]
    targets = [coreg.Reflector(*p) for p in out["source"]], [coreg.Reflector(*p) for p in out["target"]]
    out["match"] = match(coreg.match_reflectors(*targets))
    return out


def posegraph(rng):
    n = 6
    poses = [np.eye(4)] + [tf.se3_exp(np.r_[rng.normal(0, 0.05, 3), rng.normal(0, 8, 3)]) for _ in range(n - 1)]
    graph = pg.PoseGraph(n, reference=0, fixed={4: poses[4]})
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            noise = tf.se3_exp(np.r_[rng.normal(0, 0.002, 3), rng.normal(0, 0.01, 3)])
            Z = noise @ tf.invert(poses[j]) @ poses[i]
            fitness, count = float(rng.uniform(0.3, 0.9)), int(rng.integers(100, 2000))
            graph.add_edge(i, j, Z, fitness=fitness, rmse=0.01, n_correspondences=count)
            edges.append({"i": i, "j": j, "transform": Z, "fitness": fitness, "n": count})
    edges[3]["transform"] = graph.edges[3].transform = (
        tf.se3_exp(np.array([0.0, 0.0, 0.8, 5.0, -4.0, 0.0])) @ graph.edges[3].transform)
    H = (lambda a: a.T @ a)(rng.normal(size=(300, 6)))
    out = {"edges": edges, "fixed_pose": poses[4], "H": H, "pose_2": poses[2], "pose_3": poses[3],
           "default_information": pg.default_information(0.02, 0.6, 1500),
           "plane_information": pg.plane_edge_information(H, 0.003, 300, poses[2]),
           "adjoint": pg.adjoint(poses[3]), "components": [np.array(c, float) for c in graph.components()]}
    out["initialise"] = np.array(graph.initialise())
    out["total_initial"] = graph.total_error()
    out["residual_2"] = graph.residual(graph.edges[2])
    r = graph.optimise()
    out["optimise"] = {"poses": np.array(r.poses), "iterations": r.iterations, "converged": r.converged,
                       "initial_error": r.initial_error, "final_error": r.final_error,
                       "rejected": np.array(r.rejected_edges, float), "edge_errors": r.edge_errors}
    out["relative"] = graph.relative(1, 3)
    return out


def survey():
    truth, start, points, stems = cr._survey(40, n_ground=9000, n_stem=500, n_log=500)
    for k, p in enumerate(points):
        csv(f"scan{k}.csv", p)
    out = {"start": np.array(start), "stems": stems}
    messages = []
    r = coreg.refine_joint(points, start, [(0, 1), (1, 2), (0, 2)], stems, 1, points_per_scan=2500,
                           correspondences_per_pair=900, voxel_sizes=(0.2, 0.1), max_distances=(0.4, 0.2),
                           seed=3, log=messages.append)
    out["refine"] = {"poses": np.array(r.poses), "shifts": r.shifts, "rotations": r.rotations,
                     "residual_before": r.residual_before, "residual_after": r.residual_after,
                     "correspondences": r.correspondences, "log": np.array(messages, dtype="U200")}
    g = coreg.fit_ground(points[0], cell_size=0.5)
    out["ground"] = {"elevation": g.elevation, "origin": g.origin, "observed": g.observed.astype(float),
                     "slope": g.slope_deg}
    q = np.array([[0.3, 0.2], [-3.0, 4.0], [50.0, 50.0], [2.25, -1.75]])
    out["ground_query"] = q
    out["height_at"] = g.height_at(q)
    out["support"] = g.support(q).astype(float)
    out["normalised"] = g.normalise(points[0][:50])
    m = coreg.detect_stems(points[0], g, name="scan0")
    out["detected"] = {"xyz": m.positions, "dbh": m.diameters, "quality": m.qualities, "axes": m.axes}
    planar = [coreg.planar_filter(p) for p in points[:2]]
    out["planar"] = {"n": np.array([len(p) for p in planar], float), "head": [p[:20] for p in planar]}
    cfg = coreg.ICPConfig(voxel_sizes=(0.2, 0.1), max_distances=(0.5, 0.25), max_points=3000)
    initial = tf.invert(start[1]) @ start[0]
    res = coreg.icp(planar[0], planar[1], initial, cfg)
    out["icp"] = {"transform": res.transform, "fitness": res.fitness, "inlier_rmse": res.inlier_rmse,
                  "n_correspondences": res.n_correspondences, "iterations": res.iterations,
                  "converged": res.converged, "history": np.array(res.history),
                  "hessian": res.information.hessian, "sigma": res.information.sigma, "n": res.information.n,
                  "prepared": coreg.icp(planar[0], coreg.ICPTarget(planar[1], cfg), initial, cfg).transform}
    info = coreg.plane_information(planar[0], planar[1], initial, cfg)
    out["plane_information"] = {"hessian": info.hessian, "sigma": info.sigma, "n": info.n}
    out["evaluate"] = np.array(coreg.evaluate_registration(points[0], points[1], initial, voxel=0.1, max_points=2000))
    nrm, pl = coreg.estimate_normals(points[0][:400], k=12, radius=0.5)
    vox, counts = coreg.voxel_downsample(points[0], 0.3, return_counts=True)
    tree = coreg.KdTree(points[1][:, :2])
    d, i = tree.query(points[0][:40, :2], distance_upper_bound=0.1)
    out["geometry"] = {"normals": nrm, "planarity": pl, "voxel": vox, "counts": counts.astype(float),
                       "distance": d, "index": i.astype(float)}
    return out


def stems(rng):
    xy = rng.uniform(-15, 15, (12, 2))
    dbh = rng.uniform(0.1, 0.6, 12)
    a = coreg.stem_map_from_arrays(xy, dbh, rng.normal(0, 0.2, 12), name="a")
    for k, s in enumerate(a.stems):
        s.rmse, s.coverage, s.n_slices = 0.0017 * (k + 1), 0.31 + 0.037 * k, 3 + k % 5
        s.axis = np.array([0.01 * k, -0.02, 1.0]) / np.linalg.norm([0.01 * k, -0.02, 1.0])
        s.lean_deg = 0.5 * k
    assert len(set(a.qualities)) == len(a), "ties would be ordered by numpy's unstable sort"
    a.save(D / "stems_python.json")
    T = tf.yaw_transform(0.8, 3.0, -2.0, 0.1)
    b = coreg.stem_map_from_arrays(tf.transform_points(T, a.positions)[:, :2] + rng.normal(0, 0.02, (12, 2)),
                                   dbh + rng.normal(0, 0.005, 12), name="b")
    result = coreg.match_stem_maps(a, b)
    return {
        "xy": xy, "dbh": dbh, "quality": a.qualities, "sorted_x": a.sorted_by_quality().positions[:, 0],
        "top5_x": a.top(5).positions[:, 0], "moved": a.transformed(T).positions, "moved_axes": a.transformed(T).axes,
        "b_xy": b.xy, "b_dbh": b.diameters,
        "match": {"transform": result.transform, "n_inliers": result.n_inliers, "inlier_rmse": result.inlier_rmse,
                  "score": result.score, "correspondences": result.correspondences.astype(float),
                  "success": result.success, "ambiguity": result.ambiguity},
    }


def main():
    if D.exists():
        shutil.rmtree(D)
    D.mkdir(parents=True)
    rng = np.random.default_rng(2026)
    expected = {"transforms": transforms(rng), "reflectors": reflectors(rng), "posegraph": posegraph(rng),
                "survey": survey(), "stems": stems(rng)}
    (D / "expected.R").write_text("expected <- " + value(expected) + "\n")
    print(f"wrote {D}")


if __name__ == "__main__":
    main()
