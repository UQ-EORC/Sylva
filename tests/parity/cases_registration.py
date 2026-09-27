"""Parity cases for sylva.registration."""

import numpy as np

from sylva import PointCloud, registration


def scene(seed=11, n=4000):
    """A ground plane, two walls and a few cylinders: enough structure for ICP."""
    rng = np.random.default_rng(seed)
    g = np.column_stack([rng.uniform(-8, 8, (n, 2)), rng.normal(0.0, 0.005, n)])
    w1 = np.column_stack([rng.uniform(-8, 8, n // 2), np.full(n // 2, 8.0), rng.uniform(0, 4, n // 2)])
    w2 = np.column_stack([np.full(n // 2, -8.0), rng.uniform(-8, 8, n // 2), rng.uniform(0, 4, n // 2)])
    stems = []
    for cx, cy, r in [(2.0, 3.0, 0.3), (-4.0, -1.0, 0.2), (5.0, -5.0, 0.4)]:
        t = rng.uniform(0, 2 * np.pi, 800)
        stems.append(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t), rng.uniform(0, 5, 800)]))
    return np.vstack([g, w1, w2, *stems])


def transforms():
    return {"rz": registration.rotation_z(33.0), "rz_neg": registration.rotation_z(-181.5),
            "t": registration.translation(1.5, -2.0, 0.25)}


def kabsch():
    rng = np.random.default_rng(12)
    src = rng.uniform(-5, 5, (40, 3))
    m = registration.translation(0.4, -1.2, 0.3) @ registration.rotation_z(12.0)
    dst = src @ m[:3, :3].T + m[:3, 3] + rng.normal(0.0, 0.002, src.shape)
    return {"kabsch": registration.kabsch(src, dst), "kabsch_list": registration.kabsch(src.tolist(), dst.tolist())}


def icp():
    xyz = scene()
    target = PointCloud(xyz)
    m = registration.translation(0.15, -0.1, 0.05) @ registration.rotation_z(2.0)
    source = PointCloud(xyz[::2]).transform(np.linalg.inv(m))
    out = {}
    for method in ("point", "plane"):
        t, info = registration.icp(source, target, method=method, max_correspondence_distance=0.6,
                                   max_iterations=40)
        out[f"{method}_t"] = t
        out[f"{method}_rmse"] = info["rmse"]
        out[f"{method}_iter"] = info["iterations"]
        out[f"{method}_n"] = info["n_correspondences"]
    t, info = registration.icp(source, target, init=registration.translation(0.1, -0.05, 0.0),
                               method="plane", trim=0.8, normal_k=8, tolerance=1e-8)
    out["trim_t"] = t
    out["trim_rmse"] = info["rmse"]
    out["trim_n"] = info["n_correspondences"]
    return out


def merge():
    rng = np.random.default_rng(13)
    a = PointCloud(rng.uniform(0, 1, (50, 3)), {"i": np.arange(50, dtype=np.uint16),
                                                  "only_a": np.ones(50)})
    b = PointCloud(rng.uniform(0, 1, (30, 3)), {"i": np.arange(30, dtype=np.uint16)})
    tr = [registration.rotation_z(10.0), registration.translation(5.0, 0.0, 1.0)]
    m1 = registration.merge_scans([a, b], tr)
    m2 = registration.merge_scans([a, b], scan_ids=False)
    return {"m1_xyz": m1.xyz, "m1_scan": m1.attrs["scan_id"], "m1_i": m1.attrs["i"],
            "m1_names": np.array(sorted(m1.attrs)), "m2_xyz": m2.xyz, "m2_names": np.array(sorted(m2.attrs)),
            "normals": np.abs(registration.estimate_normals(PointCloud(scene()[:1500]), k=9))}


CASES = {"transforms": transforms, "kabsch": kabsch, "icp": icp, "merge": merge}
