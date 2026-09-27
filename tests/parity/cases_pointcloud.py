"""Parity cases for sylva.pointcloud."""

import numpy as np

from sylva import PointCloud, registration


def _cloud(seed=21, n=200):
    rng = np.random.default_rng(seed)
    return PointCloud(rng.uniform(-3, 3, (n, 3)), {"height": rng.uniform(0, 20, n).astype(np.float32),
                                                  "k": rng.integers(0, 5, n).astype(np.int64)})


def methods():
    c = _cloud()
    m = registration.translation(1.0, 2.0, -3.0) @ registration.rotation_z(71.0)
    m[2, 0] = 0.01  # not rigid: the full 3 x 3 block is applied
    t = c.transform(m)
    lo, hi = c.bounds
    sub = c[c.z > 0.5]
    return {"transform": t.xyz, "transform_k": t.attrs["k"], "heights": c.heights(),
            "heights_z": c.heights("absent"), "lo": lo, "hi": hi, "sub_xyz": sub.xyz, "sub_k": sub.attrs["k"],
            "idx_xyz": c[np.array([5, 3, 3, 0])].xyz, "slice_xyz": c[10:20:3].xyz,
            "with": c.with_attrs(k=np.zeros(len(c)), new=np.arange(len(c))).attrs["new"],
            "without": np.array(sorted(c.without("k", "none").attrs))}


def concat():
    a = _cloud(22, 40)
    b = _cloud(23, 25).with_attrs(extra=np.ones(25))
    c = PointCloud(np.zeros((3, 3)), {"k": np.array([1.5, 2.5, 3.5]), "height": np.zeros(3, np.float32)})
    ab = PointCloud.concatenate([a, b])
    abc = PointCloud.concatenate([a, b, c])
    return {"ab_xyz": ab.xyz, "ab_k": ab.attrs["k"], "ab_names": np.array(list(ab.attrs)),
            "abc_k": abc.attrs["k"], "abc_height": abc.attrs["height"]}


def from_array():
    rng = np.random.default_rng(24)
    a = rng.uniform(0, 1, (30, 6))
    c1 = PointCloud.from_array(a)
    c2 = PointCloud.from_array(a, {4: "intensity"})
    return {"c1_xyz": c1.xyz, "c1_names": np.array(sorted(c1.attrs)), "c1_col5": c1.attrs["col5"],
            "c2_names": np.array(sorted(c2.attrs)), "c2_intensity": c2.attrs["intensity"]}


CASES = {"methods": methods, "concat": concat, "from_array": from_array}
