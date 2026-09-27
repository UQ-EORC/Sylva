"""Parity cases for sylva.synthetic.

The generators are the inputs here: they are recorded with small sizes so
that the port must reproduce NumPy's random streams (``default_rng``'s
uniform and ziggurat normal draws) exactly. ``scan`` is also run on a cloud
drawn from a seeded NumPy generator.
"""

import numpy as np

from sylva import PointCloud, synthetic


def _cloud(prefix, c):
    out = {f"{prefix}xyz": c.xyz}
    for k, v in c.attrs.items():
        out[f"{prefix}{k}"] = v
    return out


def _shots(prefix, s):
    out = {f"{prefix}{k}": getattr(s, k) for k in ("direction", "echo_start", "echo_count", "echo_range")}
    out[f"{prefix}origin"] = np.unique(s.origin, axis=0)
    for k, v in s.echo_attrs.items():
        out[f"{prefix}attr_{k}"] = v
    return out


def terrain():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-30, 30, (2, 500))
    return {"default": synthetic.terrain_height(x, y), "slope": synthetic.terrain_height(x, y, slope=-0.3),
            "scalar": synthetic.terrain_height(2.5, 7.0)}


def tree():
    out = {}
    out.update(_cloud("a/", synthetic.tree(leaf_points=600, height=2.0, seed=0)))
    out.update(_cloud("b/", synthetic.tree(1.0, -2.0, dbh=0.5, height=2.5, z0=0.3, n_branches=4, leaf_points=500, seed=7)))
    out.update(_cloud("bare/", synthetic.tree(height=1.5, n_branches=2, leaf_points=0, seed=3)))
    c = synthetic.tree(height=1.5, n_branches=3, leaf_points=900, seed=5)
    out.update(_cloud("c/", c))
    out["c/leaf_area"] = synthetic.leaf_area(c)
    return out


def forest():
    trees = [(3.0, 3.0, 0.2, 2.5)]
    f = synthetic.forest(trees, size=10.0, ground_points=1500, margin=2.0, seed=4)
    out = _cloud("f/", f)
    out["f/leaf_area"] = synthetic.leaf_area(f)
    out.update(_cloud("ground/", synthetic.forest([], size=5.0, ground_points=200, seed=2)))
    return out


def scan():
    trees = [(3.0, 3.0, 0.2, 2.5)]
    f = synthetic.forest(trees, size=10.0, ground_points=1500, margin=2.0, seed=4)
    out = _shots("forest/", synthetic.scan(f, origin=(5.0, 5.0, 1.5), resolution_deg=2.0))
    out.update(_shots("fine/", synthetic.scan(f, origin=(4.0, 6.0, 1.2), resolution_deg=2.5, max_zenith_deg=95.0,
                                               max_echoes=3, echo_separation=0.2)))
    rng = np.random.default_rng(1)
    xyz = rng.normal(0, 6, (4000, 3))
    cloud = PointCloud(xyz, {"label": rng.integers(0, 5, 4000), "w": rng.uniform(size=4000).astype(np.float32)})
    out.update(_shots("cloud/", synthetic.scan(cloud, origin=(0.5, -0.5, 0.0), resolution_deg=3.0, max_zenith_deg=180.0,
                                               max_echoes=4, echo_separation=0.0)))
    return out


CASES = {f.__name__: f for f in (terrain, tree, forest, scan)}
