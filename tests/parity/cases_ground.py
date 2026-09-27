"""Parity cases for sylva.ground."""

import numpy as np

from sylva import PointCloud, Raster, ground


def plot(seed=5, n_ground=2500, n_veg=1500):
    """Sloping, wavy ground under a scatter of vegetation up to 15 m."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 12.0, (n_ground, 2))
    z = 0.08 * xy[:, 0] + 0.2 * np.sin(xy[:, 1] / 2.0) + rng.normal(0.0, 0.01, n_ground)
    g = np.column_stack([xy, z])
    vxy = rng.uniform(0.0, 12.0, (n_veg, 2))
    vz = 0.08 * vxy[:, 0] + rng.uniform(0.5, 15.0, n_veg)
    v = np.column_stack([vxy, vz])
    xyz = np.vstack([g, v])
    return PointCloud(xyz, {"intensity": rng.integers(0, 1000, len(xyz)).astype(np.uint16)})


def _raster(prefix, r):
    return {f"{prefix}_data": r.data, f"{prefix}_origin": np.array([r.xmin, r.ymin, r.resolution])}


def classify():
    c = plot()
    csf = ground.classify_ground_csf(c, cloth_resolution=0.5, rigidness=2, class_threshold=0.3)
    return {
        "csf_mask": ground.classify_ground_csf(c, return_mask=True),
        "csf_class": csf.attrs["classification"],
        "csf_r3_mask": ground.classify_ground_csf(c, cloth_resolution=0.8, rigidness=3,
                                                  class_threshold=0.2, iterations=300,
                                                  time_step=0.5, return_mask=True),
        "pmf_mask": ground.classify_ground_pmf(c, return_mask=True),
        "pmf_class": ground.classify_ground_pmf(c, cell_size=0.4, max_window=6.0, slope=0.2,
                                                initial_distance=0.1, max_distance=1.5)
        .attrs["classification"],
        "ground_mask": ground.ground_mask(csf),
    }


def terrain():
    c = ground.classify_ground_csf(plot())
    dtm = ground.make_dtm(c, resolution=0.5)
    dtm_b = ground.make_dtm(c, resolution=0.75, bounds=(-1.0, -0.5, 13.0, 12.5))
    n = ground.normalize_height(c, dtm)
    n2 = ground.normalize_height(c, dtm_b, attr="hag")
    f = ground.flatten(c, dtm)
    chm = ground.make_chm(n, resolution=1.0)
    chm_b = ground.make_chm(f, resolution=0.8, height_attr="missing", bounds=(0.0, 0.0, 12.0, 12.0),
                            min_height=2.0)
    out = {}
    out.update(_raster("dtm", dtm))
    out.update(_raster("dtm_b", dtm_b))
    out.update(_raster("chm", chm))
    out.update(_raster("chm_b", chm_b))
    out["height"] = n.attrs["height"]
    out["hag"] = n2.attrs["hag"]
    out["flat_xyz"] = f.xyz
    out["flat_class"] = f.attrs["classification"]
    return out


def sample_given():
    """normalize_height and flatten on a hand-made raster with an edge."""
    rng = np.random.default_rng(6)
    r = Raster(rng.normal(0.0, 1.0, (7, 9)), 1.0, -2.0, 0.5)
    xyz = rng.uniform(-1.0, 7.0, (400, 3))
    c = PointCloud(xyz)
    return {"height": ground.normalize_height(c, r).attrs["height"],
            "flat": ground.flatten(c, r).xyz}


CASES = {"classify": classify, "terrain": terrain, "sample_given": sample_given}
