"""Parity cases for sylva.voxels (ray-traced grids, their layer summaries and
files, tree sampling, scanner beams and leaf projection)."""

import tempfile
from pathlib import Path

import numpy as np

from sylva import voxels
from sylva.qsm import QSM
from sylva.raster import Raster
from sylva.shots import Shots


def _canopy(seed, n=6000):
    """Pulses up and down through a 3 m x 3 m plot with a patchy roof and floor."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 3, (n, 2))
    up = (rng.uniform(size=n) < 0.5) | (xy[:, 0] < 2.2)
    origin = np.column_stack([xy, np.where(up, 0.2, 7.0)])
    tilt = rng.normal(0, 0.05, (n, 2))
    direction = np.column_stack([tilt, np.where(up, 1.0, -1.0)])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    counts, ranges, cls, tree, inten = [], [], [], [], []
    for i in range(n):
        dz = direction[i, 2]
        hits = []
        roof = xy[i, 0] < 1.5
        if roof:
            z = rng.uniform(3.0, 4.0)
            hits.append((z - origin[i, 2]) / dz)
        if not up[i] and rng.uniform() < 0.6:
            hits.append((0.02 - origin[i, 2]) / dz)
        if up[i] and rng.uniform() < 0.3:
            hits.append((rng.uniform(1.0, 2.0) - origin[i, 2]) / dz)
        hits = sorted(h for h in hits if h > 0)
        counts.append(len(hits))
        for h in hits:
            z = origin[i, 2] + h * dz
            ranges.append(h)
            cls.append(2 if z < 0.1 else (4 if z > 2.5 else 6))
            tree.append(1 if xy[i, 1] < 1.5 else 2)
            inten.append(rng.uniform(10, 50))
    counts = np.array(counts)
    shots = Shots(origin, direction, np.r_[0, np.cumsum(counts)[:-1]], counts, np.array(ranges))
    shots.echo_attrs["classification"] = np.array(cls, np.uint8)
    shots.echo_attrs["tree_id"] = np.array(tree, np.int32)
    shots.echo_attrs["intensity"] = np.array(inten, np.float64)
    return shots


BOUNDS = ((0, 0, 0), (3, 3, 7))
FIELDS = ["num_hits", "num_beams", "num_hit_leaf", "num_hit_wood", "num_beams_occluded", "path_length",
          "free_path_length"]
METRICS = ["state", "pad_fpl", "lad_fpl", "wad_fpl", "transmittance", "distance_from_ground", "mean_zenith_angle"]


def _grid_out(prefix, g, fields=FIELDS, metrics=METRICS):
    out = {f"{prefix}origin": g.origin, f"{prefix}voxel_size": g.voxel_size, f"{prefix}shape": np.array(g.shape),
           f"{prefix}fields": ",".join(g.fields), f"{prefix}metrics": ",".join(g.metrics), f"{prefix}repr": repr(g),
           f"{prefix}observed": g.observed, f"{prefix}z_levels": g.z_levels()}
    for name in fields:
        if name in g.fields:
            out[f"{prefix}{name}"] = g[name]
    for name in metrics:
        out[f"{prefix}{name}"] = getattr(g, name)
    X, Y, Z = g.centers()
    out.update({f"{prefix}X": X, f"{prefix}Y": Y, f"{prefix}Z": Z})
    for name, mb in (("pad_fpl", 1), ("num_hits", 5), ("transmittance", 50), ("free_path_length", 1)):
        out[f"{prefix}profile_{name}_{mb}"] = g.profile(name, min_beams=mb)
    for args in ((), (0.5,), (1.0, 4.0), (0.0, 0.2), (5.0, 1.0), (0.0, 6.5)):
        p = g.occlusion_profile(*args)
        key = "_".join(str(a) for a in args)
        for k, v in p.items():
            if k == "total":
                for t, x in v.items():
                    out[f"{prefix}occ{key}_total_{t}"] = x
            else:
                out[f"{prefix}occ{key}_{k}"] = v
        out[f"{prefix}map{key}"] = g.observed_map(*args)
    return out


def in_memory():
    s = _canopy(1)
    out = {}
    g = voxels.ray_voxelize(s, 0.5, BOUNDS, ground_class=2, leaf_classes=[4], wood_classes=[6], occlusion=True,
                            attenuation=["fpl", "transmittance"], laser="VZ-400")
    out.update(_grid_out("classes_", g))
    dtm = Raster(np.full((4, 4), 0.0) + np.arange(4)[:, None] * 0.01, -0.5, -0.5, 1.0)
    g = voxels.ray_voxelize(s, 0.5, BOUNDS, dtm=dtm, ground_distance=0.15, wood_classes=[6], occlusion=True,
                            attenuation="transmittance", beam=(0.005, 0.0003))
    out.update(_grid_out("dtm_", g))
    rng = np.random.default_rng(2)
    ground = s.echo_attrs["classification"] == 2
    foliage = rng.integers(0, 4, s.n_echoes).astype(np.uint8)
    g = voxels.ray_voxelize(s, 0.75, None, ground=ground, foliage=foliage, weighting="relative", tree_attr=None)
    out.update(_grid_out("arrays_", g))
    g = voxels.ray_voxelize(s, 0.5, BOUNDS, ground_class=2, weighting="strongest", inclination=True,
                            leaf_classes=[4], wood_classes=[6])
    out.update(_grid_out("iad_", g, metrics=["state", "pad_fpl", "g_plant", "g_leaf"]))
    for tid, d in g.tree_iad.items():
        for k, v in d.items():
            out[f"iad_tree{tid}_{k}"] = v if not isinstance(v, str) and v is not None else str(v)
    g = voxels.ray_voxelize(s, 1.0, BOUNDS, weighting="first")
    out.update(_grid_out("plain_", g))
    out["missing_attr"] = _error(lambda: voxels.ray_voxelize(s, 1.0, BOUNDS, ground_class=2, class_attr="nope"))
    out["missing_intensity"] = _error(lambda: voxels.ray_voxelize(s, 1.0, BOUNDS, weighting="relative",
                                                                  intensity_attr="nope"))
    out["bad_length"] = _error(lambda: voxels.ray_voxelize(s, 1.0, BOUNDS, ground=ground[:-1]))
    out["laser_and_beam"] = _error(lambda: voxels.ray_voxelize(s, 1.0, BOUNDS, laser="VZ-400", beam=(0.1, 0.1)))
    return out


def _error(fn):
    try:
        fn()
    except (ValueError, KeyError) as e:
        return f"{type(e).__name__}: {e}"
    return "no error"


def from_file():
    s = _canopy(3, n=3000)
    out = {}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "plot.shots"
        s.save(p)
        g = voxels.ray_voxelize(p, 0.5, BOUNDS, ground_class=2, leaf_classes=[4], wood_classes=[6], occlusion=True)
        out.update(_grid_out("file_", g))
        g = voxels.ray_voxelize(str(p), 1.0, dtm=Raster(np.zeros((3, 3)), 0.0, 0.0, 1.0))
        out.update(_grid_out("file_dtm_", g))
        out["file_arrays"] = _error(lambda: voxels.ray_voxelize(p, 1.0, ground=np.zeros(s.n_echoes, bool)))
    return out


def files_and_wood():
    s = _canopy(4, n=3000)
    g = voxels.ray_voxelize(s, 0.5, BOUNDS, ground_class=2, leaf_classes=[4], wood_classes=[6], inclination=True,
                            occlusion=True)
    cyl = np.array([[1.0, 1.0, 0.1, 0, 0, 1, 2.8, 0.25, -1, 0, 0, 0],
                    [2.0, 2.0, 0.0, 0.6, 0, 0.8, 1.5, 0.1, -1, 0, 1, 0]], float)
    cyl[1, 3:6] /= np.linalg.norm(cyl[1, 3:6])
    g.add_wood_volume(QSM(cyl[:1]))
    out = {"wood_volume_1": g.wood_volume.copy(), "wood_density_1": g.wood_volume_density.copy()}
    g.add_wood_volume([QSM(cyl[1:]), QSM(cyl[:1])])
    out["wood_volume_2"] = g.wood_volume
    out["wood_density_2"] = g.wood_volume_density
    out["fields"] = ",".join(g.fields)
    with tempfile.TemporaryDirectory() as d:
        for name, kw in [("a.vox", {}), ("b.txt", {}), ("c.vox", {"filled_only": True}),
                         ("d.dat", {"format": "vox", "include_unobserved": True}), ("e.VOX", {"format": "text"})]:
            out[f"n_{name}"] = g.write(Path(d) / name, **kw)
            out[f"text_{name}"] = (Path(d) / name).read_text()
        g.write_iad_csv(str(Path(d) / "iad.csv"))
        out["iad_csv"] = (Path(d) / "iad.csv").read_text()
    out["to_dict"] = ",".join(sorted(g.to_dict()))
    out["to_dict_pad"] = g.to_dict(["pad_fpl"])["pad_fpl"]
    return out


def tree_sampling():
    s = _canopy(5)
    g = voxels.ray_voxelize(s, 0.5, BOUNDS, ground_class=2, occlusion=True, attenuation="transmittance")
    rng = np.random.default_rng(6)
    pts = np.vstack([np.c_[rng.uniform(0.2, 1.3, (80, 2)), rng.uniform(0.5, 2.5, 80)],
                     np.c_[rng.uniform(1.7, 2.8, (80, 2)), rng.uniform(0.5, 3.5, 80)],
                     np.c_[rng.uniform(0.2, 2.8, (20, 2)), rng.uniform(0.5, 1.0, 20)]])
    lab = np.r_[np.zeros(80, int), np.full(80, 3), np.full(20, -1)]
    out = {}
    for k, v in voxels.tree_sampling(g, pts, lab).items():
        out[f"default_{k}"] = v
    for k, v in voxels.tree_sampling(g, pts, lab, min_beams=3.0, above=1.0).items():
        out[f"tuned_{k}"] = v
    return out


def beams_and_projection():
    out = {}
    for name in ("VZ-400", "LMS-Q780", "FARO-FOCUS-X330"):
        out[f"laser_{name}"] = np.array(voxels.laser_spec(name))
    out["laser_bad"] = _error(lambda: voxels.laser_spec("nope"))
    theta = np.linspace(0, np.pi / 2, 11)
    for lad in ("spherical", "uniform", "planophile", "erectophile", "plagiophile", "extremophile"):
        out[f"g_{lad}"] = voxels.leaf_projection(theta, lad)
    out["g_ellipsoidal"] = voxels.leaf_projection(theta, "ellipsoidal", [1.7])
    out["g_beta"] = voxels.leaf_projection(0.4, "twoParamBeta", [2.0, 1.5])
    out["g_bad"] = _error(lambda: voxels.leaf_projection(theta, "round"))
    return out


CASES = {"in_memory": in_memory, "from_file": from_file, "files_and_wood": files_and_wood,
         "tree_sampling": tree_sampling, "beams_and_projection": beams_and_projection}
