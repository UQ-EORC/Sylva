"""Parity cases for sylva.qsm (clipping, buttress fusion, writers, build_plot)."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np

from sylva import PointCloud, qsm, trees


def _rows(seed, n=60):
    """A random model: a stem chain, branches off it, some odd rows."""
    rng = np.random.default_rng(seed)
    rows = np.zeros((n, 12))
    z = 0.0
    for i in range(n):
        if i < n // 3:
            a = np.array([rng.normal(0, 0.1), rng.normal(0, 0.1), 1.0])
            start = np.array([0.0, 0.0, z]) if i == 0 else rows[i - 1, 0:3] + rows[i - 1, 3:6] * rows[i - 1, 6]
            parent, order, branch = i - 1, 0, 0
        else:
            parent = int(rng.integers(0, i))
            a = rng.normal(0, 1, 3)
            start = rows[parent, 0:3] + rows[parent, 3:6] * rows[parent, 6] * rng.uniform()
            order, branch = int(rows[parent, 9]) + 1, i
        a = a / np.linalg.norm(a)
        rows[i, 0:3] = start
        rows[i, 3:6] = a
        rows[i, 6] = rng.uniform(0.05, 0.6)
        rows[i, 7] = rng.uniform(0.01, 0.3)
        rows[i, 8] = parent
        rows[i, 9] = order
        rows[i, 10] = branch
        rows[i, 11] = int(rng.integers(0, 200))
    rows[n // 3 + 1, 3:6] = [1.0, 0.0, 0.0]                 # horizontal
    rows[n // 3 + 2, 3:6] = [0.0, 0.6, -0.8]                # pointing down
    rows[n // 3 + 3, 6] = 0.0                               # no length
    rows[n // 3 + 4, 8] = n + 5                             # a parent that is not there
    return rows


def _valid_rows(seed, n):
    """As :func:`_rows`, every parent present (metrics need that)."""
    rows = _rows(seed, n)
    rows[n // 3 + 4, 8] = 0
    return rows


def _model_values(prefix, m):
    return {f"{prefix}cylinders": m.cylinders, f"{prefix}end": m.end, f"{prefix}volumes": m.volumes,
            f"{prefix}total_volume": m.total_volume, f"{prefix}stem_volume": m.stem_volume,
            f"{prefix}branch_volume": m.branch_volume, f"{prefix}total_length": m.total_length,
            f"{prefix}max_branch_order": m.max_branch_order}


def clipping():
    out = {}
    for seed in range(3):
        m = qsm.QSM(_rows(seed))
        out.update(_model_values(f"s{seed}_", m))
        zs = np.r_[-1.0, np.quantile(m.start[:, 2], [0.1, 0.4, 0.8]), m.start[5, 2], 100.0]
        for k, z in enumerate(zs):
            out[f"s{seed}_volume_above_{k}"] = m.volume_above(float(z))
            out.update(_model_values(f"s{seed}_above_{k}_", m.above(float(z))))
    empty = qsm.QSM(np.zeros((0, 12)))
    out.update(_model_values("empty_", empty))
    out["empty_volume_above"] = empty.volume_above(1.0)
    out["empty_above"] = empty.above(1.0).cylinders
    # A long model, where NumPy's pairwise sums differ from a running sum.
    big = qsm.QSM(np.vstack([_rows(s, 400) for s in range(4)]))
    out.update(_model_values("big_", big))
    out["big_volume_above"] = big.volume_above(1.3)
    return out


def _bytes(path):
    return np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)


def _array_digest(a):
    a = np.ascontiguousarray(a, dtype=np.int64)
    return np.array([hashlib.sha256(a.tobytes()).hexdigest(), str(a.shape)])


def _digest(path):
    """A large file by its SHA-256 and size."""
    data = Path(path).read_bytes()
    return np.array([hashlib.sha256(data).hexdigest(), str(len(data))])


def _stem(rng, cx, cy, radius, height, density=3000, noise=0.003, flanges=False):
    n = int(density * height)
    t = rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(0, height, n)
    r = radius * (1 + (3 * np.clip(1 - h / 1.5, 0, None) * np.cos(2.5 * t) ** 8 if flanges else 0))
    r = r + rng.normal(0, noise, n)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t), h])


def _branchy(rng, cx, cy, radius, height, density=3000):
    """A stem with two side branches."""
    parts = [_stem(rng, cx, cy, radius, height, density)]
    for z0, d in [(0.5 * height, (1.0, 0.3)), (0.7 * height, (-0.4, 1.0))]:
        n = int(density * 1.5)
        s = rng.uniform(0, 1.5, n)
        t = rng.uniform(0, 2 * np.pi, n)
        d = np.array([d[0], d[1], 0.6])
        d /= np.linalg.norm(d)
        u = np.cross(d, [0.0, 0.0, 1.0])
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        r = 0.4 * radius
        p = (np.array([cx, cy, z0]) + s[:, None] * d + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v))
        parts.append(p)
    return np.vstack(parts)


def _flanged_base(seed, n=60_000):
    """A 0.25 m stem with flanges fading out by 1.5 m, heights equal to z."""
    rng = np.random.default_rng(seed)
    return _stem(rng, 0.0, 0.0, 0.25, 4.0, density=n // 4, flanges=True)


def fuse():
    out = {}
    xyz = _flanged_base(1, 40_000)
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    b = qsm.buttress_mesh(cloud, (0.0, 0.0), ground_z=0.0, top=1.4, resolution=0.03)
    auto = qsm.buttress_mesh(cloud, (0.02, -0.01), resolution=0.05, slice_height=0.1)
    out.update({f"auto_{k}": getattr(auto, k) for k in ("vertices", "faces", "volume", "top", "top_z", "heights",
                                                        "areas", "solidities", "open")})
    model = qsm.QSM(np.array([[0, 0, 0, 0, 0, 1, 1.0, 0.26, -1, 0, 0, 10],
                              [0, 0, 1.0, 0.05, 0, 0.99875, 1.2, 0.25, 0, 0, 0, 10],
                              [0.06, 0, 2.2, 0.6, 0, 0.8, 1.0, 0.1, 1, 1, 1, 10],
                              [0.06, 0, 2.2, 0, 0.1, 0.995, 1.5, 0.22, 1, 0, 0, 10]], float))
    model.cylinders[:, 3:6] /= np.linalg.norm(model.cylinders[:, 3:6], axis=1, keepdims=True)
    off = qsm.QSM(model.cylinders + [0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for name, m, kw in [("fused", model, {}), ("flat", model, {"overlap": 0.0, "contiguous": False, "sides": 8}),
                        ("deep", model, {"overlap": 0.5}), ("off", off, {}),
                        ("above", qsm.QSM(model.cylinders[3:]), {"overlap": 0.0})]:
        t = b.fuse(m, **kw)
        nv, nf = len(b.vertices), len(b.faces)
        np.testing.assert_array_equal(t.vertices[:nv], b.vertices)
        np.testing.assert_array_equal(t.faces[:nf], b.faces)
        out[f"{name}_wood_vertices"] = t.vertices[nv:]
        out[f"{name}_wood_faces"] = t.faces[nf:]
        for k in ("part", "buttress_volume", "wood_volume", "top_z", "offset", "overhang"):
            out[f"{name}_{k}"] = getattr(t, k)
        out[f"{name}_volume"] = t.volume
    out["total_volume"] = b.total_volume(model)
    # The cross-sections and the join fit on their own.
    base = qsm._section(np.asarray(b.vertices, float), np.asarray(b.faces), b.top_z - 0.05)
    wv, wf, _ = model.above(b.top_z).mesh(12, True)
    stem = qsm._section(wv, wf, b.top_z + 0.05)
    out["section_base"] = base
    out["section_stem"] = stem
    out["join_fit"] = np.array(qsm._join_fit(base, stem))
    out["join_fit_coarse"] = np.array(qsm._join_fit(base, stem, cell=0.07))
    out["join_fit_empty"] = np.array(qsm._join_fit(base, np.zeros((0, 2, 2))))
    rng = np.random.default_rng(5)
    pts = rng.uniform(-1.2, 1.2, (3000, 2))
    out["inside_base"] = qsm._inside(base, pts)
    out["inside_stem"] = qsm._inside(stem, pts)
    # A mesh with vertices exactly on the plane and flat triangles.
    v = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0.5], [2, 0, 0.5], [2, 2, 0.5]], float)
    f = np.array([[0, 1, 2], [1, 3, 2], [3, 4, 5], [0, 4, 1], [1, 4, 3]])
    for k, z in enumerate([0.0, 0.5, 0.75, 1.0]):
        out[f"section_small_{k}"] = qsm._section(v, f, z)
    return out


def writers():
    out = {}
    rng = np.random.default_rng(9)
    model = qsm.QSM(_rows(4, 30))
    xyz = _flanged_base(2, 24_000)
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    b = qsm.buttress_mesh(cloud, (0.0, 0.0), ground_z=0.0, top=1.2)
    stem = qsm.QSM(np.array([[0, 0, 0, 0, 0, 1, 3.0, 0.24, -1, 0, 0, 10],
                             [0, 0, 3.0, 0.6, 0, 0.8, 1.0, 0.08, 0, 1, 1, 10]], float))
    fused = b.fuse(stem, sides=10)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        v = rng.normal(0, 1000, (7, 3))
        v[0] = [0.03125, -0.00004, 1.23455]
        qsm.write_obj(d / "a.obj", [(v, np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]])),
                                    (v[:3] * 0.001, np.array([[0, 1, 2]], np.uint32))])
        out["write_obj"] = _bytes(d / "a.obj")
        qsm.write_obj(d / "b.obj", [(v[:3], [[0, 1, 2]])], names=["first"])
        out["write_obj_named"] = _bytes(d / "b.obj")
        qsm.write_ply_mesh(d / "a.ply", v, np.array([[0, 1, 2], [2, 3, 4]]))
        out["write_ply"] = _bytes(d / "a.ply")
        qsm.write_ply_mesh(d / "b.ply", v, np.array([[0, 1, 2], [2, 3, 4]], np.uint32),
                           np.array([[1, 2, 3], [250, 0, 7]]))
        out["write_ply_rgb"] = _bytes(d / "b.ply")
        for name, kw in [("", {}), ("_8", {"sides": 8}), ("_cont", {"contiguous": True})]:
            model.to_obj(d / "m.obj", **kw)
            out[f"model_obj{name}"] = _bytes(d / "m.obj")
            model.to_ply(d / "m.ply", **kw)
            out[f"model_ply{name}"] = _bytes(d / "m.ply")
        model.to_ply(d / "c.ply", color=(10, 200, 30))
        out["model_ply_color"] = _bytes(d / "c.ply")
        model.to_csv(d / "m.csv")
        out["model_csv"] = _bytes(d / "m.csv")
        out["from_csv"] = qsm.QSM.from_csv(d / "m.csv").cylinders
        model.to_treefile(d / "m.txt")
        out["model_treefile"] = _bytes(d / "m.txt")
        (d / "hand.csv").write_text("sx,sy,sz,ax,ay,az,length,radius,parent,branch_order,branch_id,n_points\n"
                                    "1, 2,3,0,0,1,0.5,0.1,-1,0,0,4\n\n# a note\n"
                                    "1,2,3.5,0,0,1,0.5,nan,0,0,0,4  # trailing\n")
        out["from_csv_hand"] = qsm.QSM.from_csv(d / "hand.csv").cylinders
        (d / "head.csv").write_text("sx,sy,sz,ax,ay,az,length,radius,parent,branch_order,branch_id,n_points\n")
        out["from_csv_header_only"] = qsm.QSM.from_csv(d / "head.csv").cylinders
        (d / "one.csv").write_text("header\n1,2,3,4,5,6,7,8,9,10,11,12\n")
        out["from_csv_one"] = qsm.QSM.from_csv(d / "one.csv").cylinders
        b.to_obj(d / "b.obj")
        out["buttress_obj"] = _digest(d / "b.obj")
        b.to_ply(d / "b.ply")
        out["buttress_ply"] = _digest(d / "b.ply")
        fused.to_obj(d / "f.obj")
        out["fused_obj"] = _digest(d / "f.obj")
        fused.to_ply(d / "f.ply")
        out["fused_ply"] = _digest(d / "f.ply")
        fused.to_ply(d / "g.ply", color=[5, 6, 7])
        out["fused_ply_color"] = _digest(d / "g.ply")
    return out


def _plot(seed):
    """Four trees on flat ground, one flanged, one a leafy ball, and a clump."""
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for tid, (x, y, r, h) in enumerate([(0.0, 0.0, 0.15, 5.0), (6.0, 1.0, 0.1, 4.0)], start=1):
        p = _branchy(rng, x, y, r, h)
        parts.append(p)
        labels.append(np.full(len(p), tid))
    flanged = _flanged_base(seed + 10, 40_000) + [12.0, -3.0, 0.0]
    parts.append(flanged)
    labels.append(np.full(len(flanged), 7))
    ball = rng.normal(0, 0.7, (3000, 3)) + [3.0, 8.0, 3.0]
    parts.append(ball)
    labels.append(np.full(len(ball), 4))
    dot = rng.normal(0, 0.0005, (1100, 3)) + [-4.0, 5.0, 1.0]     # thins to a point or two
    parts.append(dot)
    labels.append(np.full(len(dot), 5))
    clump = _stem(rng, 20.0, 0.0, 0.05, 0.4, density=300)
    parts.append(clump)
    labels.append(np.full(len(clump), 9))
    noise = rng.uniform(-2, 22, (500, 3)) * [1, 1, 0.3]
    parts.append(noise)
    labels.append(np.full(len(noise), -1))
    xyz = np.vstack(parts)
    return xyz, np.concatenate(labels)


def _plot_values(prefix, plot, tmp):
    out = {f"{prefix}ids": np.array(sorted(plot.models)), f"{prefix}skipped_ids": np.array(list(plot.skipped)),
           f"{prefix}skipped": np.array([str(v) for v in plot.skipped.values()]),
           f"{prefix}buttress_ids": np.array(sorted(plot.buttresses), dtype=int),
           f"{prefix}total_volume": plot.total_volume, f"{prefix}len": len(plot)}
    for t, m in plot.models.items():
        out[f"{prefix}model_{t}"] = m.cylinders
        out[f"{prefix}volume_{t}"] = plot.volume(t)
    for t, b in plot.buttresses.items():
        out[f"{prefix}buttress_{t}_vertices"] = b.vertices
        out[f"{prefix}buttress_{t}_faces"] = _array_digest(b.faces)
        out[f"{prefix}buttress_{t}_numbers"] = np.array([b.volume, b.top, b.top_z])
    rows = plot.table()
    for k in (rows[0] if rows else {}):
        out[f"{prefix}table_{k}"] = np.array([str(r[k]) for r in rows])
    d = Path(tmp) / prefix
    plot.to_csv(d.with_suffix(".csv"))
    out[f"{prefix}csv"] = _bytes(d.with_suffix(".csv"))
    plot.write_cylinders(d / "cyl", prefix="t")
    for p in sorted((d / "cyl").iterdir()):
        out[f"{prefix}cyl_{p.name}"] = _bytes(p)
    for fmt in ("ply", "obj"):
        written = plot.write_meshes(d / fmt, fmt=fmt, sides=8)
        out[f"{prefix}written_{fmt}"] = np.array([p.name for p in written])
        for p in written:
            out[f"{prefix}{fmt}_{p.name}"] = _digest(p)
    written = plot.write_meshes(d / "loose", contiguous=False, prefix="x")
    for p in written:
        out[f"{prefix}loose_{p.name}"] = _digest(p)
    return out


def build_plot():
    import warnings

    out = {}
    xyz, labels = _plot(3)
    cloud = PointCloud(xyz, {"height": xyz[:, 2].astype(np.float32)})
    stems = [trees.Tree(2, 6.0, 1.0, 0.2), trees.Tree(7, 12.02, -3.01, 0.5), trees.Tree(2, 6.01, 1.0, 0.2)]
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plain = qsm.build_plot(cloud, labels, wood=False, min_points=1000)
        out.update(_plot_values("plain_", plain, tmp))
        full = qsm.build_plot(cloud, labels, stems, wood=True, buttress=True, min_points=1000, bin_length=0.15)
        out.update(_plot_values("full_", full, tmp))
        bare = PointCloud(xyz[labels != 7])
        coarse = qsm.build_plot(bare, labels[labels != 7], voxel_size=0.05, wood=False, min_points=100,
                                buttress=True, spacing_scale=0.0)
        out.update(_plot_values("coarse_", coarse, tmp))
        out["warnings"] = np.array([str(c.message) for c in caught if "fitted to points" in str(c.message)])
        # A plot built by hand, with no point counts or heights.
        manual = qsm.PlotQSMs({3: qsm.QSM(_valid_rows(1, 20)), 1: qsm.QSM(_valid_rows(2, 10))}, {}, {5: "why"})
        out.update(_plot_values("manual_", manual, tmp))
    return out


CASES = {"clipping": clipping, "fuse": fuse, "writers": writers, "build_plot": build_plot}
