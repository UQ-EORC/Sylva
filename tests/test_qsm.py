from collections import Counter

import numpy as np
import pytest

from sylva import PointCloud, filters, qsm, synthetic, trees


def test_fit_cylinder(rng):
    n = 500
    theta = rng.uniform(0, 2 * np.pi, n)
    t = rng.uniform(0, 3, n)
    axis = np.array([0.3, 0.1, 1.0])
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    pts = [1, 2, 3] + np.outer(t, axis)
    pts = pts + 0.2 * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    pts += rng.normal(0, 0.002, pts.shape)
    fit = qsm.fit_cylinder(pts)
    assert abs(fit["axis"] @ axis) > 0.999
    assert fit["radius"] == pytest.approx(0.2, abs=0.005)
    assert fit["rmse"] < 0.01
    noisy = np.vstack([pts, rng.uniform(0, 4, (100, 3))])
    r = qsm.fit_cylinder_ransac(noisy, threshold=0.01)
    assert r["radius"] == pytest.approx(0.2, abs=0.01)
    assert r["inliers"][:n].mean() > 0.9


def test_build_qsm(single_tree):
    model = qsm.build_qsm(single_tree, bin_length=0.5)
    assert len(model) > 5
    s = model.summary()
    true_stem = np.pi * 0.15**2 * 6
    assert s["stem_volume_m3"] == pytest.approx(true_stem, rel=0.25)
    assert s["max_branch_order"] >= 1
    assert model.branch_volume > 0
    assert s["dbh_m"] == pytest.approx(0.3, abs=0.02)


def test_qsm_io(single_tree, tmp_path):
    model = qsm.build_qsm(single_tree, bin_length=0.5)
    model.to_csv(tmp_path / "q.csv")
    back = qsm.QSM.from_csv(tmp_path / "q.csv")
    assert back.cylinders.shape == (len(model), 12)
    model.to_treefile(tmp_path / "q_trees.txt")
    text = (tmp_path / "q_trees.txt").read_text()
    assert "x,y,z,radius" in text


def test_mesh_export(single_tree, tmp_path):
    model = qsm.build_qsm(single_tree, bin_length=0.5)
    v, f, owner = model.mesh(sides=8)
    assert v.shape[1] == 3 and f.shape[1] == 3
    assert len(v) == len(model) * (2 * 8 + 2) and len(f) == len(model) * 4 * 8
    assert f.max() < len(v) and len(owner) == len(f) and owner.max() == len(model) - 1
    model.to_obj(tmp_path / "t.obj", sides=8)
    text = (tmp_path / "t.obj").read_text()
    assert text.count("\nv ") == len(v) and text.count("\nf ") == len(f)
    model.to_ply(tmp_path / "t.ply", sides=8)
    raw = (tmp_path / "t.ply").read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    assert f"element face {len(f)}".encode() in head
    assert len(body) == len(v) * 12 + len(f) * (1 + 12 + 3)


def test_taper_clamp(rng):
    # A cylinder with a foliage blob attached must not grow a metre-wide branch.
    theta = rng.uniform(0, 2 * np.pi, 3000)
    z = rng.uniform(0, 4, 3000)
    stem = np.column_stack([0.1 * np.cos(theta), 0.1 * np.sin(theta), z])
    blob = rng.normal([0.5, 0, 3.0], 0.4, (1500, 3))
    pc = __import__("sylva").PointCloud(np.vstack([stem, blob]))
    model = qsm.build_qsm(pc, bin_length=0.4, max_radius=0.15)
    assert model.column("radius").max() <= 0.15 + 1e-9
    assert model.total_volume < 0.5


def test_skeletonize(single_tree):
    sk = qsm.skeletonize(single_tree, bin_length=0.5)
    assert sk["segment_id"].max() >= 10
    assert (sk["segment_id"] >= 0).mean() > 0.95
    assert sk["edges"].shape[1] == 2


def test_build_qsm_defaults(single_tree):
    # The default path: 10 cm shells, greedy chaining, pipe-model branches.
    model = qsm.build_qsm(single_tree)
    s = model.summary()
    assert s["stem_volume_m3"] == pytest.approx(np.pi * 0.15**2 * 6, rel=0.2)
    assert s["dbh_m"] == pytest.approx(0.3, abs=0.02)
    order = model.column("branch_order")
    stem_length = model.column("length")[order == 0].sum()
    assert 5.5 < stem_length < 7.0  # a zigzagging chain would be much longer
    radius, z = model.column("radius"), model.start[:, 2]
    assert radius[(order == 0) & (z < 5.0)].max() < 0.16  # no inflated butt
    assert radius.max() < 0.21
    # The branch (r = 0.05) is reconstructed, thinner than the stem.
    assert s["max_branch_order"] >= 1
    assert 0 < model.column("radius")[order >= 1].max() < 0.15
    # Nothing is prolonged below the cloud.
    assert model.start[:, 2].min() > single_tree.z.min() - 0.05


def test_root_in_main_component(single_tree, rng):
    # A stray clump nearest the given stem position, out of reach of the tree
    # (edges are capped at 1 m), must not become the root.
    clump = rng.normal([1.6, 0.0, 0.05], 0.02, (40, 3))
    pc = __import__("sylva").PointCloud(np.vstack([single_tree.xyz, clump]))
    model = qsm.build_qsm(pc, base_xy=(1.6, 0.0))
    assert model.stem_volume == pytest.approx(np.pi * 0.15**2 * 6, rel=0.2)
    root = model.start[model.column("parent") < 0][0]
    assert np.hypot(root[0], root[1]) < 0.3


def _rough_trunk_tree(rng):
    """A rough-barked trunk (5 cm radial noise), a clean branch and a foliage blob."""
    n = 3000
    theta, z = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 8, n)
    r = 0.25 + rng.normal(0, 0.05, n)
    trunk = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    m = 4000
    t, phi = rng.uniform(0, 3, m), rng.uniform(0, 2 * np.pi, m)
    branch = np.column_stack([0.25 + t, 0.03 * np.cos(phi), 7.5 + 0.2 * t + 0.03 * np.sin(phi)])
    leaves = rng.normal([3.3, 0.0, 8.2], 0.25, (3000, 3))
    return trunk, branch, leaves


def test_wood_points_passage_recovers_rough_trunk(rng):
    sylva = __import__("sylva")
    trunk, branch, leaves = _rough_trunk_tree(rng)
    pc = sylva.PointCloud(np.vstack([trunk, branch, leaves]))

    def kept(cloud, part):
        tree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(cloud.xyz)
        return (tree.query(part)[0] < 1e-9).mean()

    aniso = qsm.wood_points(pc, voxel_size=None, passage=False, medium_threshold=1.0,
                            dilate_dist=0.0, component_min=1)
    default = qsm.wood_points(pc, voxel_size=None)
    wide = qsm.wood_points(pc, voxel_size=None, assign_scale=0.03)
    # Rough bark is neither planar nor linear: anisotropy alone drops it. The
    # paths to the crown run through it, and a scaled reach takes the section.
    assert kept(aniso, trunk) < 0.3
    assert kept(default, trunk) > kept(aniso, trunk) + 0.1
    assert kept(wide, trunk) > 0.65
    for wood in (default, wide):
        assert kept(wood, branch) > 0.95
        assert kept(wood, leaves) < 0.6
    # The anisotropy-only result is what passage=False returns.
    assert kept(aniso, leaves) < 0.1


def test_wood_points_thins(single_tree):
    wood = qsm.wood_points(single_tree, voxel_size=0.05)
    assert 0 < len(wood) < len(single_tree)
    d = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(wood.xyz).query(wood.xyz, k=2)[0][:, 1]
    assert np.median(d) > 0.02


def test_tree_metrics(single_tree):
    model = qsm.build_qsm(single_tree)
    m = model.metrics()
    assert m["height"] == pytest.approx(6.0, abs=0.2)
    assert m["dbh"] == pytest.approx(0.3, abs=0.02)
    assert m["total_volume"] == pytest.approx(m["stem_volume"] + m["branch_volume"])
    assert m["n_branches_by_order"][0] == 1 and sum(m["n_branches_by_order"][1:]) >= 1
    assert sum(m["length_by_order"]) == pytest.approx(m["total_length"])
    assert m["lean"] < 3 and m["sweep"] < 0.05  # a straight vertical stem
    assert m["crown_base_height"] == pytest.approx(4.0, abs=0.3)  # the branch leaves at 4 m
    assert 0.9 < m["measured_volume_fraction"] <= 1.0
    assert 0 < m["path_fraction"] <= 1
    assert len(m["taper_heights"]) == len(m["taper_radii"]) > 10
    b = model.branches()
    stem = b["order"] == 0
    assert stem.sum() == 1 and b["parent"][stem][0] == -1 and np.isnan(b["insertion_angle"][stem][0])
    first = np.flatnonzero(b["order"] == 1)[np.argmax(b["length"][b["order"] == 1])]
    # The branch runs along +x rising 0.3 per metre: 73 deg from the vertical stem.
    assert b["zenith"][first] == pytest.approx(73.3, abs=6)
    assert b["insertion_angle"][first] == pytest.approx(73.3, abs=8)
    assert b["length"][first] == pytest.approx(2.0, abs=0.3)
    assert b["tortuosity"][first] < 1.1


def test_crown_shape(rng):
    from sylva import trees

    # A cone of radius 2 m from 5 to 11 m, offset 1 m east of the stem at (0, 0).
    n = 20000
    z = rng.uniform(5, 11, n)
    r = 2.0 * (11 - z) / 6 * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    pts = np.c_[1.0 + r * np.cos(a), r * np.sin(a), z]
    c = trees.crown_shape(pts, base_xy=(0.0, 0.0), crown_base=5.0, slice_height=0.25)
    assert c["projected_area"] == pytest.approx(np.pi * 4, rel=0.1)  # few samples reach the rim
    assert c["volume"] == pytest.approx(np.pi * 4 * 6 / 3, rel=0.1)
    assert c["offset"] == pytest.approx(1.0, abs=0.05) and abs(c["offset_direction"]) < 5
    assert c["asymmetry"] == pytest.approx(0.5, abs=0.05)
    assert c["top_height"] == pytest.approx(11, abs=0.05)


def test_build_plot_models_every_tree(rng, tmp_path):
    from conftest import make_stem

    # Three stems of different size, plus a clump too small to model.
    parts, labels = [], []
    for tid, (x, r, h) in enumerate([(0.0, 0.15, 6.0), (6.0, 0.10, 4.0), (12.0, 0.2, 7.0)], start=1):
        pts = make_stem(rng, x, 0.0, r, h, density=3000)
        parts.append(pts)
        labels.append(np.full(len(pts), tid))
    clump = make_stem(rng, 20.0, 0.0, 0.05, 0.4, density=300)
    parts.append(clump)
    labels.append(np.full(len(clump), 4))
    xyz = np.vstack(parts)
    labels = np.concatenate(labels)
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})

    plot = qsm.build_plot(cloud, labels, wood=False, min_points=2000)
    assert set(plot.models) == {1, 2, 3} and set(plot.skipped) == {4}
    assert "points" in plot.skipped[4]
    assert plot.total_volume == pytest.approx(sum(m.total_volume for m in plot.models.values()))
    # The same as fitting that tree on its own.
    alone = qsm.build_qsm(filters.voxel_downsample(cloud[labels == 1], 0.01),
                          base_xy=tuple(np.median(cloud.xyz[labels == 1][:, :2], axis=0)))
    assert plot.models[1].total_volume == pytest.approx(alone.total_volume, rel=1e-9)
    # Volume rises with stem size, as the stems were built.
    assert plot.volume(3) > plot.volume(1) > plot.volume(2)

    rows = plot.table()
    assert len(rows) == 3 and rows[0]["tree_id"] == 1
    assert rows[0]["points"] > 0 and rows[0]["height_m"] == pytest.approx(6.0, abs=0.2)
    plot.to_csv(tmp_path / "trees.csv")
    assert (tmp_path / "trees.csv").read_text().startswith("tree_id,points,volume_m3")
    plot.write_cylinders(tmp_path / "qsms")
    assert sorted(p.name for p in (tmp_path / "qsms").glob("*.csv")) == ["tree1.csv", "tree2.csv", "tree3.csv"]
    # Surface meshes, one file per tree, in either format.
    written = plot.write_meshes(tmp_path / "meshes")
    assert [p.name for p in written] == ["tree1.ply", "tree2.ply", "tree3.ply"]
    assert (tmp_path / "meshes" / "tree1.ply").read_bytes()[:3] == b"ply"
    plot.write_meshes(tmp_path / "obj", fmt="obj", sides=8)
    text = (tmp_path / "obj" / "tree2.obj").read_text()
    assert text.startswith("#") or text.startswith("o ") or text.startswith("v ")
    assert text.count("\nv ") > 50
    with pytest.raises(ValueError):
        plot.write_meshes(tmp_path / "nope", fmt="stl")

    with pytest.raises(ValueError):
        qsm.build_plot(cloud, labels[:-1])


def test_build_plot_takes_stem_centres(rng):
    from conftest import make_stem

    pts = make_stem(rng, 3.0, -2.0, 0.12, 5.0, density=3000)
    cloud = PointCloud(pts, {"height": pts[:, 2].copy()})
    labels = np.ones(len(pts), int)
    stem = trees.Tree(tree_id=1, x=3.0, y=-2.0, dbh=0.24)
    plot = qsm.build_plot(cloud, labels, [stem], wood=False)
    assert len(plot) == 1
    base = plot.models[1].start[0]
    assert np.hypot(base[0] - 3.0, base[1] + 2.0) < 0.2


def test_buttress_mesh_of_a_flanged_base():
    # Five flanges fading out by 2 m on a 0.3 m stem, seen all round; area known per slice.
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    pts, truth = [], 0.0
    for k in range(100):
        h = (k + 0.5) * 0.04
        r = 0.3 * (1 + 3 * max(1 - h / 2, 0) * np.cos(2.5 * t) ** 8)
        pts.append(np.column_stack([r * np.cos(t), r * np.sin(t), np.full(len(t), h)]))
        if h < 2.0:
            truth += 0.5 * np.sum(r**2) * (2 * np.pi / len(t)) * 0.04
    xyz = np.vstack(pts)
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    b = qsm.buttress_mesh(cloud, (0.0, 0.0), ground_z=0.0, top=2.0)
    assert abs(b.volume - truth) / truth < 0.05
    assert len(b.faces) and b.solidities[0] < 0.7
    # Joined to a QSM: cylinders above the top plane count, those below do not.
    model = qsm.QSM(np.array([[0, 0, 0, 0, 0, 1, 4.0, 0.3, -1, 0, 0, 10]], float))
    assert model.volume_above(2.0) == pytest.approx(model.total_volume / 2)
    assert b.total_volume(model) == pytest.approx(b.volume + model.total_volume / 2)
    # The mesh closes on the ground, so it holds the volume it reports.
    edges = Counter()
    for t in b.faces:
        for k in range(3):
            e = (min(int(t[k]), int(t[(k + 1) % 3])), max(int(t[k]), int(t[(k + 1) % 3])))
            edges[e] += 1
    assert all(n % 2 == 0 for n in edges.values())  # 4 where voxels touch corner to corner
    v, f = b.vertices, b.faces
    enclosed = float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6)
    assert enclosed == pytest.approx(b.volume, rel=0.1)


def test_ground_around_the_base_is_not_closed_into_the_buttress():
    # A plain 0.3 m stem, with one scan line on the ground ringing it at 1.5 m.
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    stem = [np.column_stack([0.3 * np.cos(t), 0.3 * np.sin(t), np.full(len(t), h)])
            for h in np.arange(0.02, 1.5, 0.02)]
    ring = np.column_stack([1.5 * np.cos(t), 1.5 * np.sin(t), np.full(len(t), 0.01)])
    xyz = np.vstack(stem + [ring])
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    truth = np.pi * 0.3**2 * 1.4
    # The ring encloses the stem, so a flood fill would take the whole disc.
    loose = qsm.buttress_mesh(cloud, (0.0, 0.0), ground_z=0.0, top=1.4, max_flare=0.0)
    assert loose.areas[0] > 6.0 and loose.volume > 1.6 * truth
    b = qsm.buttress_mesh(cloud, (0.0, 0.0), ground_z=0.0, top=1.4)
    assert b.areas[0] < 0.6 and b.volume == pytest.approx(truth, rel=0.1)


def test_above_cuts_the_model_at_a_plane():
    # A stem of four 1 m cylinders with a branch off the second.
    rows = [[0, 0, float(k), 0, 0, 1, 1.0, 0.2, k - 1, 0, 0, 10] for k in range(4)]
    rows.append([0, 0, 1.5, 1, 0, 0, 1.0, 0.1, 1, 1, 1, 10])   # horizontal, below the plane
    model = qsm.QSM(np.array(rows, float))
    cut = model.above(2.0)
    assert len(cut) == 2 and cut.total_volume == pytest.approx(model.volume_above(2.0))
    assert cut.start[:, 2].min() == pytest.approx(2.0)
    np.testing.assert_array_equal(cut.column("parent"), [-1, 0])  # the base cylinder is gone
    # A cylinder crossing the plane keeps only its upper part.
    lean = qsm.QSM(np.array([[0, 0, 0, 0, 0.6, 0.8, 5.0, 0.2, -1, 0, 0, 10]], float))
    kept = lean.above(2.0)
    assert kept.column("length")[0] == pytest.approx(5.0 - 2.0 / 0.8)
    assert lean.above(100.0).cylinders.shape == (0, 12) and model.above(-1.0).total_volume == pytest.approx(model.total_volume)


def test_fuse_joins_a_buttress_to_the_model():
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    pts = []
    for k in range(50):
        h = (k + 0.5) * 0.04
        r = 0.3 * (1 + 3 * max(1 - h / 2, 0) * np.cos(2.5 * t) ** 8)
        pts.append(np.column_stack([r * np.cos(t), r * np.sin(t), np.full(len(t), h)]))
    xyz = np.vstack(pts)
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    b = qsm.buttress_mesh(cloud, (0.0, 0.0), ground_z=0.0, top=1.0)
    model = qsm.QSM(np.array([[0, 0, 0, 0, 0, 1, 4.0, 0.3, -1, 0, 0, 10],
                              [0, 0, 4, 0, 0, 1, 2.0, 0.2, 0, 0, 0, 10]], float))
    fused = b.fuse(model)
    assert fused.buttress_volume == pytest.approx(b.volume)
    assert fused.wood_volume == pytest.approx(model.volume_above(b.top_z))
    assert fused.volume == pytest.approx(b.total_volume(model))
    assert len(fused.faces) == len(fused.part) and set(np.unique(fused.part)) == {0, 1}
    assert len(fused.faces) > len(b.faces)  # the wood is in there too
    # The wood reaches down inside the base by `overlap`, and volumes ignore it.
    def wood_bottom(m):
        return m.vertices[m.faces[m.part == 1]].reshape(-1, 3)[:, 2].min()
    assert wood_bottom(fused) == pytest.approx(b.top_z - 0.1)
    assert wood_bottom(b.fuse(model, overlap=0.0)) == pytest.approx(b.top_z)
    assert b.fuse(model, overlap=0.5).volume == pytest.approx(fused.volume)
    # The stem sits over its base, so the join is nearly concentric.
    assert fused.offset < 0.05 and fused.overhang < 0.05
    # Shift the stem half a metre sideways and the join reports it.
    off = qsm.QSM(np.array([[0.5, 0, 0, 0, 0, 1, 4.0, 0.3, -1, 0, 0, 10]], float))
    assert b.fuse(off).offset > 0.3 and b.fuse(off).overhang > 0.3


def test_fused_mesh_writes_both_parts(tmp_path):
    b = qsm.Buttress(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]),
                     np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], np.uint32),
                     0.16, 1.0, 1.0, np.array([0.0]), np.array([0.5]), np.array([0.9]), np.array([False]))
    model = qsm.QSM(np.array([[0, 0, 0, 0, 0, 1, 3.0, 0.2, -1, 0, 0, 10]], float))
    fused = b.fuse(model, sides=8)
    fused.to_obj(tmp_path / "tree.obj")
    text = (tmp_path / "tree.obj").read_text()
    assert "o buttress" in text and "o wood" in text
    fused.to_ply(tmp_path / "tree.ply")
    assert (tmp_path / "tree.ply").read_bytes()[:3] == b"ply"


def test_contiguous_mesh_is_closed_and_smaller():
    model = qsm.build_qsm(qsm.wood_points(synthetic.tree(seed=2)))

    def closed(f):
        edges = {}
        for t in f:
            for k in range(3):
                e = (min(t[k], t[(k + 1) % 3]), max(t[k], t[(k + 1) % 3]))
                edges[e] = edges.get(e, 0) + 1
        return all(c == 2 for c in edges.values())

    def enclosed(v, f):
        """Signed volume: positive when the triangles face outwards."""
        return float(np.einsum("ij,ij->i", v[f[:, 0]],
                               np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])).sum() / 6)

    def oriented(f):
        """Every edge is walked once each way, so the winding is consistent."""
        seen = set()
        for t in f:
            for k in range(3):
                e = (int(t[k]), int(t[(k + 1) % 3]))
                if e in seen:
                    return False
                seen.add(e)
        return all((b, a) in seen for a, b in seen)

    v0, f0, o0 = model.mesh(12)
    v1, f1, o1 = model.mesh(12, contiguous=True)
    assert closed(f0) and closed(f1)
    assert len(f1) < 0.7 * len(f0) and len(v1) < 0.7 * len(v0)
    assert len(o1) == len(f1) and o1.max() < len(model)
    # Both meshes are wound outwards, so the volume they enclose is positive
    # and near the cylinders' own: the tube mitres the joints slightly away,
    # the per-cylinder mesh keeps every drum whole.
    assert oriented(f0) and oriented(f1)
    assert enclosed(v1, f1) == pytest.approx(model.total_volume, rel=0.1)
    assert enclosed(v0, f0) == pytest.approx(model.total_volume, rel=0.1)
