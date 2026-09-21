import numpy as np
import pytest

from sylva import qsm


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
    assert 0 < model.column("radius")[order >= 1].max() < 0.12
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
