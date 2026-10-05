import numpy as np
import pytest

from sylva import PointCloud, leaves, qsm


def make_leaves(rng, n_leaves, inclination_deg, centre=(0.0, 0.0, 5.0), spread=1.0, size=0.06, density=40000):
    """Square leaves of side ``size`` with the given normal inclinations, sampled as points."""
    pts, area = [], 0.0
    for theta in np.radians(inclination_deg(rng, n_leaves)):
        phi = rng.uniform(0, 2 * np.pi)
        n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        u = np.cross(n, [0, 0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        c = np.asarray(centre) + rng.uniform(-spread, spread, 3)
        m = int(size * size * density)
        ab = rng.uniform(-size / 2, size / 2, (m, 2))
        pts.append(c + ab[:, :1] * u + ab[:, 1:] * v + rng.normal(0, 0.001, (m, 3)))
        area += size * size
    return np.vstack(pts), area


def test_analytic_distributions():
    sph = leaves.LeafAngleDistribution.from_type("spherical")
    assert sph.mean_deg == pytest.approx(57.3, abs=0.3) and sph.de_wit == "spherical"
    np.testing.assert_allclose(sph.g([0.0, 0.6, 1.2]), 0.5, atol=0.01)  # G = 0.5 at every zenith
    assert sph.chi == pytest.approx(1.0, abs=0.1)
    plano = leaves.LeafAngleDistribution.from_type("planophile")
    erecto = leaves.LeafAngleDistribution.from_type("erectophile")
    assert plano.mean_deg < 30 < 60 < erecto.mean_deg
    assert plano.chi > 1 > erecto.chi
    assert plano.g(0.0)[0] > 0.8 and erecto.g(0.0)[0] < 0.45
    assert plano.density.sum() == pytest.approx(1.0)


def test_angle_distribution_from_points(rng):
    pts, _ = make_leaves(rng, 400, lambda r, n: r.normal(25, 8, n).clip(0, 90))
    est = leaves.leaf_angle_distribution(pts)
    assert est.mean_deg == pytest.approx(25, abs=4)
    assert est.de_wit == "planophile"
    steep, _ = make_leaves(rng, 400, lambda r, n: r.normal(70, 8, n).clip(0, 90))
    assert leaves.leaf_angle_distribution(steep).mean_deg == pytest.approx(70, abs=4)


def test_point_leaf_area(rng):
    pts, area = make_leaves(rng, 600, lambda r, n: np.degrees(np.arccos(r.uniform(0, 1, n))), spread=1.5)
    grid = leaves.leaf_area_density(pts, voxel_size=0.5)
    assert grid.total_area == pytest.approx(area, rel=0.25)
    z, prof = grid.profile()
    assert prof.sum() == pytest.approx(grid.total_area) and len(z) == grid.density.shape[0]
    assert grid.scaled_to(10.0).total_area == pytest.approx(10.0)


def test_add_leaves(single_tree, rng, tmp_path):
    model = qsm.build_qsm(single_tree, bin_length=0.5)
    foliage, _ = make_leaves(rng, 300, lambda r, n: r.normal(30, 10, n).clip(0, 90), centre=(1.2, 0.0, 4.5), spread=0.6)
    angles = leaves.leaf_angle_distribution(foliage)
    grid = leaves.leaf_area_density(foliage, voxel_size=0.25)
    mesh = leaves.add_leaves(model, grid, angles, leaf_points=foliage, leaf_length=0.06, leaf_width=0.03,
                             max_branch_distance=1.5)
    one = leaves.single_leaf_area(0.06, 0.03)
    assert mesh.leaf_area == pytest.approx(one)
    assert mesh.total_area == pytest.approx(grid.total_area, abs=2 * one)  # met to within a leaf or two
    assert mesh.vertices.shape == (6 * len(mesh), 3) and mesh.faces.shape == (4 * len(mesh), 3)
    assert mesh.faces.max() < len(mesh.vertices)
    # Mesh triangles add up to the stated area and carry the sampled normals.
    a, b, c = (mesh.vertices[mesh.faces[:, k]] for k in range(3))
    tri = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert tri.sum() == pytest.approx(mesh.total_area, rel=1e-6)
    placed = leaves.leaf_angle_distribution(mesh.inclination, inclinations=True)
    assert placed.mean_deg == pytest.approx(angles.mean_deg, abs=3)
    # Leaves sit where the foliage is, and know their branch.
    assert np.abs(mesh.centres - foliage.mean(0)).max() < 1.5
    assert (mesh.cylinder >= 0).mean() > 0.5 and mesh.cylinder.max() < len(model)
    # Same seed, same leaves; a total area can be given instead of a grid.
    again = leaves.add_leaves(model, grid, angles, leaf_points=foliage, leaf_length=0.06, leaf_width=0.03,
                              max_branch_distance=1.5)
    np.testing.assert_array_equal(again.vertices, mesh.vertices)
    fixed = leaves.add_leaves(model, 2.0, "spherical", leaf_points=foliage)
    assert fixed.total_area == pytest.approx(2.0, abs=0.01)
    with pytest.raises(ValueError):
        leaves.add_leaves(model, 2.0)
    leaves.write_tree_obj(tmp_path / "tree.obj", model, mesh)
    text = (tmp_path / "tree.obj").read_text()
    assert "o wood" in text and "o leaves" in text


def test_leaf_shape_and_default_size(tmp_path):
    rng = np.random.default_rng(3)  # session rng is shared: draw from our own
    built_in = leaves.LeafShape()
    assert built_in.area == pytest.approx(leaves.single_leaf_area(0.08, 0.04))
    assert built_in.resized(0.16, 0.08).area == pytest.approx(4 * built_in.area)
    assert built_in.scaled_to(0.01).area == pytest.approx(0.01)
    # A blade given in metres keeps the size it was drawn at, in unit leaf space.
    v = np.array([[0, 0, 0], [0.1, 0.03, 0.01], [0.2, 0, 0], [0.1, -0.03, 0.01]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    custom = leaves.LeafShape.from_mesh(v, f)
    assert (custom.length, custom.width) == pytest.approx((0.2, 0.06))
    np.testing.assert_allclose(custom.vertices.max(0) - custom.vertices.min(0), [1, 1, 0.01 / 0.06])
    assert custom.area == pytest.approx(0.5 * np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0])) * 2)
    (tmp_path / "leaf.obj").write_text("".join(f"v {x} {y} {z}\n" for x, y, z in v) + "f 1 2 3 4\n")
    assert leaves.LeafShape.from_obj(tmp_path / "leaf.obj").area == pytest.approx(custom.area)
    for bad in [dict(vertices=v[:2], faces=f), dict(faces=np.zeros((0, 3), np.uint32)), dict(length=0.0)]:
        with pytest.raises(ValueError):
            leaves.LeafShape(**bad)

    pts = rng.uniform(0, 2, (4000, 3))
    grid = leaves.leaf_area_density(pts, voxel_size=0.25).scaled_to(10.0)
    plain = leaves.add_leaves(None, grid, "spherical", leaf_points=pts)
    shaped = leaves.add_leaves(None, grid, "spherical", leaf_points=pts, shape=custom)
    assert shaped.leaf_area == pytest.approx(custom.area)
    assert shaped.faces.shape == (2 * len(shaped), 3) and shaped.vertices.shape == (4 * len(shaped), 3)
    assert shaped.total_area == pytest.approx(10.0, abs=2 * custom.area)
    assert len(shaped) < len(plain)  # a bigger leaf, so fewer of them
    # The default size applies to calls that ask for none, and can be put back.
    big = leaves.add_leaves(None, grid, "spherical", leaf_points=pts, leaf_length=0.16, leaf_width=0.08)
    was = leaves.set_default_leaf(length=0.16, width=0.08)
    try:
        assert leaves.default_leaf().length == 0.16
        now = leaves.add_leaves(None, grid, "spherical", leaf_points=pts)
        np.testing.assert_array_equal(now.vertices, big.vertices)
    finally:
        leaves.set_default_leaf(was)
    assert (leaves.default_leaf().length, leaves.default_leaf().width) == (0.08, 0.04)


def test_classify_leaf_wood(single_tree):
    rng = np.random.default_rng(0)
    foliage, _ = make_leaves(rng, 500, lambda r, n: np.degrees(np.arccos(r.uniform(0, 1, n))), centre=(1.8, 0.0, 4.8), spread=0.4)
    cloud = PointCloud(np.vstack([single_tree.xyz, foliage]))
    n = len(single_tree)
    wood = leaves.classify_leaf_wood(cloud)  # graph-based
    assert wood.shape == (len(cloud),) and wood.dtype == bool
    assert wood[:n].mean() > 0.9  # the stem and branch
    assert (~wood[n:]).mean() > 0.8  # the foliage
    # The passage filter: over a few centimetres a leaf is as planar as bark, so
    # one anisotropy scale finds almost no leaves and the second scale helps.
    narrow = leaves.classify_leaf_wood(cloud, method="passage", scale_radius=0.0)
    wide = leaves.classify_leaf_wood(cloud, method="passage")
    assert wide[:n].mean() > 0.9
    assert (~narrow[n:]).mean() < (~wide[n:]).mean() - 0.2 < (~wood[n:]).mean()
    with pytest.raises(ValueError):
        leaves.classify_leaf_wood(cloud, method="nope")
    # The QSM input filter can use it too.
    assert 0 < len(qsm.wood_points(cloud, method="gbs")) < len(cloud)
