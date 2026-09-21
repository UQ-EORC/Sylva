import numpy as np

from sylva import PointCloud, filters


def test_voxel_downsample(forest):
    thin = filters.voxel_downsample(forest, 0.5)
    assert 0 < len(thin) < len(forest)
    assert "intensity" in thin.attrs
    cen = filters.voxel_downsample(forest, 0.5, method="centroid")
    assert len(cen) == len(thin)


def test_random_subsample(forest):
    assert len(filters.random_subsample(forest, n=100, seed=1)) == 100
    assert len(filters.random_subsample(forest, fraction=0.1, seed=1)) == round(len(forest) * 0.1)
    a = filters.random_subsample(forest, n=50, seed=3)
    b = filters.random_subsample(forest, n=50, seed=3)
    np.testing.assert_array_equal(a.xyz, b.xyz)


def test_min_distance(rng):
    pc = PointCloud(rng.uniform(0, 1, (2000, 3)))
    out = filters.min_distance_subsample(pc, 0.1)
    d, _ = filters.knn(out.xyz, out.xyz, 2)
    assert d[:, 1].min() >= 0.1 - 1e-9


def test_crop(forest):
    box = filters.crop_box(forest, (0, 0, None), (10, 10, None))
    assert box.x.max() <= 10 and box.y.max() <= 10
    cyl = filters.crop_cylinder(forest, (5, 5), 3)
    assert np.hypot(cyl.x - 5, cyl.y - 5).max() <= 3
    rf = filters.range_filter(forest, (5, 5, 0), max_range=3)
    assert len(rf) > 0


def test_outlier_removal(rng):
    dense = rng.normal(0, 0.1, (1000, 3))
    far = np.array([[5, 5, 5], [-5, -5, -5]], dtype=float)
    pc = PointCloud(np.vstack([dense, far]))
    sor = filters.statistical_outlier_removal(pc, k=8, std_ratio=2.0)
    assert len(sor) == 1000
    ror = filters.radius_outlier_removal(pc, radius=0.3, min_neighbors=3)
    assert len(ror) == 1000
    mask = filters.radius_outlier_removal(pc, radius=0.3, min_neighbors=3, return_mask=True)
    assert not mask[-2:].any()


def test_normals_and_planarity(rng):
    xy = rng.uniform(0, 5, (3000, 2))
    plane = PointCloud(np.column_stack([xy, 0.3 * xy[:, 0] + rng.normal(0, 0.001, 3000)]))
    n = filters.estimate_normals(plane, k=12)
    expected = np.array([-0.3, 0, 1]) / np.linalg.norm([-0.3, 0, 1])
    assert np.abs(n @ expected).mean() > 0.99
    planarity, linearity = filters.planarity_linearity(plane, k=20)
    line = PointCloud(np.column_stack([np.linspace(0, 5, 3000), np.zeros(3000), np.zeros(3000)])
                      + rng.normal(0, 0.001, (3000, 3)))
    planarity_l, linearity_l = filters.planarity_linearity(line, k=20)
    assert planarity.mean() > linearity.mean()
    assert linearity_l.mean() > 0.9 and planarity_l.mean() < 0.1


def test_clusters(rng):
    a = rng.normal(0, 0.05, (200, 3))
    b = rng.normal(0, 0.05, (100, 3)) + [3, 0, 0]
    c = np.array([[10.0, 10, 10]])
    labels = filters.euclidean_clusters(np.vstack([a, b, c]), 0.3, min_points=5)
    assert set(labels[:200]) == {0}
    assert set(labels[200:300]) == {1}
    assert labels[-1] == -1


def test_knn(rng):
    pts = rng.uniform(0, 1, (500, 3))
    d, i = filters.knn(pts, pts[:10], 3)
    assert d.shape == (10, 3) and i.shape == (10, 3)
    assert np.all(i[:, 0] == np.arange(10))
    assert np.all(d[:, 0] == 0)
