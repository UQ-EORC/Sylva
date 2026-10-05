# Tests of the coregistration ground model and local geometry (voxel
# downsampling, normals, planarity filter).
import numpy as np
import pytest

from sylva.coreg import estimate_normals, fit_ground, planar_filter, voxel_downsample


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def test_voxel_downsample_reduces_density(rng):
    points = rng.uniform(0, 10, size=(20000, 3))
    reduced = voxel_downsample(points, 1.0)
    assert 900 <= len(reduced) <= 1000
    assert reduced.min() >= 0 and reduced.max() <= 10


def test_voxel_downsample_keeps_one_point_per_voxel(rng):
    points = rng.uniform(0, 4, size=(5000, 3))
    reduced = voxel_downsample(points, 0.5, centroid=False)
    keys = np.floor(reduced / 0.5).astype(int)
    assert len(np.unique(keys, axis=0)) == len(reduced)


def test_voxel_downsample_empty():
    assert voxel_downsample(np.zeros((0, 3)), 0.1).shape == (0, 3)


def test_voxel_downsample_rejects_bad_size(rng):
    with pytest.raises(ValueError):
        voxel_downsample(rng.normal(size=(10, 3)), 0.0)


def test_voxel_downsample_counts_match_occupancy_and_default_is_unaffected(rng):
    # A dense cluster well inside one 1.0 m cell, and one isolated point far away.
    dense = np.array([5.5, 5.5, 5.5]) + rng.normal(0, 0.001, (7, 3))
    stray = np.array([[20.5, 20.5, 20.5]])
    points = np.vstack([dense, stray])

    plain = voxel_downsample(points, 1.0)
    out, counts = voxel_downsample(points, 1.0, return_counts=True)
    assert np.array_equal(out, plain), "asking for counts must not change the downsampled points"
    assert sorted(counts.tolist()) == [1, 7]
    dense_row = out[counts == 7][0]
    stray_row = out[counts == 1][0]
    assert np.allclose(dense_row, dense.mean(axis=0))
    assert np.allclose(stray_row, stray[0])

    plain_first = voxel_downsample(points, 1.0, centroid=False)
    out_first, counts_first = voxel_downsample(points, 1.0, centroid=False, return_counts=True)
    assert np.array_equal(out_first, plain_first), (
        "counts must line up with, not reorder, first-point mode"
    )
    for row, c in zip(out_first, counts_first, strict=True):
        assert c == (1 if np.allclose(row, stray[0]) else 7)

    empty_pts, empty_counts = voxel_downsample(np.zeros((0, 3)), 0.1, return_counts=True)
    assert empty_pts.shape == (0, 3) and empty_counts.shape == (0,)


def test_normals_on_a_plane(rng):
    points = np.column_stack(
        [rng.uniform(0, 5, 3000), rng.uniform(0, 5, 3000), np.zeros(3000)]
    ) + rng.normal(0, 0.002, (3000, 3))
    normals, planarity = estimate_normals(points, k=20)
    assert np.abs(normals[:, 2]).mean() > 0.99
    assert planarity.mean() > 0.4


def test_planar_filter_keeps_stems_and_drops_foliage(rng):
    """The filter must be strongly selective: it is what keeps ICP out of the crown."""
    theta = rng.uniform(0, 2 * np.pi, 6000)
    stem = np.column_stack([0.2 * np.cos(theta), 0.2 * np.sin(theta), rng.uniform(0, 4, 6000)])
    foliage = rng.normal(0, 0.5, (6000, 3))

    def retention(points):
        baseline = len(planar_filter(points, min_planarity=0.0, voxel=0.03))
        return len(planar_filter(points, min_planarity=0.35, voxel=0.03)) / baseline

    stem_retention = retention(stem)
    foliage_retention = retention(foliage)
    assert stem_retention > 0.9
    assert foliage_retention < 0.4
    assert stem_retention > 3 * foliage_retention


def test_ground_fit_accuracy(rng):
    n = 120000
    x = rng.uniform(0, 40, n)
    y = rng.uniform(0, 40, n)
    truth = 0.1 * x + 0.05 * y + 1.5 * np.sin(x / 8)
    ground_points = np.column_stack([x, y, truth + rng.normal(0, 0.02, n)])
    shrubs = ground_points + np.column_stack([np.zeros((n, 2)), rng.uniform(0.3, 1.2, n)])
    canopy = np.column_stack([rng.uniform(0, 40, n), rng.uniform(0, 40, n), rng.uniform(5, 25, n)])
    model = fit_ground(np.vstack([ground_points, shrubs, canopy]), cell_size=0.5)

    query = np.column_stack([rng.uniform(2, 38, 4000), rng.uniform(2, 38, 4000)])
    expected = 0.1 * query[:, 0] + 0.05 * query[:, 1] + 1.5 * np.sin(query[:, 0] / 8)
    error = model.height_at(query) - expected
    assert np.abs(error).mean() < 0.06
    assert abs(error.mean()) < 0.05
    assert model.slope_deg > 1.0


def test_ground_normalise_zeroes_the_terrain(rng):
    x = rng.uniform(0, 20, 40000)
    y = rng.uniform(0, 20, 40000)
    points = np.column_stack([x, y, 0.08 * x])
    heights = fit_ground(points, cell_size=0.5).normalise(points)
    assert np.abs(heights).mean() < 0.05


def test_ground_reports_unobserved_cells(rng):
    points = np.column_stack(
        [rng.uniform(0, 10, 20000), rng.uniform(0, 10, 20000), np.zeros(20000)]
    )
    model = fit_ground(points, cell_size=0.5)
    assert model.support(np.array([[5.0, 5.0]]))[0]


def test_ground_rejects_empty_cloud():
    with pytest.raises(ValueError):
        fit_ground(np.zeros((0, 3)))


def test_ground_ignores_deep_below_ground_ghost_echoes():
    """A few weak echoes far below the terrain must not drag the surface down.

    Seen on tilted VZ-400i scans: 0.02% of points 20 to 30 m under the ground
    left the terrain model 17 m too low and the scan without a single stem.
    """
    rng = np.random.default_rng(0)
    n = 60000
    xy = rng.uniform(-30, 30, (n, 2))
    ground = np.c_[xy, 0.02 * xy[:, 0] + rng.normal(0, 0.02, n)]  # gentle slope
    canopy = np.c_[rng.uniform(-30, 30, (20000, 2)), rng.uniform(2, 15, 20000)]
    ghosts = np.c_[
        rng.uniform(15, 28, (300, 1)) * np.array([[1.0]]),
        rng.uniform(-3, 3, (300, 1)),
        rng.uniform(-30, -18, (300, 1)),
    ]
    cloud = np.vstack([ground, canopy, ghosts])
    model = fit_ground(cloud, 0.5)
    heights = model.normalise(ground)
    assert abs(np.median(heights)) < 0.1, (
        f"ground should sit at height 0, got median {np.median(heights):.2f}"
    )
    assert np.percentile(heights, 99) < 0.5
    assert model.normalise(ghosts).max() < -10, "ghost echoes stay far below the surface"
