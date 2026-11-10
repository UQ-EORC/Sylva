"""Grids of the DTM and CHM: their extent, their size and what they refuse."""
import numpy as np
import pytest

from sylva import PointCloud, ground

# A plot in projected (UTM-scale) coordinates.
OX, OY = 512_000.0, 6_234_000.0


def _plot(n=40_000, seed=3):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    canopy = (x >= 40) & (x < 60) & (y >= 40) & (y < 60)
    h = np.where(canopy, 15.0, 0.1)
    attrs = {"height": h, "classification": np.where(canopy, 1, 2).astype(np.uint8)}
    return PointCloud(np.column_stack([OX + x, OY + y, 300.0 + h]), attrs)


def test_chm_covers_the_cloud_not_only_the_canopy():
    cloud = _plot()
    chm = ground.make_chm(cloud, resolution=1.0, min_height=2.0)
    assert chm.data.shape == (100, 100)
    assert (chm.xmin, chm.ymin) == (OX, OY)
    assert abs((chm.data > 2.0).mean() - 0.04) < 0.005
    # Nothing above the threshold is a raster of zeros over the cloud.
    bare = ground.make_chm(cloud, resolution=1.0, min_height=50.0)
    assert bare.data.shape == (100, 100) and not bare.data.any()


def test_bounds_may_be_a_list_or_an_array_and_size_the_grid_as_documented():
    cloud = _plot()
    b = (OX, OY, OX + 20.0, OY + 20.0)
    for bounds in (b, list(b), np.array(b)):
        dtm = ground.make_dtm(cloud, resolution=0.5, bounds=bounds)
        chm = ground.make_chm(cloud, resolution=0.5, bounds=bounds)
        # floor(extent / resolution) + 1 cells: one more than 20 / 0.5.
        assert dtm.data.shape == chm.data.shape == (41, 41)
        assert (dtm.xmin, dtm.ymin) == (OX, OY)
    exact = ground.make_chm(cloud, resolution=0.5, bounds=[OX, OY, OX + 19.75, OY + 19.75])
    assert exact.data.shape == (40, 40)
    with pytest.raises(ValueError):
        ground.make_chm(cloud, bounds=[OX, OY, OX + 20.0])


def test_sizes_that_cannot_make_a_grid_are_refused():
    cloud = _plot(2000)
    for res in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            ground.make_chm(cloud, resolution=res)
        with pytest.raises(ValueError):
            ground.classify_ground_csf(cloud, cloth_resolution=res)
        with pytest.raises(ValueError):
            ground.classify_ground_pmf(cloud, cell_size=res)
    with pytest.raises(ValueError):
        ground.classify_ground_pmf(cloud, max_window=0.0)
    with pytest.raises(ValueError):
        ground.classify_ground_pmf(cloud, slope=float("nan"))


def test_csf_takes_a_slope_the_same_whichever_way_it_faces():
    rng = np.random.default_rng(11)
    x, y = rng.uniform(0, 60, 120_000), rng.uniform(0, 60, 120_000)
    inner = (x > 8) & (x < 52) & (y > 8) & (y < 52)
    for g in (0.36, -0.36):
        z = 300.0 + g * x + rng.uniform(-0.02, 0.02, x.size)
        cloud = PointCloud(np.column_stack([OX + x, OY + y, z]))
        mask = ground.classify_ground_csf(cloud, cloth_resolution=1.0, return_mask=True)
        assert mask[inner].mean() > 0.99, g


def test_a_point_without_a_position_is_not_ground_and_does_not_stop_the_filter():
    cloud = _plot(5000)
    xyz = cloud.xyz.copy()
    xyz[7] = [np.nan, OY, 300.0]
    xyz[9] = [OX, np.inf, 300.0]
    mask = ground.classify_ground_csf(PointCloud(xyz), return_mask=True)
    assert mask.shape == (5000,) and not mask[7] and not mask[9]
