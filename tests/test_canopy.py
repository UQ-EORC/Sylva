import numpy as np
import pytest

from sylva import PointCloud, Shots, canopy


def test_voxelize(forest):
    grid = canopy.voxelize(forest, 1.0)
    assert grid.counts.sum() == len(forest)
    prof = grid.vertical_profile()
    assert prof.shape == (grid.shape[2],)
    assert prof[0] > prof[-1]
    centers = grid.occupied_centers()
    assert centers.shape[1] == 3
    assert len(centers) == grid.occupied.sum()


def test_pad_profile(rng):
    n = 40000
    xyz = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n), rng.uniform(5, 10, n)])
    pc = PointCloud(xyz, {"height": xyz[:, 2]})
    z, pad = canopy.pad_profile_voxel(pc, voxel_size=0.5)
    assert len(z) == len(pad)
    assert pad[z < 5].sum() == 0
    assert pad[z >= 5].sum() > 0


def test_gap_fraction_and_lai():
    zen = np.arange(2.5, 90, 5)
    pai = canopy.lai_from_gap_fraction(zen, np.full_like(zen, 0.5), method="hinge")
    assert pai == pytest.approx(-np.log(0.5) * np.cos(np.radians(57.5)) / 0.5, rel=1e-6)
    miller = canopy.lai_from_gap_fraction(zen, np.full_like(zen, 0.5), method="miller")
    assert miller == pytest.approx(-np.log(0.5), rel=0.05)


def test_gap_fraction_zenith(rng):
    n = 5000
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    xyz = v * 10
    heights = np.where(rng.uniform(size=n) < 0.3, 5.0, -1.0)  # 30% canopy hits
    shots = Shots.from_pointcloud(PointCloud(xyz))
    assert shots.n_shots == n
    centres, gap = canopy.gap_fraction_zenith(shots, heights, min_height=0.0)
    ok = np.isfinite(gap)
    assert np.nanmean(gap[ok]) == pytest.approx(0.7, abs=0.05)


def test_density_grid_uniform_foliage(rng):
    # Scanner below a slab of foliage: rays going up either hit the slab or escape.
    n = 20000
    theta = rng.uniform(0, np.radians(30), n)
    phi = rng.uniform(0, 2 * np.pi, n)
    d = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    hit = rng.uniform(size=n) < 0.5
    # hits uniformly inside z in [5, 7] along the ray
    z_hit = rng.uniform(5, 7, n)
    ranges = z_hit / d[:, 2]
    echo_count = hit.astype(np.int64)
    echo_start = np.concatenate([[0], np.cumsum(echo_count)[:-1]])
    shots = Shots(np.zeros((n, 3)), d, echo_start, echo_count, ranges[hit])
    grid = canopy.density_grid(shots, voxel_size=1.0, origin=(-6, -6, 0), shape=(12, 12, 9))
    prof = grid.profile
    assert grid.n_rays.shape == (9, 12, 12)
    assert np.nansum(prof[:5]) == 0  # nothing below the slab
    assert np.nanmean(prof[5:7]) > 0
    assert grid.pai > 0
    # Free space above the slab has rays but no hits -> NaN density (below min_hits)
    assert np.all(np.isnan(grid.density[8]))


def test_canopy_cover():
    chm = np.array([[0, 1], [3, np.nan]], dtype=float)
    assert canopy.canopy_cover(chm, 2.0) == pytest.approx(1 / 3)
