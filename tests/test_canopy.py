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


def _turbid_layer(rng, g, pad=0.3, z0=5.0, z1=15.0, scans=3, n=200000, clumped=False):
    """Pulses from 1.5 m up through a layer of plant area density ``pad``
    with projection ``g(theta)``; ``clumped`` halves the layer's azimuth
    sectors (dense in half, empty in the other)."""
    from sylva import Shots

    prof = canopy.GapProfile.empty(np.arange(5.0, 75.0, 5.0), n_azimuth=12, height_bin=0.5, max_height=30)
    for _ in range(scans):
        zen = np.degrees(np.arccos(rng.uniform(np.cos(np.radians(72)), np.cos(np.radians(3)), n)))
        az = rng.uniform(0, 2 * np.pi, n)
        th = np.radians(zen)
        d = np.c_[np.sin(th) * np.sin(az), np.sin(th) * np.cos(az), np.cos(th)]
        density = np.full(n, pad)
        if clumped:
            density = np.where(np.sin(6 * az) > 0, 2 * pad, 1e-9)
        free = rng.exponential(1 / (g(th) * density))
        z_hit = z0 + free * d[:, 2]
        hit = z_hit < z1
        count = hit.astype(np.int64)
        s = Shots(np.c_[np.zeros((n, 2)), np.full(n, 1.5)], d, np.r_[0, np.cumsum(count)[:-1]], count,
                  ((z_hit - 1.5) / d[:, 2])[hit])
        prof.add_scan(s, s.echo_xyz()[:, 2])
    return prof


def test_gap_profile_recovers_pai(rng):
    sph = _turbid_layer(rng, lambda t: np.full_like(t, 0.5)).report()
    assert sph["pai_hinge"] == pytest.approx(3.0, rel=0.05)
    assert sph["pai_weighted"] == pytest.approx(3.0, rel=0.05)
    assert sph["pai_linear"] == pytest.approx(3.0, rel=0.1)
    assert sph["clumping"] == pytest.approx(1.0, abs=0.03)
    assert sph["canopy_height"] == pytest.approx(15.0, abs=0.6)
    assert not sph["saturated"] and sph["gap_57"] == pytest.approx(np.exp(-3.0 / 1.1), rel=0.1)
    layer = (sph["height"] > 6) & (sph["height"] < 14)
    np.testing.assert_allclose(sph["pavd_hinge"][layer].mean(), 0.3, rtol=0.1)
    horiz = _turbid_layer(rng, np.cos).report()
    vert = _turbid_layer(rng, lambda t: 2 / np.pi * np.sin(t)).report()
    assert horiz["pai_linear"] == pytest.approx(3.0, rel=0.05) and horiz["mla_linear"] < 10
    assert vert["pai_linear"] == pytest.approx(3.0, rel=0.05) and vert["mla_linear"] > 80


def test_gap_profile_clumping(rng):
    rep = _turbid_layer(rng, lambda t: np.full_like(t, 0.5), clumped=True).report()
    assert rep["clumping"] < 0.8  # half the sectors dense, half open
    assert rep["pai_hinge_corrected"] > rep["pai_hinge"]


def test_fit_ground_plane(rng):
    xy = rng.uniform(-20, 20, (20000, 2))
    z = 0.1 * xy[:, 0] - 0.05 * xy[:, 1] + 3.0 + rng.normal(0, 0.01, len(xy))
    trunks = np.c_[rng.uniform(-20, 20, (500, 2)), rng.uniform(3, 20, 500)]
    coef = canopy.fit_ground_plane(np.vstack([np.c_[xy, z], trunks]))
    np.testing.assert_allclose(coef, [0.1, -0.05, 3.0], atol=0.02)


def test_fired_pulses_from_ground_lines(rng):
    from sylva import Shots

    # Nominal 1 deg zenith lines from 30 to 130 deg and 360 azimuth steps, but
    # the scanner fires 1 % more (364 per line). Downward pulses all hit the
    # ground; upward ones return 40 % of the time and the rest are missing
    # from the stream.
    pattern = dict(theta_start=30.0, theta_delta=1.0, theta_count=101, phi_start=0.0, phi_delta=1.0, phi_count=360)
    theta = np.repeat(30.0 + np.arange(101), 364) + rng.normal(0, 0.05, 101 * 364)
    phi = np.tile(np.linspace(0, 360, 364, endpoint=False), 101)
    kept = (theta > 90) | (rng.uniform(size=theta.size) < 0.4)
    th, ph = np.radians(theta[kept]), np.radians(phi[kept])
    d = np.c_[np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph), np.cos(th)]
    n = len(d)
    s = Shots(np.zeros((n, 3)), d, np.arange(n), np.ones(n, np.int64), np.full(n, 5.0))
    edges = np.arange(30.0, 75.0, 5.0)
    fired = canopy.fired_pulses_per_ring(s, pattern, edges)
    np.testing.assert_allclose(fired, 5 * 364, rtol=0.01)
    zen, _ = s.zenith_azimuth()
    obs, _ = np.histogram(zen, bins=edges)
    np.testing.assert_allclose(1 - obs / fired, 0.6, atol=0.03)
