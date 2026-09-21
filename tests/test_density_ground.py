import numpy as np

from sylva import PointCloud, Raster, Shots, canopy


def _slab_shots(rng, n=20000, ground_z=0.0, scanner_z=1.5):
    """Scanner at (0, 0, scanner_z); upward rays hit a foliage slab at 6-8 m,
    downward rays hit flat ground at ground_z."""
    theta = rng.uniform(np.radians(20), np.radians(160), n)
    phi = rng.uniform(0, 2 * np.pi, n)
    d = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    up = d[:, 2] > 0
    ranges = np.where(up, rng.uniform(6, 8, n) / np.where(up, d[:, 2], 1),
                      (ground_z - scanner_z) / np.where(up, 1, d[:, 2]))
    hit = up & (rng.uniform(size=n) < 0.5) | ~up
    hit &= ranges < 40
    count = hit.astype(np.int64)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    origin = np.tile([0, 0, scanner_z], (n, 1)).astype(float)
    return Shots(origin, d, start, count, ranges[hit])


def test_mask_ground_and_profile(rng):
    shots = _slab_shots(rng)
    dtm = Raster(np.zeros((40, 40)), -20, -20, 1.0)
    g = canopy.density_grid(shots, 1.0, origin=(-20, -20, -2), shape=(40, 40, 14), min_hits=1)
    h = g.height_above(dtm)
    assert h.shape == g.density.shape
    np.testing.assert_allclose(h[0, 0, 0], -1.5)
    # Ground layer has hits in the raw grid but is NaN after masking.
    k_ground = 1  # z in [-1, 0): centre -0.5
    assert np.nanmax(g.density[k_ground + 1]) > 0  # layer just above ground has ground hits
    gm = g.mask_ground(dtm, margin=1.0)
    assert np.all(np.isnan(gm.density[: k_ground + 2]))
    assert gm.pai < g.pai
    for pooled in (True, False):
        z, pad = gm.profile_above_ground(dtm, bin_size=1.0, max_height=12, pooled=pooled)
        assert len(z) == 12
        assert np.nansum(pad[z < 5]) == 0
        assert np.nanmax(pad[(z >= 6) & (z < 8)]) > 0
    # Pooled estimate of a uniform slab: PAD = 2 * hits / path; with 50 % of
    # upward rays absorbed over ~2 m of slab, 2 * 0.5 / 2 ~ 0.5 per metre-ish.
    z, pad = gm.profile_above_ground(dtm, bin_size=2.0, max_height=10, pooled=True)
    assert 0.2 < np.nanmax(pad) < 2.0


def test_pulses_per_line_overshoot():
    pattern = {"theta_start": 30.0, "theta_delta": 1.0, "theta_count": 10,
               "phi_start": 0.0, "phi_delta": 1.0, "phi_count": 100}
    # Scanner actually fired 105 pulses per line on 9 saturated lines, 20 on one.
    per_line = [105] * 9 + [20]
    theta = np.radians(np.concatenate([np.full(n, 30 + i) for i, n in enumerate(per_line)]))
    az = np.linspace(0, 2 * np.pi, theta.size)
    d = np.column_stack([np.sin(theta) * np.sin(az), np.sin(theta) * np.cos(az), np.cos(theta)])
    shots = Shots.from_pointcloud(PointCloud(d * 5))
    assert shots.pulses_per_line(pattern) == 105
    full = shots.fill_missing(pattern)
    assert full.n_shots == 105 * 10
    assert shots.fill_missing(pattern, pulses_per_line=100).n_shots == 9 * 105 + 100
