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


def test_fired_pulses_from_points_recovers_the_misses():
    """A regular scan grid whose upper pulses mostly miss: the pulses fired
    per ring come from the returns alone, via the downward lines."""
    from sylva import Shots
    from sylva.canopy import fired_pulses_from_points

    rng = np.random.default_rng(0)
    zen = np.radians(np.arange(30.05, 130.0, 0.1))
    az = np.radians(np.arange(0.0, 360.0, 0.5))
    zz, aa = np.meshgrid(zen, az)
    zz, aa = zz.ravel(), aa.ravel()
    direction = np.column_stack([np.sin(zz) * np.sin(aa), np.sin(zz) * np.cos(aa), np.cos(zz)])
    # Upward pulses miss with probability rising to 0.7 at the top; downward ones all return.
    p_miss = np.clip((90.0 - np.degrees(zz)) / 60.0 * 0.7, 0.0, None)
    returned = rng.uniform(size=len(zz)) >= p_miss
    n = int(returned.sum())
    shots = Shots(np.zeros((n, 3)), direction[returned], np.arange(n), np.ones(n, np.int64), np.full(n, 10.0))
    edges = np.arange(5.0, 80.0, 5.0)
    truth = np.histogram(np.degrees(zz), bins=edges)[0]
    fired = fired_pulses_from_points(shots, edges)
    inside = truth > 0
    assert np.allclose(fired[inside], truth[inside], rtol=0.01)
    assert (fired[~inside] < 0.01 * truth.max()).all()
    # Open sky: nothing above 60 degrees zenith returns, yet those rings were
    # fired into; the field of view, not the returns, bounds the scan above.
    open_sky = returned & (np.degrees(zz) > 60.0)
    m = int(open_sky.sum())
    shots = Shots(np.zeros((m, 3)), direction[open_sky], np.arange(m), np.ones(m, np.int64), np.full(m, 10.0))
    fired = fired_pulses_from_points(shots, edges)
    assert np.allclose(fired[inside], truth[inside], rtol=0.01)


def _dir(zen_deg, az_deg):
    z, a = np.radians(zen_deg), np.radians(az_deg)
    return np.array([np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)])


def _single_echo_shots(directions, hit, ranges=10.0, attrs=None):
    counts = hit.astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_echo = int(counts.sum())
    return Shots(np.zeros((len(directions), 3)), directions, starts, counts,
                 np.full(n_echo, ranges), attrs or {})


def test_hemispherical_photo_from_pulses_reads_the_gap_fraction(rng):
    # Uniform directions over the hemisphere, 30 % hits: every ring reads 0.7.
    n = 200000
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    hit = rng.uniform(size=n) < 0.3
    shots = _single_echo_shots(v, hit, attrs={"reflectance": np.full(hit.sum(), -5.0)})
    photo = canopy.hemispherical_photo(shots, size=200, value="reflectance")
    assert photo.from_pulses and photo.value_name == "reflectance"
    assert photo.shots.sum() == n and photo.hits.sum() == hit.sum() and photo.ground.sum() == 0
    zen, gap = photo.gap_fraction_zenith()
    assert zen.tolist() == list(np.arange(2.5, 90, 5))
    assert np.allclose(gap, 0.7, atol=0.03)
    hinge = -np.log(0.7) * np.cos(np.radians(57.5)) / 0.5
    assert photo.lai("hinge") == pytest.approx(hinge, rel=0.05)
    assert np.nanmean(photo.value) == pytest.approx(-5.0)
    assert np.nanmin(photo.range) == 10.0
    g = photo.gap
    assert np.isnan(g[~photo.mask]).all()
    assert np.nanmean(g) == pytest.approx(0.7, abs=0.02)
    # Pixels are gap-fraction, so the thresholded canopy mask is sparse at 30 % hits.
    assert photo.canopy.mean() < 0.1
    for kind in ("binary", "gap", "range", "value"):
        rgb = photo.to_rgb(kind)
        assert rgb.shape == (200, 200, 3) and rgb.dtype == np.uint8
    assert (photo.to_rgb("binary")[~photo.mask] == 0).all()
    with pytest.raises(ValueError, match="unknown kind"):
        photo.to_rgb("depth")
    # The same rings as gap_fraction_zenith counts directly, up to the pixel
    # quantisation of the ring edges.
    heights = np.full(shots.n_echoes, 5.0)
    _, direct = canopy.gap_fraction_zenith(shots, heights)
    assert np.allclose(gap, direct[: len(gap)], atol=0.02)
    with pytest.raises(ValueError, match="echo_heights has"):
        canopy.hemispherical_photo(shots, size=50, echo_heights=np.zeros(3))


def test_hemispherical_photo_ground_pulses(rng):
    # Half the hits are below min_height: ground pulses, which count as gap,
    # canopy or not at all.
    n = 100000
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    hit = rng.uniform(size=n) < 0.6
    shots = _single_echo_shots(v, hit)
    heights = np.where(rng.uniform(size=hit.sum()) < 0.5, 5.0, 0.5)
    photos = {g: canopy.hemispherical_photo(shots, size=200, echo_heights=heights, min_height=1.0,
                                            ground_as=g) for g in ("gap", "canopy", "unseen")}
    assert photos["gap"].ground.sum() == (heights <= 1.0).sum()
    for g, expect in (("gap", 0.7), ("canopy", 0.4), ("unseen", 0.4 / 0.7)):
        _, gap = photos[g].gap_fraction_zenith()
        assert np.allclose(gap, expect, atol=0.05), g   # fewer pulses per ring when unseen
        assert np.nanmean(photos[g].gap) == pytest.approx(expect, abs=0.03), g
    rgb = photos["gap"].to_rgb("binary")
    shades = set(np.unique(rgb[photos["gap"].mask]).tolist())
    assert 64 in shades and shades <= {0, 64, 128, 255}   # 128: pixels no pulse passed through
    with pytest.raises(ValueError, match="ground_as must be"):
        canopy.hemispherical_photo(shots, size=50, ground_as="sky")


def test_hemispherical_photo_orientation_and_lenses():
    shots = _single_echo_shots(_dir(45.0, 45.0)[None, :], np.array([True]), ranges=5.0)
    # North-east at 45 degrees: top-left as a camera sees the sky, top-right as a map.
    (row, col), = np.argwhere(canopy.hemispherical_photo(shots, size=100).hits)
    assert row < 50 and col < 50
    (row, col), = np.argwhere(canopy.hemispherical_photo(shots, size=100, mirror=False).hits)
    assert row < 50 and col > 50
    # Turning north to the pulse's azimuth puts it straight up the image.
    (row, col), = np.argwhere(canopy.hemispherical_photo(shots, size=100, north=45.0).hits)
    assert row == 25 and col in (49, 50)
    # Equidistant puts 45 degrees half way out; equisolid further, stereographic nearer.
    def r_of(lens):
        (row, col), = np.argwhere(canopy.hemispherical_photo(shots, size=400, lens=lens).hits)
        return np.hypot(row + 0.5 - 200, col + 0.5 - 200) / 200
    assert r_of("equidistant") == pytest.approx(0.5, abs=0.01)
    assert r_of("equisolid") > 0.52 > 0.48 > r_of("stereographic")
    # Pixel angles: zenith zero at the centre, the north azimuth up the middle column.
    photo = canopy.hemispherical_photo(shots, size=101, north=30.0)
    zen, az = photo.zenith_azimuth()
    assert zen[50, 50] == pytest.approx(0.0, abs=1.0)
    assert az[10, 50] == pytest.approx(30.0, abs=1.0)
    assert np.isnan(zen[0, 0])
    with pytest.raises(ValueError, match="unknown lens"):
        canopy.hemispherical_photo(shots, size=50, lens="fish")
    with pytest.raises(ValueError, match="orthographic"):
        canopy.hemispherical_photo(shots, size=50, lens="orthographic", max_zenith=100)
    # Beyond the rim is not in the image; a wider lens shows it.
    low = _single_echo_shots(_dir(120.0, 0.0)[None, :], np.array([True]))
    assert canopy.hemispherical_photo(low, size=50).shots.sum() == 0
    assert canopy.hemispherical_photo(low, size=50, max_zenith=130).shots.sum() == 1


def test_hemispherical_photo_from_points(tmp_path):
    # A disc of points at 10 m above the viewpoint fills the sky to 30 degrees.
    g = np.arange(-6.0, 6.01, 0.05)
    x, y = np.meshgrid(g, g)
    keep = np.hypot(x, y) <= 10 * np.tan(np.radians(30))
    xyz = np.column_stack([x[keep], y[keep], np.full(keep.sum(), 10.0)])
    cloud = PointCloud(xyz, {"height": xyz[:, 2]})
    photo = canopy.hemispherical_photo_from_cloud(cloud, size=300, footprint=1.0, value="height")
    assert not photo.from_pulses and photo.shots.sum() == 0
    zen, gap = photo.gap_fraction_zenith(np.arange(0, 91, 10))
    assert gap[:2].max() < 0.05 and gap[0] == 0.0
    assert gap[4:].min() == 1.0
    assert np.nanmean(photo.value) == pytest.approx(10.0)
    assert set(np.unique(photo.gap[photo.mask])) == {0.0, 1.0}
    # Filters by height and range; a bad attribute name is an error.
    assert canopy.hemispherical_photo_from_cloud(cloud, size=50, min_height=20.0).hits.sum() == 0
    assert canopy.hemispherical_photo_from_cloud(cloud, size=50, max_range=5.0).hits.sum() == 0
    # A spacing paints each point as wide as it subtends: 0.5 m at 10 m is
    # about 2.9 degrees, which closes the 0.05 m sampling gaps completely.
    spaced = canopy.hemispherical_photo_from_cloud(cloud, size=300, spacing=0.5)
    assert spaced.gap_fraction_zenith(np.arange(0, 31, 10))[1].max() == 0.0
    with pytest.raises(ValueError, match="spacing"):
        canopy.hemispherical_photo_from_cloud(cloud, size=50, spacing=-1.0)
    with pytest.raises(ValueError, match="no attribute"):
        canopy.hemispherical_photo_from_cloud(cloud, size=50, value="colour")
    with pytest.raises(ValueError, match="without a value"):
        canopy.hemispherical_photo_from_cloud(cloud, size=50).to_rgb("value")
    # A PNG of the right size comes out.
    out = photo.save(tmp_path / "hemi.png", kind="range")
    head = out.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = np.frombuffer(head[16:24], dtype=">u4")
    assert (width, height) == (300, 300)

