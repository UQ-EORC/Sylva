# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Ray-based canopy structure from airborne lidar: trajectories, pulses,
gap-fraction profiles and ray-traced voxels, checked against the geometry of
the synthetic scanner and against a turbid layer of known plant area."""

import struct
import warnings

import numpy as np
import pytest

from sylva import PointCloud, Raster, Shots, als, canopy, coords, synthetic, voxels

# A homogeneous turbid layer: spheres of radius R with centres uniform in a
# slab. A thin beam is stopped with probability n * pi * R^2 per metre in any
# direction, the extinction of a turbid medium of spherical leaf angles
# (G = 0.5) with plant area density PAD = 2 n pi R^2.
R = 0.05
PAD = 0.3
Z1, Z2 = 5.0, 15.0
AREA = (0.0, 0.0, 40.0, 40.0)


def _slab(pad=PAD, lo=-15.0, hi=55.0, seed=1):
    rng = np.random.default_rng(seed)
    n = rng.poisson(pad / 2 / (np.pi * R * R) * (hi - lo) ** 2 * (Z2 - Z1))
    xyz = np.column_stack([rng.uniform(lo, hi, n), rng.uniform(lo, hi, n), rng.uniform(Z1, Z2, n)])
    return PointCloud(xyz, {"classification": np.full(n, 4, np.uint8)})


@pytest.fixture(scope="module")
def slab():
    return _slab()


def _fly(scene, footprint=1, **kw):
    args = dict(altitude=60.0, speed=10.0, line_spacing=20.0, pulse_rate=50_000, scan_angle=30.0,
                footprint_samples=footprint, target_radius=R, terrain_slope=0.0, bounds=AREA,
                range_noise=0.0, clip=False, seed=2)
    args.update(kw)
    return synthetic.als_flight(scene, **args)


@pytest.fixture(scope="module")
def slab_flight(slab):
    return _fly(slab)


@pytest.fixture(scope="module")
def slab_flight7(slab):
    return _fly(slab, footprint=7)


@pytest.fixture(scope="module")
def forest_flight():
    scene = synthetic.forest(ground_points=100)
    return synthetic.als_flight(scene, altitude=70.0, pulse_rate=20_000, line_spacing=12.0,
                                attitude=(2.0, 1.0, 1.5), seed=1)


def _interior():
    mask = np.zeros((5, 5), bool)
    mask[1:3, 1:3] = True
    return mask


# ---------------------------------------------------------------- trajectory


def test_trajectory_sorts_and_checks():
    t = als.Trajectory([2.0, 0.0, 1.0, 2.0], [2, 0, 1, 9], [0, 0, 0, 0], [5, 5, 5, 5])
    assert list(t.time) == [0.0, 1.0, 2.0] and list(t.x) == [0.0, 1.0, 2.0]
    assert len(t) == 3 and not t.has_attitude
    np.testing.assert_allclose(t.positions([0.5, 5.0])[0], [0.5, 0, 5])
    assert np.isnan(t.positions([5.0])).all()
    with pytest.raises(ValueError, match="two samples"):
        als.Trajectory([0.0], [0], [0], [0])
    with pytest.raises(ValueError, match="values for"):
        als.Trajectory([0.0, 1.0], [0, 1, 2], [0, 0], [0, 0])
    with pytest.raises(ValueError, match="finite"):
        als.Trajectory([0.0, np.nan], [0, 1], [0, 0], [0, 0])
    with pytest.raises(ValueError, match="no roll"):
        t.attitude([0.5])


def test_trajectory_gaps_are_not_bridged():
    t = als.Trajectory([0, 1, 2, 3, 23, 24], [0, 1, 2, 3, 23, 24], np.zeros(6), np.zeros(6))
    p = t.positions([1.5, 5.0, 23.5])
    assert np.isfinite(p[0]).all() and np.isnan(p[1]).all() and np.isfinite(p[2]).all()
    assert np.isfinite(t.positions([5.0], max_gap=25.0)).all()
    assert np.isnan(t.positions([1.5], max_gap=0.5)).all()


def test_positions_match_the_synthetic_scanner(forest_flight):
    traj = als.Trajectory.from_dict(forest_flight.trajectory)
    t = forest_flight.points.attrs["gps_time"]
    np.testing.assert_allclose(traj.positions(t), forest_flight.sensor_positions(t), atol=1e-9)
    att = traj.attitude(t[:1000])
    tt = forest_flight.trajectory
    k = np.searchsorted(tt["time"], t[:1000], side="right") - 1
    f = (t[:1000] - tt["time"][k]) / (tt["time"][k + 1] - tt["time"][k])
    roll = (1 - f) * tt["roll"][k] + f * tt["roll"][k + 1]
    np.testing.assert_allclose(att[:, 0], roll, atol=1e-9)


def test_heading_is_interpolated_across_north():
    t = als.Trajectory([0, 1], [0, 1], [0, 0], [0, 0], roll=[0, 0], pitch=[0, 0], heading=[350, 10])
    assert t.attitude([0.5])[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert t.attitude([0.25])[0, 2] == pytest.approx(355.0)


def test_text_trajectories(tmp_path, forest_flight):
    tt = forest_flight.trajectory
    p = tmp_path / "traj.csv"
    with open(p, "w") as f:
        f.write("# exported\nGPS_Time,Easting,Northing,Height,Roll,Pitch,Yaw,quality\n")
        for i in range(len(tt["time"])):
            f.write(f"{tt['time'][i]:.6f},{tt['x'][i]:.4f},{tt['y'][i]:.4f},{tt['z'][i]:.4f},"
                    f"{tt['roll'][i]:.5f},{tt['pitch'][i]:.5f},{tt['heading'][i]:.5f},1\n")
    t = als.read_trajectory(p)
    assert t.has_attitude and len(t) == len(tt["time"])
    np.testing.assert_allclose(t.x, tt["x"], atol=1e-4)
    q = tmp_path / "traj.txt"
    np.savetxt(q, np.column_stack([tt["time"], tt["x"], tt["y"], tt["z"]]))
    with pytest.raises(ValueError, match="time, x, y and z"):
        als.read_trajectory(q)
    t = als.read_trajectory(q, columns=["t", "x", "y", "z"], z_offset=-2.0)
    assert not t.has_attitude
    np.testing.assert_allclose(t.z, tt["z"] - 2.0)
    with pytest.raises(ValueError, match="columns"):
        als.read_trajectory(q, columns=["t", "x"])


def _write_sbet(path, time, lat, lon, h, roll, pitch, heading, wander):
    """SBET records written independently: 17 little-endian doubles, radians."""
    with open(path, "wb") as f:
        for i in range(len(time)):
            rec = [time[i], np.radians(lat[i]), np.radians(lon[i]), h[i], 0, 0, 0,
                   np.radians(roll[i]), np.radians(pitch[i]), np.radians(heading[i] + wander[i]),
                   np.radians(wander[i]), 0, 0, 0, 0, 0, 0]
            f.write(struct.pack("<17d", *rec))


def test_sbet(tmp_path, forest_flight):
    tt = forest_flight.trajectory
    # The synthetic flight placed in UTM zone 55 south, then written as latitude / longitude.
    e, n = tt["x"] + 500_000.0, tt["y"] + 7_000_000.0
    lonlat = coords.reproject(np.column_stack([e, n, tt["z"]]), "EPSG:4326", "EPSG:32755")
    wander = np.linspace(-20, 20, len(e))
    p = tmp_path / "flight.out"
    _write_sbet(p, tt["time"], lonlat[:, 1], lonlat[:, 0], tt["z"], tt["roll"], tt["pitch"],
                tt["heading"], wander)
    t = als.read_trajectory(p, crs="EPSG:32755")
    np.testing.assert_allclose(t.x, e, atol=1e-3)
    np.testing.assert_allclose(t.y, n, atol=1e-3)
    np.testing.assert_allclose(t.z, tt["z"], atol=1e-6)
    np.testing.assert_allclose(t.pitch, tt["pitch"], atol=1e-9)
    d = (t.heading - tt["heading"] + 180) % 360 - 180
    assert np.abs(d).max() < 1e-9
    with pytest.raises(ValueError, match="crs"):
        als.read_trajectory(p)
    with pytest.raises(ValueError, match="geographic"):
        als.read_trajectory(p, crs="EPSG:4326")
    bad = tmp_path / "bad.out"
    bad.write_bytes(b"\0" * 100)
    with pytest.raises(OSError, match="SBET"):
        als.read_trajectory(bad, crs="EPSG:32755")


def test_week_seconds():
    assert als.week_seconds(0.0) == pytest.approx(1e9 % 604800)
    np.testing.assert_allclose(als.week_seconds(np.array([604800.0 - 1e9 + 5])), [5.0])


# -------------------------------------------------------------------- pulses


def test_pulses_rebuild_every_return(forest_flight):
    pts, traj = forest_flight.points, forest_flight.trajectory
    shots, rep = als.pulses(pts, traj, report=True)
    t, line = pts.attrs["gps_time"], pts.attrs["point_source_id"]
    assert shots.n_shots == len(np.unique(np.column_stack([line, t]), axis=0))
    assert shots.n_echoes == len(pts) == rep["n_returns"]
    assert rep["n_incomplete"] == 0 and rep["n_filled"] == 0 and rep["n_unpositioned"] == 0
    assert rep["line_offset_median"] < 1e-6
    # Echoes land back on the points, and pulses leave the sensor.
    a = np.round(shots.echo_xyz(), 6)
    b = np.round(pts.xyz, 6)
    np.testing.assert_array_equal(a[np.lexsort(a.T)], b[np.lexsort(b.T)])
    first = shots.echo_start
    sensor = forest_flight.sensor_positions(shots.echo_attrs["gps_time"][first])
    np.testing.assert_allclose(shots.origin, sensor, atol=1e-9)
    # Ranges ascend within a pulse.
    multi = np.flatnonzero(shots.echo_count > 1)
    assert np.all(shots.echo_range[first[multi] + 1] > shots.echo_range[first[multi]])
    assert rep["pulse_interval"] == pytest.approx(1 / 20_000, rel=1e-6)


def test_missing_returns_and_pulses(slab_flight7):
    pts, traj = slab_flight7.points, slab_flight7.trajectory
    t = pts.attrs["gps_time"]
    rn, nr = pts.attrs["return_number"], pts.attrs["number_of_returns"]
    # Lose the last return of some multi-return pulses, and every return of some pulses.
    rng = np.random.default_rng(0)
    times = np.unique(t)
    gone = rng.choice(times[10:-10], 300, replace=False)
    lose_last = (nr > 1) & (rn == nr) & (rng.random(len(t)) < 0.2)
    keep = ~np.isin(t, gone) & ~lose_last
    cut = pts[keep]
    shots, rep = als.pulses(cut, traj, fill_missing=True, report=True)
    assert rep["n_incomplete"] == int(lose_last[~np.isin(t, gone)].sum())
    assert rep["n_missing_returns"] == rep["n_incomplete"]
    # Isolated holes are filled; the inferred pulses point where the lost ones did.
    assert 0.9 * len(gone) <= rep["n_filled"] <= len(gone)
    # The inferred pulses start where the lost ones did (the platform moves 0.2 mm
    # between pulses) and point the same way, to within the mirror's step.
    empty = shots.echo_count == 0
    full = als.pulses(pts, traj)
    lost = np.isin(full.echo_attrs["gps_time"][full.echo_start], gone)
    d = np.linalg.norm(shots.origin[empty][:, None, :] - full.origin[lost][None, :, :], axis=2)
    j = np.argmin(d, axis=1)
    assert d[np.arange(len(j)), j].max() < 1e-6
    cosang = np.sum(full.direction[lost][j] * shots.direction[empty], axis=1)
    assert np.degrees(np.arccos(np.clip(cosang, -1, 1))).max() < 0.2
    dropped = als.pulses(cut, traj, drop_incomplete=True)
    assert dropped.n_shots == shots.n_shots - rep["n_filled"] - rep["n_incomplete"]


def test_pulses_errors(forest_flight):
    pts, traj = forest_flight.points, forest_flight.trajectory
    with pytest.raises(ValueError, match="gps_time"):
        als.pulses(pts.without("gps_time"), traj)
    shifted = PointCloud(pts.xyz, dict(pts.attrs, gps_time=pts.attrs["gps_time"] + 1e6))
    with pytest.raises(ValueError, match="trajectory"):
        als.pulses(shifted, traj)
    assert als.pulses(shifted, traj, time_offset=-1e6).n_shots > 0
    # Adjusted standard time against seconds of the week: the offset is suggested.
    tw = als.Trajectory.from_dict(traj)
    week = als.Trajectory(tw.time + 300_000.0, tw.x, tw.y, tw.z)
    adjusted = pts.attrs["gps_time"] + 300_000.0 - 1e9 + 604800 * 2000
    adj = PointCloud(pts.xyz, dict(pts.attrs, gps_time=adjusted))
    with pytest.raises(ValueError, match="time_offset = "):
        als.pulses(adj, week)
    with pytest.raises(ValueError):
        als.pulses(pts, traj, max_fill=-1)
    with pytest.raises(ValueError):
        als.pulses("points.laz", traj)
    empty = als.pulses(pts[np.zeros(len(pts), bool)], traj)
    assert empty.n_shots == 0


def test_pulses_trace_like_tls_shots(slab_flight):
    shots = als.pulses(slab_flight.points, slab_flight.trajectory)
    assert isinstance(shots, Shots)
    g = voxels.ray_voxelize(shots, 1.0, ((0, 0, -1), (40, 40, 20)), ground_class=2)
    pad = g.profile("pad_fpl", min_beams=5)
    z = g.z_levels() + 0.5
    inside = (z > Z1 + 1) & (z < Z2 - 1)
    assert np.nanmean(pad[inside]) == pytest.approx(PAD, rel=0.03)


# ---------------------------------------------------------------- estimation


def test_estimated_trajectory(forest_flight):
    truth = als.Trajectory.from_dict(forest_flight.trajectory)
    # Exact returns lie on their beams, so the lines meet at the sensor.
    est = als.estimate_trajectory(forest_flight.points, interval=0.5)
    assert est.rms is not None and est.heading is None
    err = np.linalg.norm(est.xyz - truth.positions(est.time), axis=1)
    assert err.max() < 1e-6
    # With 2 cm of noise across the beams the positions are off by decimetres,
    # the beam directions by hundredths of a degree.
    rng = np.random.default_rng(0)
    pts = forest_flight.points
    noisy = PointCloud(pts.xyz + rng.normal(0, 0.02, pts.xyz.shape), pts.attrs)
    est = als.estimate_trajectory(noisy, interval=0.5)
    err = np.linalg.norm(est.xyz - truth.positions(est.time), axis=1)
    assert np.median(err) < 0.5 and err.max() < 2.0, (np.median(err), err.max())
    shots, rep = als.pulses(noisy, est, report=True)
    exact = als.pulses(noisy, truth)
    assert rep["n_unpositioned"] == 0 and shots.n_shots == exact.n_shots
    cosang = np.sum(shots.direction * exact.direction, axis=1)
    assert np.median(np.degrees(np.arccos(np.clip(cosang, -1, 1)))) < 0.05
    with pytest.raises(ValueError, match="interval"):
        als.estimate_trajectory(pts, interval=0)
    with pytest.raises(ValueError, match="extend"):
        als.estimate_trajectory(pts, extend=-1.0)


def test_estimation_needs_multiple_returns(slab_flight):
    with pytest.raises(ValueError, match="cannot be estimated"):
        als.estimate_trajectory(slab_flight.points)


# ----------------------------------------------------------------- profiles


def test_profile_recovers_a_turbid_layer(slab_flight):
    prof = als.gap_profile(slab_flight.points, slab_flight.trajectory, resolution=10.0, bounds=AREA,
                           dtm=None, min_height=2.0, max_height=18.0)
    h, pad = prof.profile(_interior())
    inside = (h >= Z1 + 1) & (h < Z2 - 1)
    assert np.mean(pad[inside]) == pytest.approx(PAD, rel=0.02)
    assert np.all(pad[(h < Z1 - 1) | (h > Z2 + 1)] < 0.01)
    assert prof.pooled_pai(_interior()) == pytest.approx(PAD * (Z2 - Z1), rel=0.02)
    assert prof.pad().shape == prof.shape
    pai = prof.pai()
    assert pai.shape == (5, 5) and np.nanmedian(pai.data[1:4, 1:4]) == pytest.approx(3.0, rel=0.05)
    cover = prof.cover()
    assert np.all((cover.data[1:4, 1:4] > 0.7) & (cover.data[1:4, 1:4] < 0.85))
    hg, pg = prof.pgap(_interior())
    assert pg[0] == pytest.approx(np.exp(-0.5 * PAD * 10 * 1.04), rel=0.1) and pg[-1] == 1.0


def test_scan_angle_correction(slab_flight):
    pts, traj = slab_flight.points, slab_flight.trajectory
    kw = dict(resolution=10.0, bounds=AREA, dtm=None, min_height=2.0, max_height=18.0)
    corrected = als.gap_profile(pts, traj, **kw).pooled_pai(_interior())
    naive = als.gap_profile(pts, traj, angles="none", **kw).pooled_pai(_interior())
    las = als.gap_profile(pts, angles="scan_angle", **kw).pooled_pai(_interior())
    assert corrected == pytest.approx(3.0, rel=0.02)
    assert naive > 1.04 * corrected
    assert las == pytest.approx(3.0, rel=0.06)


def test_lidr_lad_definition():
    """weighting='all', angles='none', g=0.5 is lidR's LAD(z, dz, k, z0), whose
    definition (lidR 4.x, after Bouvier et al. 2015) is reproduced here."""
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.uniform(0, 0.5, 400), 20 - rng.exponential(4.0, 1600)])
    z = z[z > 0]

    def lidr_lad(z, dz=1.0, k=0.5, z0=2.0):
        lo = np.floor((z.min() - z0) / dz) * dz + z0
        hi = np.ceil((z.max() - z0) / dz) * dz + z0
        bk = np.arange(lo, hi + dz / 2, dz)
        counts, _ = np.histogram(z, bk)
        mids = (bk[:-1] + bk[1:]) / 2
        cs = np.cumsum(counts)
        with np.errstate(divide="ignore", invalid="ignore"):
            gf = np.concatenate([[np.nan], cs[:-1]]) / cs
            lad = -np.log(gf) / (k * dz)
        keep = mids > z0
        return mids[keep], lad[keep]

    xyz = np.column_stack([np.full(len(z), 0.5), np.full(len(z), 0.5), z])
    prof = als.gap_profile(PointCloud(xyz), resolution=1.0, bin_size=1.0, min_height=2.0,
                           weighting="all", angles="none", g=0.5, dtm=None)
    h, pad = prof.profile()
    mids, want = lidr_lad(z)
    n = min(len(h), len(mids))
    np.testing.assert_allclose(h[:n] + 0.5, mids[:n])
    ok = np.isfinite(want[:n])
    np.testing.assert_allclose(pad[:n][ok], want[:n][ok], rtol=1e-12)


def test_weightings_on_multi_return_pulses(slab_flight7):
    pts, traj = slab_flight7.points, slab_flight7.trajectory
    kw = dict(resolution=10.0, bounds=AREA, dtm=None, min_height=2.0, max_height=18.0)
    equal = als.gap_profile(pts, traj, weighting="equal", **kw).pooled_pai(_interior())
    first = als.gap_profile(pts, traj, weighting="first", **kw).pooled_pai(_interior())
    assert equal == pytest.approx(3.0, rel=0.03)
    # A first return fires on any part of the footprint: first-return profiles
    # read the footprint, not the leaves.
    assert first > 1.5 * equal


@pytest.fixture(scope="module")
def slab_profile(slab_flight):
    return als.gap_profile(slab_flight.points, slab_flight.trajectory, resolution=10.0, bounds=AREA,
                           dtm=None, min_height=2.0, max_height=18.0)


def test_profile_metrics_recover_a_turbid_layer(slab_profile):
    """A layer of PAD 0.3 from 5 to 15 m: uniform density in its two
    strata, its centre at 10 m, the spread of a uniform 10 m layer and the
    diversity of ten equal layers."""
    m = slab_profile.area_metrics(_interior())
    assert m["pavd_5_10"] == pytest.approx(PAD, rel=0.03)
    assert m["pavd_10_15"] == pytest.approx(PAD, rel=0.03)
    assert m["pavd_0_5"] < 0.01 and m["pavd_15_20"] < 0.01
    assert m["pai"] == pytest.approx(PAD * (Z2 - Z1), rel=0.02)
    assert m["pai_above_10"] == pytest.approx(PAD * (Z2 - 10.0), rel=0.03)
    assert m["pai_above_15"] < 0.05
    assert m["height_pad_mean"] == pytest.approx((Z1 + Z2) / 2, abs=0.2)
    assert m["height_pad_sd"] == pytest.approx((Z2 - Z1) / np.sqrt(12), rel=0.03)
    assert m["fhd"] == pytest.approx(np.log(Z2 - Z1), rel=0.02)
    assert Z1 <= m["height_pad_max"] <= Z2
    assert m["cover"] == pytest.approx(1 - slab_profile.pgap(_interior())[1][0])
    assert m["cover_above_2"] == m["cover"] and m["cover_above_15"] < 0.02
    assert 0.5 < m["cover_above_10"] < m["cover"]


def test_profile_metrics_per_cell_match_the_pooled_cell(slab_profile):
    rasters = slab_profile.metrics()
    assert list(rasters)[:4] == ["pulses", "pai", "cover", "fhd"]
    assert rasters["pavd_5_10"].shape == (5, 5)
    np.testing.assert_allclose(rasters["pai"].data, slab_profile.pai().data, equal_nan=True)
    one = np.zeros((5, 5), bool)
    one[2, 3] = True
    cell = slab_profile.area_metrics(one)
    for name, r in rasters.items():
        assert r.data[2, 3] == pytest.approx(cell[name], nan_ok=True), name
    # Wider strata: fewer, coarser columns; thinner than a layer is refused.
    assert "pavd_0_10" in slab_profile.metrics(strata=10.0)
    with pytest.raises(ValueError, match="strata"):
        slab_profile.metrics(strata=0.5)


def test_plot_metrics_pool_the_cells_of_each_plot(slab_profile):
    centres = np.array([[20.0, 20.0], [100.0, 100.0]])   # the second is off the grid
    table = slab_profile.plot_metrics(centres, radius=10.0, ids=["a", "b"])
    assert len(table) == 2 and list(table["id"]) == ["a", "b"]
    # Cell centres at 15 and 25 m lie within 10 m of (20, 20): four cells.
    assert table["area"][0] == 400.0
    mask = np.zeros((5, 5), bool)
    mask[1:3, 1:3] = True
    pooled = slab_profile.area_metrics(mask)
    assert table["pavd_5_10"][0] == pytest.approx(pooled["pavd_5_10"])
    assert table["area"][1] == 0.0 and table["pulses"][1] == 0.0 and np.isnan(table["pai"][1])
    # The same plot as a polygon.
    square = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
    by_polygon = slab_profile.plot_metrics([square])
    assert by_polygon["pai"][0] == pytest.approx(table["pai"][0])
    with pytest.raises(ValueError, match="ids"):
        slab_profile.plot_metrics(centres, radius=10.0, ids=["a"])


def test_profile_errors(slab_flight):
    pts, traj = slab_flight.points, slab_flight.trajectory
    with pytest.raises(ValueError, match="number_of_returns"):
        als.gap_profile(pts.without("number_of_returns"), traj, dtm=None)
    with pytest.raises(ValueError, match="angles"):
        als.gap_profile(pts, angles="trajectory", dtm=None)
    with pytest.raises(ValueError, match="angles"):
        als.gap_profile(pts, angles="sideways", dtm=None)
    with pytest.raises(ValueError, match="resolution"):
        als.gap_profile(pts, traj, resolution=0, dtm=None)
    with pytest.raises(ValueError, match="weighting"):
        als.gap_profile(pts, traj, weighting="loud", dtm=None)
    with pytest.raises(ValueError, match="anchor"):
        als.gap_profile(pts, traj, anchor="sky", dtm=None)
    with pytest.raises(ValueError, match="max_height"):
        als.gap_profile(pts, traj, max_height=0.5, dtm=None)
    with pytest.raises(ValueError, match="ground points"):
        als.gap_profile(pts.without("classification"), traj)
    with pytest.raises(ValueError, match="empty"):
        als.gap_profile(pts[np.zeros(len(pts), bool)], traj, dtm=None)
    with pytest.raises(ValueError, match="mask"):
        prof = als.gap_profile(pts, traj, dtm=None, bounds=AREA, resolution=10.0)
        prof.profile(np.ones((2, 2), bool))


def test_profile_with_a_dtm(forest_flight):
    pts = forest_flight.points
    dtm = Raster(np.zeros((1, 1)), -100, -100, 400.0)
    centres = np.arange(-5, 30) + 0.5
    truth = Raster(synthetic.terrain_height(*np.meshgrid(centres, centres)), -5.0, -5.0, 1.0)
    a = als.gap_profile(pts, forest_flight.trajectory, dtm=truth, resolution=5.0)
    b = als.gap_profile(pts, forest_flight.trajectory, dtm="auto", resolution=5.0)
    c = als.gap_profile(pts, forest_flight.trajectory, dtm=dtm, resolution=5.0)
    assert a.shape[1:] == b.shape[1:] == c.shape[1:]
    np.testing.assert_allclose(a.pooled_pai(), b.pooled_pai(), rtol=0.05)


# --------------------------------------------------------------- catalogues


@pytest.fixture(scope="module")
def slab_tiles(tmp_path_factory, slab_flight):
    d = tmp_path_factory.mktemp("slab_tiles")
    return slab_flight.write_tiles(d, size=20.0, format="las")


def test_catalogue_profile_does_not_depend_on_chunks(slab_tiles, slab_flight):
    traj = slab_flight.trajectory
    kw = dict(resolution=5.0, dtm=None, min_height=2.0, max_height=18.0)
    runs = [als.gap_profile(slab_tiles, traj, buffer=12.0, workers=1, **kw),
            als.gap_profile(slab_tiles, traj, chunk_size=30.0, buffer=12.0, workers=3, **kw),
            als.gap_profile(slab_tiles, traj, chunk_size=45.0, buffer=12.0, workers=2, **kw)]
    for r in runs[1:]:
        np.testing.assert_allclose(r.weight, runs[0].weight, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(r.weight_k, runs[0].weight_k, rtol=1e-12, atol=1e-9)
    # The same counts as the whole cloud on the catalogue grid.
    cloud = slab_tiles.read()
    b = slab_tiles.bounds
    whole = als.gap_profile(cloud, traj, bounds=(b[0], b[1], b[3], b[4]), **kw)
    np.testing.assert_allclose(whole.weight, runs[0].weight, rtol=1e-12, atol=1e-9)
    assert runs[0].xmin == whole.xmin and runs[0].ymin == whole.ymin


def _same_voxels(a, b, name):
    """Equal to float32 rounding, except where an echo sits on a voxel face: the
    tiles quantise to 1 mm and the lattice is on whole metres, so about one echo
    in a thousand lies on a face and rounding puts it on either side depending on
    the corner of the chunk's grid. Totals are unchanged."""
    assert np.array_equal(np.isnan(a), np.isnan(b)), name
    ok = ~np.isnan(a)
    close = np.isclose(a[ok], b[ok], rtol=2e-5, atol=1e-6)
    assert close.mean() > 0.99, (name, int((~close).sum()))
    assert np.sum(a[ok]) == pytest.approx(np.sum(b[ok]), rel=2e-4), name


def test_catalogue_voxels_do_not_depend_on_chunks_or_threads(slab_tiles, slab_flight):
    traj = slab_flight.trajectory
    kw = dict(voxel_size=2.0, z_range=(-1.0, 19.0),
              fields=["pad_fpl", "num_hits_weighted", "path_length"])
    ref = als.ray_voxelize(slab_tiles, traj, buffer=20.0, workers=1, **kw)
    assert ref.reach < 20.0
    for other in (als.ray_voxelize(slab_tiles, traj, chunk_size=30.0, buffer=20.0, workers=3, **kw),
                  als.ray_voxelize(slab_tiles, traj, chunk_size=45.0, buffer=20.0, workers=2,
                                   **kw)):
        for name in ("pad_fpl", "num_hits_weighted", "path_length", "num_beams"):
            _same_voxels(other[name], ref[name], name)
    # A cloud on the same lattice gives the same voxels.
    cloud = slab_tiles.read()
    o = ref.origin
    nz, ny, nx = ref.shape
    box = (o[0], o[1], o[2], o[0] + 2 * nx - 1, o[1] + 2 * ny - 1, o[2] + 2 * nz - 1)
    whole = als.ray_voxelize(cloud, traj, bounds=box,
                             **{k: v for k, v in kw.items() if k != "z_range"})
    assert whole.shape == ref.shape
    inner = np.isfinite(ref["pad_fpl"])
    _same_voxels(whole["num_beams"][inner], ref["num_beams"][inner], "num_beams")
    _same_voxels(whole["pad_fpl"][inner], ref["pad_fpl"][inner], "pad_fpl")
    with pytest.warns(UserWarning, match="buffer"):
        als.ray_voxelize(slab_tiles, traj, buffer=1.0, **kw)


def test_catalogue_voxels_recover_the_layer(slab_tiles, slab_flight):
    v = als.ray_voxelize(slab_tiles, slab_flight.trajectory, voxel_size=1.0, z_range=(-1.0, 19.0),
                         buffer=20.0)
    assert set(v.fields) >= {"pad_fpl", "num_beams", "transmittance"}
    x = v.origin[0] + np.arange(v.shape[2]) + 0.5
    y = v.origin[1] + np.arange(v.shape[1]) + 0.5
    area = ((y >= 0) & (y < 40))[:, None] & ((x >= 0) & (x < 40))[None, :]
    h, pad = v.profile(min_beams=5, mask=area)
    zc = v.origin[2] + h + 0.5
    inside = (zc > Z1 + 1) & (zc < Z2 - 1)
    assert np.nanmean(pad[inside]) == pytest.approx(PAD, rel=0.03)
    pai = v.pai(min_beams=5)
    assert np.nanmedian(pai.data[area]) == pytest.approx(PAD * (Z2 - Z1), rel=0.03)
    flat = Raster(np.zeros((1, 1)), -100, -100, 400.0)
    assert np.nanmedian(v.pai(dtm=flat, min_height=Z2 + 1).data[area]) < 0.05
    with pytest.raises(KeyError):
        v.profile("nope")


def test_catalogue_trajectory_estimate(tmp_path, forest_flight):
    cat = forest_flight.write_tiles(tmp_path, size=15.0, format="las")
    a = als.estimate_trajectory(cat, interval=1.0)
    b = als.estimate_trajectory(forest_flight.points, interval=1.0)
    truth = als.Trajectory.from_dict(forest_flight.trajectory)
    for est in (a, b):
        err = np.linalg.norm(est.xyz - truth.positions(est.time), axis=1)
        assert np.median(err) < 0.3


# ------------------------------------------------------- against the TLS path


def _tls_scan(scene, origin, step_deg=1.5, max_zenith=60.0):
    """An exact upward scan of the sphere scene: each pulse stops at the first sphere
    whose centre lies within R of it (brute force over a k-d tree)."""
    spatial = pytest.importorskip("scipy.spatial")
    tree = spatial.cKDTree(scene.xyz)
    zen = np.radians(np.arange(step_deg / 2, max_zenith, step_deg))
    dirs = []
    for z in zen:
        n_az = max(1, int(round(360 / step_deg * np.sin(z))))
        az = np.radians((np.arange(n_az) + 0.5) * 360 / n_az)
        dirs.append(np.column_stack([np.sin(z) * np.sin(az), np.sin(z) * np.cos(az),
                                     np.full(n_az, np.cos(z))]))
    dirs = np.vstack(dirs)
    o = np.asarray(origin, float)
    rng_hit = np.full(len(dirs), np.nan)
    for i, d in enumerate(dirs):
        t0, t1 = (Z1 - R - o[2]) / d[2], (Z2 + R - o[2]) / d[2]
        ts = np.arange(t0, t1 + R, R)
        near = tree.query_ball_point(o + ts[:, None] * d, 1.2 * R)
        cand = np.unique(np.concatenate(near)).astype(int)
        if len(cand) == 0:
            continue
        v = scene.xyz[cand] - o
        tc = v @ d
        perp2 = np.sum(v * v, axis=1) - tc * tc
        ok = perp2 < R * R
        if ok.any():
            rng_hit[i] = np.min(tc[ok] - np.sqrt(R * R - perp2[ok]))
    hit = np.isfinite(rng_hit)
    count = hit.astype(np.int64)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    return Shots(np.tile(o, (len(dirs), 1)), dirs, start, count, rng_hit[hit],
                 {"classification": np.full(int(hit.sum()), 4, np.uint8)})


def test_matches_the_tls_pipeline(slab, slab_flight):
    """The same sphere layer scanned from below: the TLS hinge PAI and ray-traced
    PAD agree with the airborne estimates and the truth."""
    tls = _tls_scan(slab, (20.0, 20.0, 1.5))
    zen, gap = canopy.gap_fraction_zenith(tls, tls.echo_xyz()[:, 2], min_height=2.0,
                                          zenith_edges=np.arange(0, 61, 5.0))
    tls_pai = canopy.lai_from_gap_fraction(zen, gap, "hinge")
    g = voxels.ray_voxelize(tls, 1.0, ((10, 10, 0), (30, 30, 20)), unbounded_range=40.0)
    # Pooled over the layer: a voxel crossed by a few dozen pulses gives a
    # ratio estimate biased upwards, the mean of many such voxels too.
    layer = (g.z_levels() + 0.5 > Z1 + 1) & (g.z_levels() + 0.5 < Z2 - 1)
    tls_pad = 2 * g.num_hits_weighted[layer].sum() / g.free_path_length[layer].sum()
    als_pai = als.gap_profile(slab_flight.points, slab_flight.trajectory, resolution=10.0,
                              bounds=AREA, dtm=None, min_height=2.0).pooled_pai(_interior())
    assert tls_pai == pytest.approx(PAD * (Z2 - Z1), rel=0.05)
    assert tls_pad == pytest.approx(PAD, rel=0.06)
    assert als_pai == pytest.approx(tls_pai, rel=0.06)


def test_pickles(slab_flight):
    import pickle
    prof = als.gap_profile(slab_flight.points, slab_flight.trajectory, resolution=20.0, dtm=None)
    assert pickle.loads(pickle.dumps(prof)).weight.shape == prof.weight.shape
    traj = als.Trajectory.from_dict(slab_flight.trajectory)
    assert len(pickle.loads(pickle.dumps(traj))) == len(traj)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        repr(prof), repr(traj)
