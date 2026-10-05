# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The synthetic airborne scanner: its timing and geometry, checked against
the conventions its documentation states, recomputed here in NumPy."""

import numpy as np
import pytest

from sylva import PointCloud, synthetic


@pytest.fixture(scope="module")
def scene():
    return synthetic.forest(ground_points=100)


@pytest.fixture(scope="module")
def flight(scene):
    return synthetic.als_flight(scene, altitude=70.0, pulse_rate=10_000, line_spacing=12.0,
                                heading=30.0, attitude=(2.0, 1.0, 1.5), seed=4)


def _rotation(roll, pitch, heading):
    """Body (x forward, y right, z down) to map (east, north, up), as documented."""
    r, p, h = np.radians(roll), np.radians(pitch), np.radians(heading)
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]])
    ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    return ned_to_enu @ rz @ ry @ rx


def test_every_return_lies_on_its_beam(flight):
    pts, traj = flight.points, flight.trajectory
    t = pts.attrs["gps_time"]
    origin = flight.sensor_positions(t)
    assert np.isfinite(origin).all()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(pts), 400, replace=False):
        k = np.searchsorted(traj["time"], t[i], side="right") - 1
        # Attitude between the two samples around the pulse, linearly.
        f = (t[i] - traj["time"][k]) / (traj["time"][k + 1] - traj["time"][k])
        roll, pitch, heading = ((1 - f) * traj[a][k] + f * traj[a][k + 1] for a in ("roll", "pitch", "heading"))
        mirror = np.radians(pts.attrs["scan_angle"][i] + roll)
        d = _rotation(roll, pitch, heading) @ np.array([0.0, np.sin(mirror), np.cos(mirror)])
        v = pts.xyz[i] - origin[i]
        assert np.dot(v, d) / np.linalg.norm(v) > 1 - 1e-8, i


def test_timing_follows_the_flight_plan(flight):
    pts, traj = flight.points, flight.trajectory
    t = pts.attrs["gps_time"]
    lines = pts.attrs["point_source_id"]
    assert set(np.unique(lines)) == set(np.unique(traj["line"]))
    # Pulses leave every 1 / pulse_rate s from each line's start.
    for line in np.unique(lines):
        t0 = traj["time"][traj["line"] == line][0]
        j = (t[lines == line] - t0) * 10_000
        assert np.abs(j - np.round(j)).max() < 1e-5
    assert (np.diff(t) >= 0).all()
    # The trajectory: samples every 1 / 100 s, straight lines at 10 m/s and 70 m.
    for line in np.unique(traj["line"]):
        m = traj["line"] == line
        dt = np.diff(traj["time"][m])
        assert np.allclose(dt[:-1], 0.01)
        speed = np.hypot(np.diff(traj["x"][m]), np.diff(traj["y"][m])) / dt
        assert np.allclose(speed, 10.0)
        assert np.allclose(traj["z"][m], 70.0)
    # Alternate lines fly opposite ways, about the given heading.
    first = traj["heading"][traj["line"] == 1]
    second = traj["heading"][traj["line"] == 2]
    assert np.abs(first - 30.0).max() <= 1.5 + 1e-9
    assert np.abs(second - 210.0).max() <= 1.5 + 1e-9
    assert np.abs(traj["roll"]).max() <= 2.0 and np.abs(traj["pitch"]).max() <= 1.0
    # Between lines the laser is off: the trajectory stops and the next line
    # starts after the turn.
    ends = [traj["time"][traj["line"] == k][[0, -1]] for k in (1, 2)]
    assert ends[1][0] - ends[0][1] == pytest.approx(10.0)
    assert flight.n_pulses >= len(np.unique(t))


def test_returns_of_a_pulse_are_numbered_nearest_first(flight):
    pts = flight.points
    t, rn, nr = pts.attrs["gps_time"], pts.attrs["return_number"], pts.attrs["number_of_returns"]
    origin = flight.sensor_positions(t)
    rng = np.linalg.norm(pts.xyz - origin, axis=1)
    first = np.flatnonzero(rn == 1)
    assert 1 <= nr.min() and nr.max() <= 5 and (rn <= nr).all()
    multi = first[nr[first] > 1]
    assert len(multi) > 0.02 * len(first)
    for i in multi[:300]:
        n = nr[i]
        if i + n > len(t) or not (t[i:i + n] == t[i]).all():
            continue        # a later return was clipped away
        assert list(rn[i:i + n]) == list(range(1, n + 1))
        assert (np.diff(rng[i:i + n]) > 0.5).all()
    # Last returns reach the ground more often than first returns.
    cls = pts.attrs["classification"]
    last = rn == nr
    assert (cls[last] == 2).mean() > (cls[rn == 1] == 2).mean()


def test_ground_returns_lie_on_the_terrain(flight):
    pts = flight.points
    g = pts.attrs["classification"] == 2
    dz = pts.z[g] - synthetic.terrain_height(pts.x[g], pts.y[g])
    assert np.abs(dz).max() < 0.15
    assert abs(np.std(dz) - 0.02) < 0.01
    assert pts.attrs["tree_id"][g].max() == 0
    assert set(np.unique(pts.attrs["tree_id"][~g])) <= {1, 2, 3, 4}


def test_scan_patterns_and_angles(scene):
    kw = dict(pulse_rate=4000, scan_rate=40.0, scan_angle=20.0, attitude=(0, 0, 0), range_noise=0.0,
              clip=False)
    osc = synthetic.als_flight(scene, scan_pattern="oscillating", **kw)
    rot = synthetic.als_flight(scene, scan_pattern="rotating", **kw)
    for f in (osc, rot):
        a = f.points.attrs["scan_angle"]
        assert -20.0 - 1e-4 <= a.min() and a.max() <= 20.0 + 1e-4
        assert a.max() - a.min() > 39.0
    # First returns in firing order: the rotating mirror only ever sweeps one way.
    def steps(f):
        first = f.points.attrs["return_number"] == 1
        return np.diff(f.points.attrs["scan_angle"][first])
    s_rot, s_osc = steps(rot), steps(osc)
    assert (s_rot > 0).mean() > 0.95
    assert 0.4 < (s_osc > 0).mean() < 0.6


def test_density_and_footprint_follow_the_settings(scene):
    lo = synthetic.als_flight(scene, pulse_rate=5_000, max_returns=1)
    hi = synthetic.als_flight(scene, pulse_rate=20_000, max_returns=1)
    assert len(hi.points) / len(lo.points) == pytest.approx(4.0, rel=0.05)
    assert hi.points.attrs["number_of_returns"].max() == 1
    thin = synthetic.als_flight(scene, pulse_rate=5_000, footprint_samples=1)
    wide = synthetic.als_flight(scene, pulse_rate=5_000, footprint_samples=19, beam_divergence=5.0)
    frac = [np.mean(f.points.attrs["number_of_returns"] > 1) for f in (thin, wide)]
    assert frac[1] > frac[0]
    assert thin.points.attrs["number_of_returns"].max() == 1


def test_a_seed_gives_the_same_flight(scene):
    kw = dict(pulse_rate=3000)
    a, b = synthetic.als_flight(scene, seed=9, **kw), synthetic.als_flight(scene, seed=9, **kw)
    c = synthetic.als_flight(scene, seed=10, **kw)
    assert np.array_equal(a.points.xyz, b.points.xyz)
    for k in a.trajectory:
        assert np.array_equal(a.trajectory[k], b.trajectory[k])
    assert not np.array_equal(a.trajectory["roll"], c.trajectory["roll"])


def test_bad_settings_are_refused(scene):
    with pytest.raises(ValueError, match="scan pattern"):
        synthetic.als_flight(scene, scan_pattern="palmer")
    with pytest.raises(ValueError, match="fly through"):
        synthetic.als_flight(scene, altitude=10.0)
    with pytest.raises(ValueError, match="footprint_samples"):
        synthetic.als_flight(scene, footprint_samples=5)
    with pytest.raises(ValueError, match="max_returns"):
        synthetic.als_flight(scene, max_returns=0)
    with pytest.raises(ValueError, match="pulse_rate"):
        synthetic.als_flight(scene, pulse_rate=float("nan"))
    with pytest.raises(ValueError, match="bounds"):
        synthetic.als_flight(scene, bounds=(0, 0, 1))
    with pytest.raises(ValueError, match="scan_angle"):
        synthetic.als_flight(scene, scan_angle=80.0)


def test_bare_terrain_can_be_flown(scene):
    f = synthetic.als_flight(PointCloud(np.empty((0, 3))), bounds=(0, 0, 20, 20), pulse_rate=2000)
    assert len(f.points) > 0
    assert (f.points.attrs["classification"] == 2).all()
    assert "tree_id" not in f.points.attrs
    with pytest.raises(ValueError, match="empty"):
        synthetic.als_flight(scene[:0])
