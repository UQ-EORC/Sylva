import numpy as np
import pytest

from sylva import PointCloud, Shots


def test_from_pointcloud_groups_by_time(rng):
    # Two echoes per pulse, sharing gps_time.
    n = 100
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    near = d * 5
    far = d * 8
    xyz = np.empty((2 * n, 3))
    xyz[0::2] = far  # deliberately out of range order
    xyz[1::2] = near
    t = np.repeat(np.arange(n, dtype=float), 2)
    pc = PointCloud(xyz, {"gps_time": t, "amplitude": rng.uniform(size=2 * n).astype(np.float32)})
    shots = Shots.from_pointcloud(pc)
    assert shots.n_shots == n
    assert shots.n_echoes == 2 * n
    assert np.all(shots.echo_count == 2)
    np.testing.assert_allclose(shots.echo_range[0::2], 5)
    np.testing.assert_allclose(shots.echo_range[1::2], 8)
    assert "amplitude" in shots.echo_attrs
    back = shots.to_pointcloud()
    assert set(back.attrs) >= {"return_number", "number_of_returns", "range", "amplitude"}
    assert list(back.attrs["return_number"][:2]) == [1, 2]


def test_subset_and_transform(rng):
    n = 50
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    shots = Shots.from_pointcloud(PointCloud(d * 3))
    sub = shots.subset(np.arange(n) < 10)
    assert sub.n_shots == 10 and sub.n_echoes == 10
    m = np.eye(4)
    m[:3, 3] = [1, 0, 0]
    moved = shots.transform(m)
    np.testing.assert_allclose(moved.origin[:, 0], 1)
    np.testing.assert_allclose(moved.echo_xyz(), shots.echo_xyz() + [1, 0, 0], atol=1e-9)


def _random_shots(rng, n=5000, origins=None):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    count = rng.choice([0, 1, 2, 3], n, p=[0.4, 0.4, 0.15, 0.05])
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    ne = int(count.sum())
    ranges = rng.uniform(2, 60, ne)
    order = np.lexsort((ranges, np.repeat(np.arange(n), count)))  # ascending within a shot
    if origins is None:
        origins = np.array([[0, 0, 1.5], [20, 5, 1.8], [-7, 30, 2.1]])[rng.integers(0, 3, n)]
    attrs = {
        "reflectance": rng.normal(-10, 3, ne).astype(np.float32),
        "classification": rng.integers(0, 7, ne).astype(np.uint8),
        "tree_id": rng.integers(-1, 40, ne).astype(np.int32),
        "gps_time": rng.uniform(0, 1e5, ne),
    }
    return Shots(origins, d, start, count, ranges[order], attrs)


def test_shots_file_roundtrip(rng, tmp_path):
    shots = _random_shots(rng)
    path = tmp_path / "plot.parquet"
    shots.save(path, row_group_size=1200)  # several row groups
    info = Shots.file_info(path)
    assert info["n_shots"] == shots.n_shots and info["n_echoes"] == shots.n_echoes
    assert info["n_groups"] == 5 and len(info["scans"]) == 3
    assert set(info["echo_attrs"]) == set(shots.echo_attrs)
    np.testing.assert_allclose(info["bounds"][0], shots.echo_xyz().min(axis=0))

    back = Shots.load(path)
    np.testing.assert_array_equal(back.echo_count, shots.echo_count)
    np.testing.assert_array_equal(back.echo_start, shots.echo_start)
    np.testing.assert_array_equal(back.origin, shots.origin)
    np.testing.assert_allclose(back.echo_xyz(), shots.echo_xyz(), atol=5e-5)  # float32 angles
    np.testing.assert_allclose(back.direction, shots.direction, atol=1e-6)    # misses too
    for k, v in shots.echo_attrs.items():
        assert back.echo_attrs[k].dtype == v.dtype
        np.testing.assert_array_equal(back.echo_attrs[k], v)

    part = Shots.load(path, groups=[1])
    assert part.n_shots == 1200
    np.testing.assert_array_equal(part.echo_count, shots.echo_count[1200:2400])

    shots.save(path, double=True, origin_tolerance=0)
    exact = Shots.load(path)
    np.testing.assert_allclose(exact.echo_xyz(), shots.echo_xyz(), atol=1e-9)


def test_shots_file_snaps_noisy_origins_and_keeps_trajectories(rng, tmp_path):
    shots = _random_shots(rng, 3000)
    noisy = Shots(shots.origin + rng.normal(0, 2e-5, shots.origin.shape), shots.direction,
                  shots.echo_start, shots.echo_count, shots.echo_range, shots.echo_attrs)
    noisy.save(tmp_path / "noisy.parquet")
    assert len(Shots.file_info(tmp_path / "noisy.parquet")["scans"]) == 3
    back = Shots.load(tmp_path / "noisy.parquet")
    np.testing.assert_allclose(back.echo_xyz(), noisy.echo_xyz(), atol=1e-4)  # echoes do not move

    # A moving platform: more origins than a scan table holds.
    n = 70000
    t = np.linspace(0, 1, n)
    track = np.column_stack([100 * t, 30 * np.sin(6 * t), 50 + t])
    mobile = _random_shots(rng, n, origins=track)
    mobile.save(tmp_path / "mobile.parquet")
    assert len(Shots.file_info(tmp_path / "mobile.parquet")["scans"]) == 0
    back = Shots.load(tmp_path / "mobile.parquet")
    np.testing.assert_allclose(back.origin, track, atol=5.1e-5)
    np.testing.assert_allclose(back.echo_xyz(), mobile.echo_xyz(), atol=2e-4)


def test_empty_shots_round_trip(tmp_path):
    import numpy as np

    from sylva import Shots

    empty = Shots(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    empty.save(tmp_path / "empty.parquet")
    back = Shots.load(tmp_path / "empty.parquet")
    assert len(back.origin) == 0 and len(back.echo_range) == 0


def _lines(n_lines=4, per_line=6, down=False, short_line=1):
    """Pulses on a regular grid: each line sweeps 30-130 degrees at its own azimuth."""
    zen_steps = np.linspace(30.0, 130.0, per_line)
    dirs, counts, rng_, amp = [], [], [], []
    for line in range(n_lines):
        zs = zen_steps[::-1] if down else zen_steps
        if line == short_line:
            zs = zs[:-1]
        for i, z in enumerate(zs):
            a = np.radians(line * 1.0)
            zr = np.radians(z)
            dirs.append([np.sin(zr) * np.sin(a), np.sin(zr) * np.cos(a), np.cos(zr)])
            miss = i == 2
            counts.append(0 if miss else 1)
            if not miss:
                rng_.append(10.0 + i)
                amp.append(float(line))
    counts = np.array(counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return Shots(np.zeros((len(dirs), 3)), np.array(dirs), starts, counts, np.array(rng_),
                 {"amplitude": np.array(amp)})


def test_scan_lines_and_scan_image(tmp_path):
    s = _lines()
    starts, sweep, along = s.scan_lines()
    assert starts.tolist() == [0, 6, 11, 17, 23] and sweep == 1.0 and along == "zenith"
    assert s.angular_steps() == pytest.approx((20.0, 1.0))
    # On the line grid the image is lines by index; on the angle grid (the
    # default) the same here, since the lines are regular.
    for grid in ("lines", "angles"):
        img = s.scan_image(grid=grid)
        assert img.field == "range" and img.n_lines == 4 and img.data.shape == (6, 4)
        assert img.grid == grid and (img.steps is None) == (grid == "lines")
        assert img.data[0, 0] == 10.0 and img.data[5, 0] == 15.0
        assert np.isnan(img.data[2, 0])      # the pulse that returned nothing
        assert np.isnan(img.data[5, 1])      # the short line's unfilled end
        assert img.line_start.tolist() == starts.tolist()
        assert s.scan_image("n_echoes", grid=grid).data[2, 0] == 0.0
        assert s.scan_image("amplitude", echo="last", grid=grid).data[0, 3] == 3.0
        assert s.scan_image("zenith", grid=grid).data[0, 0] == pytest.approx(30.0)
        assert s.scan_image("azimuth", grid=grid).data[0, 3] == pytest.approx(3.0)
        assert s.scan_image(np.arange(s.n_shots, dtype=float), grid=grid).data[1, 1] == 7.0
        assert s.scan_image(np.full(s.n_echoes, 2.0), grid=grid).data[0, 0] == 2.0
    with pytest.raises(ValueError, match="unknown field"):
        s.scan_image("colour")
    with pytest.raises(ValueError, match="first' or 'last"):
        s.scan_image(echo="middle")
    with pytest.raises(ValueError, match="grid must be"):
        s.scan_image(grid="polar")
    with pytest.raises(ValueError, match="values for"):
        s.scan_image(np.zeros(5))
    with pytest.raises(ValueError, match="steps must be"):
        s.scan_image(steps=(0.0, 1.0))
    # A downward sweep still puts the sky at the top unless asked not to.
    d = _lines(down=True)
    assert d.scan_lines()[1] == -1.0
    for grid in ("lines", "angles"):
        assert d.scan_image("zenith", grid=grid).data[0, 0] == pytest.approx(30.0)
        assert d.scan_image("zenith", grid=grid, sky_up=False).data[0, 0] == pytest.approx(130.0)
    # Misses recovered after the recorded pulses (as read_rxp_shots leaves
    # them) land in place on the angle grid, and form lines of their own on
    # the line grid.
    m = _lines(short_line=-1)
    miss = np.flatnonzero(m.echo_count == 0)
    kept = np.setdiff1d(np.arange(m.n_shots), miss)
    order = np.r_[kept, miss]
    counts = m.echo_count[order]
    reordered = Shots(m.origin, m.direction[order], np.cumsum(np.r_[0, counts])[:-1], counts,
                      m.echo_range, m.echo_attrs)
    assert reordered.scan_image("zenith").data.shape == (6, 4)
    assert np.isnan(reordered.scan_image().data[2]).all()
    assert reordered.scan_image(grid="lines").data.shape == (5, 5)   # the misses make one more line
    # Colouring and writing.
    rgb = img.to_rgb(log=True)
    assert rgb.shape == (6, 4, 3) and rgb.dtype == np.uint8
    assert (rgb[2, 0] == 0).all()
    out = img.save(tmp_path / "scan.png", cmap="gray", vmin=10, vmax=15)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_scan_lines_use_time_against_stray_pulses():
    s = _lines(short_line=-1)
    # One pulse of line two points back up the sweep, as a stray out-of-sequence
    # pulse does; without timing it starts a line, with timing it does not.
    s.direction[8] = s.direction[6]
    assert len(s.scan_lines()[0]) - 1 == 5
    t = np.arange(s.n_shots, dtype=float) * 1e-5
    t[[6, 12, 18]] += 1e-3   # the mirror turns between lines
    s.echo_count[:] = 1
    s.echo_start[:] = np.arange(s.n_shots)
    s.echo_range = np.full(s.n_shots, 10.0)
    s.echo_attrs = {"gps_time": t}
    starts, _, _ = s.scan_lines()
    assert starts.tolist() == [0, 6, 12, 18, 24]


def test_scan_image_of_an_azimuth_sweeping_scanner():
    # Lines of constant zenith sweeping the azimuth, from the ground up as a
    # turning head does: the image is laid out zenith down, azimuth across,
    # sky at the top all the same.
    zens, azs = [120.0, 90.0, 60.0], np.arange(0.0, 360.0, 30.0)
    dirs = []
    for z in zens:
        for a in azs:
            zr, ar = np.radians(z), np.radians(a)
            dirs.append([np.sin(zr) * np.sin(ar), np.sin(zr) * np.cos(ar), np.cos(zr)])
    n = len(dirs)
    s = Shots(np.zeros((n, 3)), np.array(dirs), np.arange(n), np.ones(n, dtype=int),
              np.arange(n, dtype=float))
    starts, sweep, along = s.scan_lines()
    assert starts.tolist() == [0, 12, 24, 36] and sweep == 1.0 and along == "azimuth"
    assert s.angular_steps() == pytest.approx((30.0, 30.0))
    for grid in ("lines", "angles"):
        img = s.scan_image("zenith", grid=grid)
        assert img.data.shape == (3, 12) and img.n_lines == 3
        assert img.data[0, 0] == pytest.approx(60.0) and img.data[2, 0] == pytest.approx(120.0)
        assert s.scan_image("azimuth", grid=grid).data[0, 5] == pytest.approx(150.0)
        assert s.scan_image("zenith", grid=grid, sky_up=False).data[0, 0] == pytest.approx(120.0)
        assert s.scan_image(grid=grid).data[2, 0] == 0.0
    assert s.scan_image("zenith", grid="lines").along == "azimuth"
    # A single pulse is one line.
    one = Shots(np.zeros((1, 3)), [[0.0, 0.0, 1.0]], [0], [1], [5.0])
    assert one.scan_lines()[0].tolist() == [0, 1] and one.scan_image().data.shape == (1, 1)


def test_fill_missing_from_timing():
    # Four lines of 50 slots, 0.1 us apart, 30-130 deg; line 1 lost slots
    # 10-14 (a gap) and 45-49 (its sky end), line 3 lost slot 0.
    lost = {(1, k) for k in list(range(10, 15)) + list(range(45, 50))} | {(3, 0)}
    dirs, times = [], []
    for line in range(4):
        for k in range(50):
            if (line, k) in lost:
                continue
            z, a = np.radians(30.0 + 2.0 * k), np.radians(10.0 + line)
            dirs.append([np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)])
            times.append(line * 2e-3 + k * 1e-7)
    n = len(dirs)
    attrs = {"gps_time": np.array(times), "amplitude": np.zeros(n, dtype=np.float32)}
    s = Shots(np.zeros((n, 3)), np.array(dirs), np.arange(n), np.ones(n, dtype=int),
              np.full(n, 5.0), attrs)
    filled, rep = s.fill_missing_from_timing((30.0, 128.0), return_report=True)
    assert (rep["n_gap"], rep["n_before"], rep["n_after"]) == (5, 1, 5)
    assert filled.n_shots == n + 11 and filled.n_echoes == n
    miss = filled.echo_count == 0
    zen, az = filled.zenith_azimuth()
    on_line_1 = np.sort(zen[miss][az[miss].round() == 11])
    assert np.allclose(on_line_1, 30.0 + 2.0 * np.r_[10:15, 45:50], atol=1e-6)
    assert filled.echo_attrs["amplitude"].dtype == np.float32
    # The stride keeps every other fired pulse, recorded or not.
    half = s.fill_missing_from_timing((30.0, 128.0), shot_stride=2)
    assert half.n_shots == (n + 11 + 1) // 2 or abs(half.n_shots - (n + 11) / 2) <= 2
    with pytest.raises(ValueError, match="no gps_time"):
        Shots(np.zeros((1, 3)), [[0.0, 0.0, 1.0]], [0], [1], [5.0]).fill_missing_from_timing()
    with pytest.raises(ValueError, match="already hold misses"):
        filled.fill_missing_from_timing()


def test_scan_image_takes_out_the_azimuth_swing_of_a_line():
    # 40 lines of 120 pulses at 0.5 deg zenith steps and 0.1 deg azimuth
    # steps, each line's azimuth swinging by 0.6 deg over its sweep as the
    # mirror geometry does: on the angle grid every pulse still lands in its
    # own cell, so a line keeps to one column.
    zens = 30.0 + 0.5 * np.arange(120)
    swing = 0.6 * np.cos(np.radians(zens)) ** 2
    dirs = []
    for line in range(40):
        for z, sw in zip(zens, swing, strict=True):
            zr, ar = np.radians(z), np.radians(10.0 + 0.1 * line + sw)
            dirs.append([np.sin(zr) * np.sin(ar), np.sin(zr) * np.cos(ar), np.cos(zr)])
    n = len(dirs)
    s = Shots(np.zeros((n, 3)), np.array(dirs), np.arange(n), np.ones(n, dtype=int),
              np.arange(n, dtype=float))
    img = s.scan_image(steps=(0.5, 0.1))
    assert img.data.shape == (120, 40)
    assert not np.isnan(img.data).any()
    assert img.data[:, 7].tolist() == list(np.arange(7 * 120, 8 * 120, dtype=float))
    # Misses recovered on a line (appended, azimuth swung like the line's)
    # go to that line's column; the recorded pulses they replace are gone.
    lost = np.zeros(n, bool)
    lost[7 * 120 + 50:7 * 120 + 60] = True
    m = int((~lost).sum())
    kept = Shots(s.origin[~lost], s.direction[~lost], np.arange(m), np.ones(m, dtype=int),
                 np.arange(n, dtype=float)[~lost])
    zeros = np.zeros(10, dtype=int)
    misses = Shots(s.origin[lost], s.direction[lost], zeros, zeros, np.zeros(0))
    both = Shots.concatenate([kept, misses])
    img = both.scan_image("n_echoes", steps=(0.5, 0.1))
    assert img.data.shape == (120, 40) and not np.isnan(img.data).any()
    assert (img.data[50:60, 7] == 0).all() and (img.data[:50, 7] == 1).all()
