import numpy as np

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
