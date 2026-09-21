import numpy as np
import pytest

from sylva import PointCloud, Shots, io


@pytest.fixture
def cloud(rng):
    xyz = rng.uniform(-10, 10, (500, 3))
    return PointCloud(xyz, {
        "intensity": rng.integers(0, 65535, 500, dtype=np.uint16),
        "classification": rng.integers(0, 3, 500, dtype=np.uint8),
        "height": rng.uniform(0, 30, 500),
        "tree_id": rng.integers(-1, 5, 500, dtype=np.int32),
    })


@pytest.mark.parametrize("ext", [".las", ".laz", ".ply", ".xyz", ".csv", ".txt"])
def test_roundtrip(cloud, tmp_path, ext):
    path = tmp_path / f"cloud{ext}"
    io.write(cloud, path)
    back = io.read(path)
    assert len(back) == len(cloud)
    np.testing.assert_allclose(back.xyz, cloud.xyz, atol=1e-3)
    for name in ("intensity", "classification", "height", "tree_id"):
        assert name in back.attrs, name
        np.testing.assert_allclose(back.attrs[name], cloud.attrs[name], atol=1e-5)


def test_las_extra_bytes_dtype(cloud, tmp_path):
    path = tmp_path / "c.las"
    io.write(cloud, path)
    back = io.read(path)
    assert back.attrs["height"].dtype == np.float64
    assert back.attrs["tree_id"].dtype == np.int32
    assert back.attrs["intensity"].dtype == np.uint16


def test_ascii_ply(cloud, tmp_path):
    path = tmp_path / "c.ply"
    io.write(cloud, path, binary=False)
    back = io.read(path)
    np.testing.assert_allclose(back.xyz, cloud.xyz, atol=1e-5)
    assert "height" in back.attrs


def test_xyz_no_header(tmp_path):
    path = tmp_path / "raw.xyz"
    path.write_text("1 2 3 10\n4 5 6 20\n")
    pc = io.read(path)
    assert pc.xyz.shape == (2, 3)
    assert list(pc.attrs["col3"]) == [10, 20]


def test_ascii_named_columns(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("1 2 3 10\n4 5 6 20\n")
    pc = io.read_ascii(path, columns=["intensity"])
    assert list(pc.attrs["intensity"]) == [10, 20]


def test_pts_count_line(tmp_path):
    path = tmp_path / "raw.pts"
    path.write_text("2\n1 2 3\n4 5 6\n")
    assert len(io.read(path)) == 2


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        io.read(tmp_path / "x.foo")


def test_missing_file(tmp_path):
    with pytest.raises(OSError):
        io.read(tmp_path / "nope.las")


def test_ray_cloud_ply(rng, tmp_path):
    # raycloudtools convention: nx,ny,nz point from the end back to the sensor.
    n = 200
    origin = np.array([1.0, 2.0, 3.0])
    ends = origin + rng.normal(size=(n, 3)) * 5
    to_sensor = origin - ends
    alpha = np.ones(n, dtype=np.uint8) * 100
    alpha[:20] = 0  # unbounded
    pc = PointCloud(ends, {"nx": to_sensor[:, 0], "ny": to_sensor[:, 1], "nz": to_sensor[:, 2],
                           "alpha": alpha})
    path = tmp_path / "rays.ply"
    io.write(pc, path)
    shots = Shots.from_ray_cloud(io.read(path))
    assert shots.n_shots == n
    assert shots.n_echoes == n - 20
    np.testing.assert_allclose(shots.origin, np.tile(origin, (n, 1)), atol=1e-6)
    np.testing.assert_allclose(shots.echo_xyz(), ends[20:], atol=1e-6)


def test_matrix_file(tmp_path):
    path = tmp_path / "sop.dat"
    m = np.eye(4)
    m[:3, 3] = [1, 2, 3]
    np.savetxt(path, m)
    np.testing.assert_allclose(io.read_matrix_file(path), m)


def test_riegl_missing_library(tmp_path, monkeypatch):
    monkeypatch.delenv("RIVLIB_PATH", raising=False)
    monkeypatch.delenv("RIVLIB_HOME", raising=False)
    try:
        io.find_rivlib()
    except ValueError:
        with pytest.raises(ValueError, match="libscanifc"):
            io.read_rxp(tmp_path / "scan.rxp")
    else:
        pytest.skip("RiVLib is installed")
