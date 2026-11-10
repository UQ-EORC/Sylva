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


def test_rdb_missing_library(tmp_path, monkeypatch):
    monkeypatch.delenv("RDBLIB_PATH", raising=False)
    monkeypatch.delenv("RDBLIB_HOME", raising=False)
    try:
        io.find_rdblib()
    except ValueError as e:
        assert "RDBLIB_PATH" in str(e)
        with pytest.raises(ValueError, match="librdb"):
            io.read_rdbx(tmp_path / "scan.rdbx")
        with pytest.raises(ValueError, match="librdb"):
            io.rdbx_info(tmp_path / "scan.rdbx")
    else:
        pytest.skip("the RDB SDK is installed")


def test_read_some_attributes(cloud, tmp_path):
    path = tmp_path / "cloud.laz"
    io.write(cloud, path)
    some = io.read(path, attrs=["height", "tree_id", "nothing_like_this"])
    assert sorted(some.attrs) == ["height", "tree_id"]
    full = io.read(path)
    np.testing.assert_array_equal(some.xyz, full.xyz)
    np.testing.assert_array_equal(some.attrs["tree_id"], full.attrs["tree_id"])
    assert io.read(path, attrs=[]).attrs == {}
    # Other formats read everything and drop the rest.
    io.write(cloud, tmp_path / "cloud.ply")
    assert sorted(io.read(tmp_path / "cloud.ply", attrs=["height"]).attrs) == ["height"]


def test_write_annotated_is_write_with_attrs(cloud, tmp_path):
    src = tmp_path / "src.laz"
    cloud.crs = "EPSG:7855"
    io.write(cloud, src)
    back = io.read(src)
    label = np.arange(len(cloud), dtype=np.int32) % 7 - 1
    score = np.linspace(0, 1, len(cloud), dtype=np.float32)
    whole = tmp_path / "whole.laz"
    io.write(back.with_attrs(tree_id=label, score=score), whole)   # tree_id replaced, score added
    streamed = tmp_path / "streamed.laz"
    io.write_annotated(src, streamed, {"tree_id": label, "score": score}, origin=back.bounds[0])
    assert streamed.read_bytes() == whole.read_bytes()
    # Without an origin the source header's minimum is used, which is the same corner.
    io.write_annotated(src, tmp_path / "streamed2.laz", {"tree_id": label, "score": score})
    assert (tmp_path / "streamed2.laz").read_bytes() == whole.read_bytes()
    out = io.read(streamed)
    assert out.crs == back.crs
    np.testing.assert_array_equal(out.attrs["tree_id"], label)
    with pytest.raises(ValueError):
        io.write_annotated(src, tmp_path / "bad.laz", {"tree_id": label[:-1]})
    with pytest.raises(ValueError):
        io.write_annotated(tmp_path / "cloud.ply", tmp_path / "bad.laz", {"tree_id": label})


def test_las_write_spans_batches(tmp_path):
    # Past one batch of the writer (2**20 points), so the pieces meet in the file.
    n = (1 << 20) + 1234
    r = np.random.default_rng(5)
    xyz = np.column_stack([r.uniform(0, 50, n), r.uniform(0, 50, n), r.uniform(0, 30, n)])
    c = PointCloud(xyz, {"intensity": r.integers(0, 65535, n, dtype=np.uint16),
                         "tree_id": np.arange(n, dtype=np.int32)})
    path = tmp_path / "big.las"
    io.write(c, path)
    back = io.read(path)
    assert len(back) == n
    np.testing.assert_allclose(back.xyz, xyz, atol=1e-3)
    np.testing.assert_array_equal(back.attrs["tree_id"], c.attrs["tree_id"])
    np.testing.assert_array_equal(back.attrs["intensity"], c.attrs["intensity"])
    io.write_annotated(path, tmp_path / "big2.las", {"flag": (np.arange(n) % 2).astype(np.uint8)})
    again = io.read(tmp_path / "big2.las")
    np.testing.assert_array_equal(again.attrs["flag"], np.arange(n) % 2)
    np.testing.assert_array_equal(again.attrs["tree_id"], c.attrs["tree_id"])


def test_las_keeps_colour_flags_and_class_twelve(tmp_path):
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    c = PointCloud(xyz, {"red": np.array([1, 2, 3], np.uint16), "green": np.array([4, 5, 6], np.uint16),
                         "blue": np.array([7, 8, 9], np.uint16),
                         "classification": np.array([12, 2, 5], np.uint8),
                         "withheld": np.array([0, 0, 1], np.uint8),
                         "id": np.array([2**53 + 1, 5, -2**60 - 3], np.int64)})
    io.write(c, tmp_path / "c.laz")                     # default format 6 is promoted to 7
    back = io.read(tmp_path / "c.laz")
    for k in ("red", "green", "blue", "withheld", "id"):
        np.testing.assert_array_equal(back.attrs[k], c.attrs[k], err_msg=k)
    np.testing.assert_array_equal(back.attrs["classification"], [1, 2, 5])
    np.testing.assert_array_equal(back.attrs["overlap"], [1, 0, 0])
    io.write(c, tmp_path / "old.las", point_format=1)   # formats 0-5 store class 12 itself
    np.testing.assert_array_equal(io.read(tmp_path / "old.las").attrs["classification"], [12, 2, 5])


def test_unwritable_las_coordinates_raise(tmp_path):
    far = PointCloud(np.array([[0.0, 0.0, 0.0], [3.0e6, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="larger scale"):
        io.write(far, tmp_path / "far.las")
    with pytest.raises(ValueError):
        io.write(far, tmp_path / "zero.las", scale=0.0)


def test_text_precision_follows_the_crs(tmp_path):
    c = PointCloud(np.array([[153.0123456789, -27.4987654321, 12.0]]))
    c.crs = "EPSG:4326"
    io.write(c, tmp_path / "geo.csv")
    np.testing.assert_allclose(io.read(tmp_path / "geo.csv").xyz, c.xyz, atol=1e-8)
    (tmp_path / "gap.csv").write_text("x,y,z,intensity,label\n1,2,3,,5\n4,5,6,9,7\n")
    gap = io.read(tmp_path / "gap.csv")
    assert np.isnan(gap.attrs["intensity"][0]) and list(gap.attrs["label"]) == [5.0, 7.0]


def test_malformed_ply_raises(tmp_path):
    head = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    (tmp_path / "short.ply").write_text(head + "0 0 0\n1 1 1\n")
    (tmp_path / "prop.ply").write_text(head.replace("property float z\n", "property float\n") + "0 0 0\n")
    (tmp_path / "huge.ply").write_text(head.replace("ascii", "binary_little_endian").replace("vertex 3", "vertex 4000000000000000"))
    for name in ("short.ply", "prop.ply", "huge.ply"):
        with pytest.raises((OSError, ValueError)):
            io.read(tmp_path / name)
    big = PointCloud(np.zeros((1, 3)), {"tree_id": np.array([5_000_000_000], np.int64)})
    io.write(big, tmp_path / "big.ply")
    assert io.read(tmp_path / "big.ply").attrs["tree_id"][0] == 5_000_000_000
