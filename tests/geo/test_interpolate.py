"""sylva.geo.interpolate: attribute transfer, gridding and raster sampling, checked against
brute-force references and analytic surfaces."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from sylva import PointCloud, Raster, filters, ground, io
from sylva.geo import interpolate

LITCH = Path(__file__).parents[2] / "docs" / "examples" / "data" / "litch_tile.laz"


def _cloud(rng, n, size=10.0):
    return PointCloud(rng.uniform(0, size, (n, 3)))


def _brute_knn(src, tgt, k):
    d = np.linalg.norm(tgt[:, None, :] - src[None, :, :], axis=2)
    order = np.argsort(d, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(d, order, axis=1)


# ---------------------------------------------------------------- transfer_attributes

def test_nearest_matches_brute_force_and_keeps_dtypes():
    rng = np.random.default_rng(1)
    src = _cloud(rng, 500)
    src = src.with_attrs(tree_id=rng.integers(-1, 20, 500), intensity=rng.random(500).astype(np.float32),
                         classification=rng.integers(1, 5, 500).astype(np.uint8),
                         wood=rng.random(500) > 0.5)
    tgt = _cloud(rng, 300)
    out = interpolate.transfer_attributes(src, tgt)
    idx, _ = _brute_knn(src.xyz, tgt.xyz, 1)
    for name, a in src.attrs.items():
        assert out.attrs[name].dtype == a.dtype
        np.testing.assert_array_equal(out.attrs[name], a[idx[:, 0]])
    assert out.xyz is tgt.xyz


def test_fill_values_and_widening():
    src = PointCloud(np.zeros((1, 3)), {"cls": np.array([3], np.uint8), "tid": np.array([7]),
                                        "h": np.array([1.5]), "wood": np.array([True])})
    tgt = PointCloud(np.array([[0.05, 0, 0], [5, 0, 0], [np.nan, 0, 0]]))
    out = interpolate.transfer_attributes(src, tgt, max_distance=1.0)
    assert out.attrs["cls"].dtype == np.int16
    np.testing.assert_array_equal(out.attrs["cls"], [3, -1, -1])
    np.testing.assert_array_equal(out.attrs["tid"], [7, -1, -1])
    assert out.attrs["wood"].dtype == np.int8
    np.testing.assert_array_equal(out.attrs["h"], [1.5, np.nan, np.nan])
    kept = interpolate.transfer_attributes(src, tgt, "cls", max_distance=1.0, fill=0)
    assert kept.attrs["cls"].dtype == np.uint8
    np.testing.assert_array_equal(kept.attrs["cls"], [3, 0, 0])
    # Without max_distance a finite target always gets a value and the dtype is kept.
    near = interpolate.transfer_attributes(src, tgt[:2], "cls")
    assert near.attrs["cls"].dtype == np.uint8
    np.testing.assert_array_equal(near.attrs["cls"], [3, 3])
    idw = interpolate.transfer_attributes(src, tgt, "h", method="idw", max_distance=1.0, fill=-9.0)
    np.testing.assert_array_equal(idw.attrs["h"], [1.5, -9.0, -9.0])


def test_majority_matches_reference_with_ties_to_nearest():
    rng = np.random.default_rng(2)
    src = _cloud(rng, 400)
    labels = rng.integers(0, 4, 400)
    src = src.with_attrs(tree_id=labels, f=labels.astype(np.float64) / 2)
    tgt = _cloud(rng, 200)
    for k in (1, 2, 5, 8):
        out = interpolate.transfer_attributes(src, tgt, method="majority", k=k)
        order, _ = _brute_knn(src.xyz, tgt.xyz, k)
        expect = []
        for row in order:
            lab = labels[row]
            counts = {v: np.sum(lab == v) for v in lab}
            best = max(counts.values())
            expect.append(next(v for v in lab if counts[v] == best))   # first = nearest
        np.testing.assert_array_equal(out.attrs["tree_id"], expect)
        np.testing.assert_array_equal(out.attrs["f"], np.asarray(expect) / 2)
    one = interpolate.transfer_attributes(src, tgt, "tree_id", method="majority", k=1)
    nearest = interpolate.transfer_attributes(src, tgt, "tree_id")
    np.testing.assert_array_equal(one.attrs["tree_id"], nearest.attrs["tree_id"])


def test_majority_respects_max_distance():
    src = PointCloud(np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [3, 0, 0], [3.1, 0, 0]], float),
                     {"tid": np.array([1, 2, 2, 5, 5])})
    tgt = PointCloud(np.array([[0.0, 0, 0], [10, 0, 0]]))
    out = interpolate.transfer_attributes(src, tgt, "tid", method="majority", k=5)
    assert out.attrs["tid"][0] == 2            # 2 and 5 tie at two votes; 2 is nearer
    near = interpolate.transfer_attributes(src, tgt, "tid", method="majority", k=5, max_distance=0.5)
    np.testing.assert_array_equal(near.attrs["tid"], [2, -1])


def test_idw_matches_reference_and_is_bounded():
    rng = np.random.default_rng(3)
    src = _cloud(rng, 300)
    v = np.sin(src.x) + src.z
    src = src.with_attrs(v=v, n=rng.integers(0, 100, 300), s=v.astype(np.float32))
    tgt = _cloud(rng, 150, size=12.0)
    out = interpolate.transfer_attributes(src, tgt, ["v", "n", "s"], method="idw", k=6, power=1.5)
    order, dist = _brute_knn(src.xyz, tgt.xyz, 6)
    w = dist ** -1.5
    np.testing.assert_allclose(out.attrs["v"], (w * v[order]).sum(1) / w.sum(1), rtol=1e-12)
    np.testing.assert_allclose(out.attrs["n"], (w * src.attrs["n"][order]).sum(1) / w.sum(1), rtol=1e-12)
    assert out.attrs["n"].dtype == np.float64 and out.attrs["s"].dtype == np.float32
    assert v.min() <= out.attrs["v"].min() and out.attrs["v"].max() <= v.max()
    same = interpolate.transfer_attributes(src, src.without("v"), "v", method="idw")
    np.testing.assert_allclose(same.attrs["v"], v, rtol=0, atol=1e-12)


def test_idw_skips_nan_values():
    src = PointCloud(np.array([[0, 0, 0], [1, 0, 0]], float), {"v": np.array([np.nan, 4.0])})
    out = interpolate.transfer_attributes(src, PointCloud(np.array([[0.1, 0, 0]])), "v", method="idw")
    assert out.attrs["v"][0] == 4.0


def test_transfer_edge_cases():
    rng = np.random.default_rng(4)
    src = _cloud(rng, 50).with_attrs(tid=np.arange(50), name=np.array(["a"] * 50))
    empty = PointCloud(np.empty((0, 3)))
    out = interpolate.transfer_attributes(src, empty, "tid")
    assert len(out.attrs["tid"]) == 0 and out.attrs["tid"].dtype == src.attrs["tid"].dtype
    out = interpolate.transfer_attributes(empty.with_attrs(tid=np.empty(0, np.int64)), _cloud(rng, 5))
    np.testing.assert_array_equal(out.attrs["tid"], -1)
    out = interpolate.transfer_attributes(src, _cloud(rng, 5), [])
    assert out.attrs == {}
    with pytest.raises(ValueError, match="no attribute"):
        interpolate.transfer_attributes(src, _cloud(rng, 5), "nope")
    with pytest.raises(ValueError, match="unknown method"):
        interpolate.transfer_attributes(src, _cloud(rng, 5), method="kriging")
    with pytest.raises(ValueError, match="needs numbers"):
        interpolate.transfer_attributes(src, _cloud(rng, 5), "name", method="idw")
    for bad in ({"k": 0}, {"k": -2}, {"k": 1.5}, {"power": -1}, {"max_distance": -1.0},
                {"max_distance": np.nan}):
        with pytest.raises(ValueError):
            interpolate.transfer_attributes(src, _cloud(rng, 5), "tid", method="majority", **bad)
    # Strings work with nearest and majority.
    out = interpolate.transfer_attributes(src, _cloud(rng, 5), "name", method="majority")
    assert list(out.attrs["name"]) == ["a"] * 5


# ------------------------------------------------------------------------------ grid

def _scatter(n, seed=5, size=20.0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, size, (n, 2))
    return xy


def test_tin_and_natural_reproduce_a_plane():
    xy = _scatter(600)
    plane = lambda x, y: 100.0 + 0.3 * x - 0.7 * y   # noqa: E731
    cloud = PointCloud(np.column_stack([xy, plane(xy[:, 0], xy[:, 1])]))
    for method in ("tin", "natural"):
        r = interpolate.grid(cloud, 0.5, method=method)
        X, Y = r.cell_centers()
        ok = np.isfinite(r.data)
        assert ok.mean() > 0.8
        np.testing.assert_allclose(r.data[ok], plane(X, Y)[ok], rtol=0, atol=1e-9)


def test_projected_coordinates_keep_precision():
    xy = _scatter(400) + [712_345.0, 8_512_345.0]
    z = 12.0 + 0.1 * (xy[:, 0] - 712_345.0) + 0.05 * (xy[:, 1] - 8_512_345.0)
    r = interpolate.grid(PointCloud(np.column_stack([xy, z])), 0.5, method="tin")
    X, Y = r.cell_centers()
    ok = np.isfinite(r.data)
    np.testing.assert_allclose(r.data[ok], (12.0 + 0.1 * (X - 712_345.0) + 0.05 * (Y - 8_512_345.0))[ok],
                               atol=1e-7)


def test_smooth_surface_against_scipy():
    spi = pytest.importorskip("scipy.interpolate")
    xy = _scatter(2000, seed=6)
    f = lambda x, y: np.sin(x / 3) * np.cos(y / 4) + 0.02 * x   # noqa: E731
    cloud = PointCloud(np.column_stack([xy, f(xy[:, 0], xy[:, 1])]))
    tin = interpolate.grid(cloud, 0.4, method="tin")
    nat = interpolate.grid(cloud, 0.4, method="natural")
    X, Y = tin.cell_centers()
    ref = spi.LinearNDInterpolator(xy, f(xy[:, 0], xy[:, 1]))(X, Y)
    # Same hull; same values up to the choice of diagonal in cocircular cases.
    np.testing.assert_array_equal(np.isfinite(tin.data), np.isfinite(ref))
    ok = np.isfinite(ref)
    np.testing.assert_allclose(tin.data[ok], ref[ok], atol=1e-9)
    # Away from the hull, where long thin triangles are, both follow the surface closely.
    inner = ok & (X > 2) & (X < 18) & (Y > 2) & (Y < 18)
    truth = f(X, Y)[inner]
    for r in (tin, nat):
        assert np.abs(r.data[inner] - truth).max() < 0.05
    # Sibson coordinates are a convex combination: no overshoot beyond the data.
    vals = f(xy[:, 0], xy[:, 1])
    finite = nat.data[np.isfinite(nat.data)]
    assert vals.min() - 1e-12 <= finite.min() and finite.max() <= vals.max() + 1e-12


def test_idw_is_bounded_and_exact_at_data():
    xy = _scatter(300, seed=7)
    v = np.sin(xy[:, 0]) * xy[:, 1]
    cloud = PointCloud(np.column_stack([xy, v]))
    r = interpolate.grid(cloud, 0.5, bounds=(-5, -5, 25, 25), k=8, power=3.0)
    assert np.isfinite(r.data).all()
    assert v.min() - 1e-12 <= r.data.min() and r.data.max() <= v.max() + 1e-12
    # A point at a cell centre gives that cell its value.
    on = PointCloud(np.array([[0.25, 0.25, 7.0], [3.0, 3.0, 1.0]]))
    assert interpolate.grid(on, 0.5, bounds=(0, 0, 4, 4)).data[0, 0] == 7.0


def test_outside_hull_and_max_distance_are_nan():
    tri = PointCloud(np.array([[0, 0, 1], [10, 0, 2], [0, 10, 3]], float))
    for method in ("tin", "natural"):
        r = interpolate.grid(tri, 1.0, method=method, bounds=(0, 0, 10, 10))
        assert np.isfinite(r.data[0, 0]) and np.isnan(r.data[9, 9])
    r = interpolate.grid(tri, 1.0, bounds=(0, 0, 10, 10), max_distance=2.0)
    assert np.isnan(r.data[5, 5]) and np.isfinite(r.data[0, 0])
    r = interpolate.grid(tri, 1.0, method="tin", bounds=(0, 0, 10, 10), max_distance=2.0)
    assert np.isnan(r.data[2, 2]) and np.isfinite(r.data[0, 0])


def test_grid_geometry_matches_make_dtm():
    rng = np.random.default_rng(8)
    xyz = rng.uniform([3.3, -2.1, 0], [17.9, 9.4, 1], (500, 3))
    cloud = PointCloud(xyz, {"classification": np.full(500, 2, np.uint8)})
    for bounds in (None, (-1.0, -0.5, 13.0, 12.5)):
        a = ground.make_dtm(cloud, 0.75, bounds)
        b = interpolate.grid(cloud, 0.75, bounds=bounds)
        assert (a.shape, a.xmin, a.ymin, a.resolution) == (b.shape, b.xmin, b.ymin, b.resolution)


def test_grid_value_attribute_duplicates_and_nan():
    pts = np.array([[0, 0, 0], [0, 0, 0], [4, 0, 0], [0, 4, 0], [2, 2, 0]], float)
    cloud = PointCloud(pts, {"v": np.array([1.0, 3.0, 2.0, 2.0, np.nan])})
    r = interpolate.grid(cloud, 0.5, value="v", method="tin", bounds=(-0.25, -0.25, 0, 0))
    assert r.data[0, 0] == pytest.approx(2.0)     # duplicates averaged, NaN ignored


def test_grid_errors_and_empty():
    cloud = PointCloud(np.random.default_rng(9).random((20, 3)), {"s": np.array(["a"] * 20)})
    with pytest.raises(ValueError, match="empty"):
        interpolate.grid(PointCloud(np.empty((0, 3))), 1.0)
    r = interpolate.grid(PointCloud(np.empty((0, 3))), 1.0, method="tin", bounds=(0, 0, 2, 2))
    assert r.shape == (3, 3) and np.isnan(r.data).all()
    for kw in ({"method": "spline"}, {"value": "nope"}, {"value": "s"}, {"k": 0}, {"power": -2},
               {"max_distance": -1}, {"bounds": (0, 0, 1)}, {"bounds": (1, 0, 0, 1)}):
        with pytest.raises(ValueError):
            interpolate.grid(cloud, 0.5, **kw)
    for res in (0, -1, np.nan):
        with pytest.raises(ValueError, match="resolution"):
            interpolate.grid(cloud, res)
    # Fewer than three points: TIN has no triangle, IDW still works.
    two = PointCloud(np.array([[0, 0, 1], [1, 1, 2]], float))
    assert np.isfinite(interpolate.grid(two, 0.5).data).all()
    interpolate.grid(two, 0.5, method="natural")


def test_make_dtm_methods():
    rng = np.random.default_rng(10)
    xy = rng.uniform(0, 20, (5000, 2))
    z = 0.05 * xy[:, 0] + 0.2 * np.sin(xy[:, 1] / 3)
    cloud = PointCloud(np.column_stack([xy, z]), {"classification": np.full(5000, 2, np.uint8)})
    default = ground.make_dtm(cloud, 0.5)
    np.testing.assert_array_equal(default.data, ground.make_dtm(cloud, 0.5, method="lowest").data)
    X, Y = default.cell_centers()
    truth = 0.05 * X + 0.2 * np.sin(Y / 3)
    for method in ("tin", "natural", "idw"):
        dtm = ground.make_dtm(cloud, 0.5, method=method)
        assert dtm.shape == default.shape and not np.isnan(dtm.data).any()
        inner = (X > 1) & (X < 19) & (Y > 1) & (Y < 19)
        assert np.abs(dtm.data - truth)[inner].max() < (0.01 if method != "idw" else 0.05)
    with pytest.raises(ValueError, match="unknown method"):
        ground.make_dtm(cloud, method="kriging")
    with pytest.raises(ValueError, match="3 ground"):
        ground.make_dtm(cloud[:2], method="tin")


# ---------------------------------------------------------------------- sample_raster

def test_sample_raster_bilinear_nearest_and_outside():
    rng = np.random.default_rng(11)
    r = Raster(rng.random((8, 10)), 100.0, 50.0, 0.5)
    inside = rng.uniform([100, 50], [r.xmax, r.ymax], (200, 2))
    pts = np.vstack([inside, [[99.9, 51], [105.0, 51], [101, 49.99], [101, 54.0], [np.nan, 51]]])
    cloud = PointCloud(np.column_stack([pts, np.zeros(len(pts))]))
    out = interpolate.sample_raster(cloud, r, "v")
    np.testing.assert_array_equal(out.attrs["v"][:200], r.sample(inside[:, 0], inside[:, 1]))
    assert np.isnan(out.attrs["v"][200:]).all()
    near = interpolate.sample_raster(cloud, r, "v", method="nearest").attrs["v"]
    row, col = r.cell_index(inside[:, 0], inside[:, 1])
    np.testing.assert_array_equal(near[:200], r.data[row, col])
    assert np.isnan(near[200:]).all()


def test_sample_raster_recovers_a_plane_and_several_rasters():
    X, Y = np.meshgrid(np.arange(20) + 0.5, np.arange(15) + 0.5)
    a = Raster(2 * X + 3 * Y, 0.0, 0.0, 1.0)
    b = Raster(np.where(X > 10, np.nan, 1.0), 0.0, 0.0, 1.0)
    rng = np.random.default_rng(12)
    xy = rng.uniform([0.5, 0.5], [19.5, 14.5], (300, 2))
    cloud = PointCloud(np.column_stack([xy, np.zeros(300)]))
    out = interpolate.sample_rasters(cloud, {"a": a, "b": b})
    np.testing.assert_allclose(out.attrs["a"], 2 * xy[:, 0] + 3 * xy[:, 1], atol=1e-12)
    assert np.isnan(out.attrs["b"][xy[:, 0] > 11]).all()
    np.testing.assert_allclose(out.attrs["b"][xy[:, 0] < 9.5], 1.0, rtol=1e-12)
    with pytest.raises(ValueError, match="unknown method"):
        interpolate.sample_raster(cloud, a, "a", method="cubic")
    with pytest.raises(ValueError, match="Raster"):
        interpolate.sample_raster(cloud, a.data, "a")
    empty = interpolate.sample_raster(PointCloud(np.empty((0, 3))), a, "a")
    assert empty.attrs["a"].shape == (0,)


def test_grid_then_sample_round_trip():
    xy = _scatter(1500, seed=13)
    z = 5 + 0.2 * xy[:, 0] - 0.1 * xy[:, 1]
    cloud = PointCloud(np.column_stack([xy, z]))
    r = interpolate.grid(cloud, 0.5, method="tin")
    back = interpolate.sample_raster(cloud, r, "g", method="nearest").attrs["g"]
    ok = np.isfinite(back)
    # Nearest-cell sampling of a plane is off by at most half a cell's slope.
    assert np.abs(back[ok] - z[ok]).max() <= 0.5 * 0.5 * (0.2 + 0.1) + 1e-9


# --------------------------------------------------------------------------- real data

@pytest.mark.skipif(not LITCH.exists(), reason="example tile not present")
def test_litchfield_tile():
    cloud = io.read(LITCH)
    thin = filters.voxel_downsample(cloud, 0.2)
    back = interpolate.transfer_attributes(thin, cloud, "classification", method="majority", k=5)
    assert back.attrs["classification"].dtype == cloud.attrs["classification"].dtype
    assert np.mean(back.attrs["classification"] == cloud.attrs["classification"]) > 0.95
    dtm = ground.make_dtm(cloud, 0.5)
    g = cloud[ground.ground_mask(cloud)]
    tin = interpolate.grid(filters.voxel_downsample(g, 0.25), 0.5, method="tin", bounds=(dtm.xmin, dtm.ymin, dtm.xmax - 1e-9, dtm.ymax - 1e-9))
    assert tin.shape == dtm.shape
    ok = np.isfinite(tin.data)
    assert ok.mean() > 0.9
    assert np.median(np.abs(tin.data[ok] - dtm.data[ok])) < 0.1
    h = interpolate.sample_raster(g, tin, "ground").attrs["ground"]
    assert np.nanmedian(np.abs(g.z - h)) < 0.1


# ------------------------------------------------------------------------------- docs

def test_public_api_is_documented():
    for name in interpolate.__all__:
        obj = getattr(interpolate, name)
        doc = inspect.getdoc(obj)
        assert doc and "Parameters" in doc and "Returns" in doc, name
