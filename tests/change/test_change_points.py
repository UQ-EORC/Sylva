"""sylva.change distances (C2C, M3C2) and rasters of difference, checked against known
displacements, the analytic false-positive rate of the level of detection and, when it is
installed, the independent M3C2 of py4dgeo."""

import inspect

import numpy as np
import pytest

from sylva import PointCloud, Raster, change
from sylva.change import points as change_points
from sylva.change import voxels as change_voxels

Z95 = 1.959963984540054


def _surface(rng, n, shift=0.0, noise=0.005, size=10.0):
    xy = rng.uniform(0, size, (n, 2))
    z = 0.3 * np.sin(xy[:, 0]) + 0.2 * np.cos(0.7 * xy[:, 1]) + shift + rng.normal(0, noise, n)
    return np.column_stack([xy, z])


def _core(step=0.3, lo=0.5, hi=9.5):
    g = np.arange(lo, hi, step)
    cx, cy = np.meshgrid(g, g)
    cx, cy = cx.ravel(), cy.ravel()
    return np.column_stack([cx, cy, 0.3 * np.sin(cx) + 0.2 * np.cos(0.7 * cy)])


M3C2 = dict(normal_scale=0.5, projection_scale=0.2, max_depth=0.3)


@pytest.fixture(scope="module")
def surfaces():
    rng = np.random.default_rng(7)
    a = _surface(rng, 400_000)
    same = _surface(rng, 400_000)
    raised = _surface(rng, 400_000, shift=0.02)
    return a, same, raised


# ---------------------------------------------------------------- c2c

def test_c2c_is_the_nearest_distance():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 5, (400, 3))
    b = rng.uniform(0, 5, (300, 3))
    d = change.distances(PointCloud(a), PointCloud(b), "c2c")
    brute = np.linalg.norm(b[:, None] - a[None], axis=2).min(1)
    assert d.method == "c2c" and d.lod is None
    np.testing.assert_allclose(d.distance, brute, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(d.core_points, b)
    capped = change.distances(a, b, "c2c", max_distance=0.3)
    np.testing.assert_array_equal(np.isnan(capped.distance), brute > 0.3)
    cloud = d.to_cloud()
    assert list(cloud.attrs) == ["distance"] and len(cloud) == 300


def test_c2c_edge_cases():
    b = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    assert np.isnan(change.distances(np.empty((0, 3)), b, "c2c").distance).all()
    d = change.distances(np.array([[1.0, 0, 0], [np.inf, 0, 0]]), b, "c2c").distance
    assert d[0] == 1.0 and np.isnan(d[1])
    assert len(change.distances(b, np.empty((0, 3)), "c2c")) == 0
    with pytest.raises(ValueError, match="core_points"):
        change.distances(b, b, "c2c", core_points=b)
    with pytest.raises(ValueError, match="max_distance"):
        change.distances(b, b, "c2c", max_distance=-1)


# ---------------------------------------------------------------- m3c2

def test_m3c2_recovers_a_known_displacement(surfaces):
    a, _, raised = surfaces
    d = change.distances(a, raised, "m3c2", _core(), **M3C2)
    # The surface moved 2 cm in z, so 2 cm * n_z along each (upward) normal.
    expected = 0.02 * d.normal[:, 2]
    assert np.all(d.normal[:, 2] > 0.8)
    assert np.nanmean(np.abs(d.distance - expected)) < 0.001
    assert abs(np.nanmean(d.distance - expected)) < 0.0003
    assert d.significant.all()
    assert (d.n_a > 50).all() and (d.n_b > 50).all()


def test_m3c2_false_positive_rate_matches_the_level_of_detection(surfaces):
    a, same, _ = surfaces
    core = _core(step=0.21)
    d = change.distances(a, same, "m3c2", core, **M3C2)
    rate = d.significant.mean()
    # About 1900 independent cylinders: a 95 % level gives 5 % +- 1 %.
    assert 0.035 < rate < 0.07, rate
    # The level of detection is eq. 1 of Lague et al. (2013).
    lod = Z95 * np.sqrt(d.spread_a ** 2 / d.n_a + d.spread_b ** 2 / d.n_b)
    np.testing.assert_allclose(d.lod, lod, rtol=1e-12)
    # Registration error is added linearly and removes the false positives.
    r = change.distances(a, same, "m3c2", core, registration_sigma=0.005, **M3C2)
    np.testing.assert_allclose(r.lod - d.lod, Z95 * 0.005, rtol=0, atol=1e-12)
    assert r.significant.mean() < 0.001
    np.testing.assert_array_equal(r.distance, d.distance)


def test_m3c2_on_a_stem_with_radial_normals():
    rng = np.random.default_rng(3)
    def stem(r, n):
        th = rng.uniform(0, 2 * np.pi, n)
        z = rng.uniform(0, 3, n)
        rr = r + rng.normal(0, 0.002, n)
        return np.column_stack([rr * np.cos(th), rr * np.sin(th), z])
    a, b = stem(0.20, 200_000), stem(0.21, 200_000)
    th = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    core = np.column_stack([0.2 * np.cos(th), 0.2 * np.sin(th), np.full(36, 1.5)])
    normals = np.column_stack([np.cos(th), np.sin(th), np.zeros(36)])
    d = change.distances(a, b, "m3c2", core, normal_scale=0.1, projection_scale=0.05, max_depth=0.1, normals=normals)
    np.testing.assert_allclose(d.distance, 0.01, atol=0.001)
    assert d.significant.all()
    # Fitted normals oriented towards a location outside the stem agree on the near side.
    near = core[:9]
    f = change.distances(a, b, "m3c2", near, normal_scale=0.1, projection_scale=0.05, max_depth=0.1,
                         orientation=("towards", (5.0, 2.0, 1.5)))
    np.testing.assert_allclose(f.distance, 0.01, atol=0.0015)


def test_m3c2_defaults_and_result_cloud(surfaces):
    a, _, raised = surfaces
    sub = a[::4000]
    d = change.distances(PointCloud(a), PointCloud(raised), normal_scale=0.5, projection_scale=0.2)
    assert d.method == "m3c2" and len(d) == len(a)
    e = change.distances(a, raised, core_points=sub, normal_scale=0.5, projection_scale=0.2, max_depth=0.5)
    np.testing.assert_array_equal(e.distance, d.distance[::4000])
    cloud = e.to_cloud()
    assert {"distance", "lod", "significant", "nx", "ny", "nz", "n_a", "n_b"} <= set(cloud.attrs)
    assert "significant" in repr(e)
    down = change.distances(a, raised, core_points=sub, orientation=("direction", (0, 0, -1)), **M3C2)
    np.testing.assert_allclose(down.distance, -e.distance, atol=1e-12)


def test_m3c2_edge_cases():
    rng = np.random.default_rng(5)
    a = _surface(rng, 20_000)
    core = np.array([[5.0, 5.0, 0.3 * np.sin(5.0) + 0.2 * np.cos(3.5)], [np.nan, 0, 0], [50.0, 50.0, 0.0]])
    d = change.distances(a, np.empty((0, 3)), "m3c2", core, **M3C2)
    assert np.isnan(d.distance).all() and (d.n_b == 0).all() and not d.significant.any()
    d = change.distances(a, a, "m3c2", core, min_points=10 ** 6, **M3C2)
    assert np.isfinite(d.distance[0]) and np.isnan(d.lod[0]) and not d.significant[0]
    assert np.isnan(d.normal[1:]).all()
    assert len(change.distances(a, a, "m3c2", np.empty((0, 3)), **M3C2)) == 0
    bad = [
        (dict(normal_scale=0.5), "projection_scale"),
        (dict(M3C2, normal_scale=0), "normal_scale"),
        (dict(M3C2, max_depth=np.nan), "max_depth"),
        (dict(M3C2, registration_sigma=-0.1), "registration_sigma"),
        (dict(M3C2, min_points=0), "min_points"),
        (dict(M3C2, orientation="down"), "orientation"),
        (dict(M3C2, orientation=("direction", (0, 0, 0))), "zero"),
        (dict(M3C2, normals=np.ones((2, 3))), "normals"),
    ]
    for kw, msg in bad:
        with pytest.raises(ValueError, match=msg):
            change.distances(a, a, "m3c2", core, **kw)
    with pytest.raises(ValueError, match="method"):
        change.distances(a, a, "icp")
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        change.distances(a[:, :2], a, "c2c")


def test_m3c2_matches_py4dgeo(surfaces, tmp_path, monkeypatch):
    """py4dgeo (Zahs et al.) is only a test reference, never a dependency."""
    monkeypatch.chdir(tmp_path)  # it writes a log file to the working directory
    py4dgeo = pytest.importorskip("py4dgeo")
    a, _, raised = surfaces
    core = _core()
    m = py4dgeo.M3C2(epochs=(py4dgeo.Epoch(a), py4dgeo.Epoch(raised)), corepoints=core, normal_radii=(0.25,),
                     cyl_radius=0.1, max_distance=0.3, registration_error=0.002)
    dist, unc = m.run()
    d = change.distances(a, raised, "m3c2", core, registration_sigma=0.002, **M3C2)
    np.testing.assert_allclose(np.abs(np.sum(m.directions() * d.normal, axis=1)), 1.0, atol=1e-9)
    np.testing.assert_array_equal(unc["num_samples1"], d.n_a)
    np.testing.assert_array_equal(unc["num_samples2"], d.n_b)
    np.testing.assert_allclose(dist, d.distance, rtol=0, atol=1e-9)
    np.testing.assert_allclose(unc["lodetection"], d.lod, rtol=0, atol=1e-6)
    np.testing.assert_allclose(unc["spread1"], d.spread_a, rtol=0, atol=1e-6)


# ---------------------------------------------------------------- dod

def test_dod_significance_matches_the_analytic_rate():
    rng = np.random.default_rng(11)
    n = 400
    sa, sb, step = 0.05, 0.08, 0.2
    truth = np.zeros((n, n))
    truth[:, n // 2:] = step
    a = Raster(rng.normal(0, sa, (n, n)), 0.0, 0.0, 0.5)
    b = Raster(truth + rng.normal(0, sb, (n, n)), 0.0, 0.0, 0.5)
    d = change.dod(a, b, sigma_a=sa, sigma_b=sb)
    s = np.hypot(sa, sb)
    np.testing.assert_allclose(d.lod.data, Z95 * s)
    unchanged = d.significant[:, : n // 2].mean()
    changed = d.significant[:, n // 2:].mean()
    from math import erf, sqrt
    phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    power = phi((step - Z95 * s) / s) + phi((-step - Z95 * s) / s)
    se = lambda p: np.sqrt(p * (1 - p) / (n * n / 2))
    assert abs(unchanged - 0.05) < 4 * se(0.05), unchanged
    assert abs(changed - power) < 4 * se(power), (changed, power)
    diff = d.difference.data
    area = 0.25
    assert d.volume_gained == pytest.approx(diff[d.significant & (diff > 0)].sum() * area)
    assert d.volume_lost == pytest.approx(-diff[d.significant & (diff < 0)].sum() * area)
    assert d.net_volume == pytest.approx(d.volume_gained - d.volume_lost)
    assert d.area_changed == pytest.approx(d.significant.sum() * area)
    assert d.area_compared == pytest.approx(n * n * area)


def test_dod_overlap_nan_and_rasters_of_uncertainty():
    a = Raster(np.zeros((4, 5)), 0.0, 0.0, 1.0, crs="EPSG:28355")
    bd = np.full((4, 5), 0.3)
    bd[0, 0] = np.nan
    b = Raster(bd, 2.0, 1.0, 1.0)
    d = change.dod(a, b, min_detectable=0.2)
    assert d.difference.shape == (3, 3) and (d.difference.xmin, d.difference.ymin) == (2.0, 1.0)
    assert d.difference.crs == "EPSG:28355"
    assert np.isnan(d.difference.data[0, 0]) and not d.significant[0, 0]
    assert d.significant.sum() == 8
    t = d.thresholded()
    assert np.isnan(t.data[0, 0]) and t.data[1, 1] == pytest.approx(0.3)
    sig = Raster(np.full((4, 5), 0.01), 0.0, 0.0, 1.0)
    sig.data[2, 3] = 1.0
    e = change.dod(a, b, sigma_a=sig)
    assert e.lod.data[1, 0] == pytest.approx(Z95 * 0.01 * np.sqrt(2))
    assert e.lod.data[1, 1] == pytest.approx(Z95 * np.sqrt(2))
    assert not e.significant[1, 1] and e.significant[1, 0]
    m = change.dod(a, b, min_detectable=Raster(np.full((2, 2), 0.5), 2.0, 1.0, 1.0))
    assert np.isnan(m.lod.data[2, 2]) and m.significant.sum() == 0
    z = change.dod(a, b, min_detectable=0.5)
    assert z.thresholded().data[1, 1] == 0.0


def test_dod_rejects_bad_input():
    a = Raster(np.zeros((3, 3)), 0.0, 0.0, 1.0)
    for kw, msg in [({}, "level of detection"), (dict(min_detectable=0.1, sigma_a=0.1), "not both"),
                    (dict(min_detectable=-0.1), "non-negative"), (dict(sigma_b=np.nan), "non-negative")]:
        with pytest.raises(ValueError, match=msg):
            change.dod(a, a, **kw)
    with pytest.raises(ValueError, match="lattice"):
        change.dod(a, Raster(np.zeros((3, 3)), 0.5, 0.0, 1.0), min_detectable=0.1)
    with pytest.raises(ValueError, match="resolution"):
        change.dod(a, Raster(np.zeros((3, 3)), 0.0, 0.0, 2.0), min_detectable=0.1)
    with pytest.raises(ValueError, match="overlap"):
        change.dod(a, Raster(np.zeros((3, 3)), 30.0, 0.0, 1.0), min_detectable=0.1)
    with pytest.raises(ValueError, match="lattice"):
        change.dod(a, a, sigma_a=Raster(np.zeros((3, 3)), 0.25, 0.0, 1.0))
    with pytest.raises(ValueError, match="Raster"):
        change.dod(np.zeros((3, 3)), a, min_detectable=0.1)


# ---------------------------------------------------------------- docs

@pytest.mark.parametrize("mod", [change_points, change_voxels])
def test_public_api_is_documented(mod):
    for name in mod.__all__:
        obj = getattr(mod, name)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        assert inspect.getdoc(obj), name
        if inspect.isfunction(obj):
            doc = inspect.getdoc(obj)
            for p in inspect.signature(obj).parameters:
                assert p in doc, f"{name}: parameter {p} undocumented"
    assert set(mod.__all__) <= set(change.__all__)
