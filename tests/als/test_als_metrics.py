# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Area-based ALS metrics: definitions, grids and plots.

The definitions are checked against NumPy and SciPy on random data, and
against an independent NumPy implementation of ``entropy`` and the
cumulative deciles (R-style ``seq`` breaks and half-open bins). Grids and plots from a
catalogue are checked against the metrics of the merged cloud's points,
and for independence from the chunking and the number of workers.
"""

import numpy as np
import pytest
from scipy import stats

from sylva import PointCloud, Raster, als, cli, synthetic
from sylva.geo import masks

SIZE = 60.0


# ------------------------------------------------------------------ references

def r_seq(to, by):
    """R's ``seq(0, to, by)``."""
    n = int(to / by + 1e-10)
    return np.minimum(np.arange(n + 1) * by, to)


def find_interval_table(z, breaks):
    """Counts of ``z`` in the half-open bins ``[breaks[i], breaks[i + 1])``."""
    k = len(breaks) - 1
    idx = np.searchsorted(breaks, z, side="right")
    return np.bincount(idx, minlength=k + 2)[1:k + 1].astype(float)


def ref_entropy(z, by=1.0):
    """Shannon index of the heights in bins of ``by`` from 0 to
    ``ceil(zmax / by) * by``, over that of a uniform distribution."""
    zmax = z.max()
    if zmax < 2 * by or z.min() < 0:
        return np.nan
    hist = find_interval_table(z, r_seq(np.ceil(zmax / by) * by, by))
    p = hist / hist.sum()
    p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(len(hist))


def ref_zpcum(z):
    """Cumulative deciles ``zpcum1`` .. ``zpcum9`` of ``[0, zmax)``."""
    zmax = z.max()
    if zmax <= 0:
        return np.zeros(9)
    d = find_interval_table(z, r_seq(zmax, zmax / 10))
    return np.cumsum(d / d.sum() * 100)[:9]


def reference(z, i, rn, cls, th=2.0, cover_break=2.0):
    """Every standard metric from NumPy and SciPy."""
    n = len(z)
    zq = {f"zq{5 * k}": np.quantile(z, k / 20) for k in range(1, 20)}
    first = rn == 1
    ref = {
        "n": n, "zmax": z.max(), "zmean": z.mean(), "zsd": z.std(ddof=1),
        "zskew": stats.skew(z, bias=True), "zkurt": stats.kurtosis(z, fisher=False, bias=True),
        "zentropy": ref_entropy(z), "pzabovezmean": (z > z.mean()).mean() * 100,
        "pzabove2": (z > th).mean() * 100, **zq,
        **{f"zpcum{k + 1}": v for k, v in enumerate(ref_zpcum(z))},
        "cover": (z[first] > cover_break).mean() * 100,
        "gap_fraction": (z[first] <= cover_break).mean(),
        "itot": i.sum(), "imax": i.max(), "imean": i.mean(), "isd": i.std(ddof=1),
        "iskew": stats.skew(i, bias=True), "ikurt": stats.kurtosis(i, fisher=False, bias=True),
        "ipground": i[cls == 2].sum() / i.sum() * 100,
        **{f"p{k}th": (rn == k).mean() * 100 for k in range(1, 6)},
        "pground": (cls == 2).mean() * 100,
    }
    for p in (10, 30, 50, 70, 90):
        ref[f"ipcumzq{p}"] = i[z <= np.quantile(z, p / 100)].sum() / i.sum() * 100
    return ref


def random_cloud(seed, n=2000):
    rng = np.random.default_rng(seed)
    z = np.where(rng.random(n) < 0.3, rng.normal(0.05, 0.03, n).clip(0), rng.gamma(3, 4, n))
    attrs = {"intensity": rng.integers(10, 3000, n).astype(np.uint16),
             "return_number": rng.integers(1, 6, n).astype(np.uint8),
             "classification": np.where(z < 0.2, 2, 1).astype(np.uint8)}
    xyz = np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 20, n), z])
    return PointCloud(xyz, attrs)


# ------------------------------------------------------------------ definitions

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_definitions_match_numpy_and_scipy(seed):
    c = random_cloud(seed)
    got = als.cloud_metrics(c)
    a = c.attrs
    ref = reference(c.z, a["intensity"].astype(float), a["return_number"].astype(float),
                    a["classification"].astype(float))
    assert list(got) == als.metric_names()
    assert set(got) == set(ref)
    for k, v in ref.items():
        np.testing.assert_allclose(got[k], v, rtol=1e-12, atol=1e-12, err_msg=k)


def test_entropy_and_deciles_on_analytic_samples():
    z = np.arange(11.0)          # one return per 1 m bin; 10 is on the top edge
    m = als.cloud_metrics(PointCloud(np.column_stack([z, z, z])))
    assert m["zentropy"] == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose([m[f"zpcum{k}"] for k in range(1, 10)], np.arange(10, 100, 10))
    assert m["zkurt"] == pytest.approx(1.78)   # Pearson kurtosis, not the excess
    assert m["zskew"] == pytest.approx(0.0, abs=1e-15)
    # Four returns in the first of three bins, one in the last.
    z = np.array([0.1, 0.2, 0.3, 0.4, 2.5])
    m = als.cloud_metrics(PointCloud(np.column_stack([z, z, z])))
    p = np.array([0.8, 0.2])
    assert m["zentropy"] == pytest.approx(-(p * np.log(p)).sum() / np.log(3))
    # Too low for entropy, or a negative height: NaN.
    for z in (np.array([0.1, 1.9]), np.array([-0.1, 5.0])):
        assert np.isnan(als.cloud_metrics(PointCloud(np.column_stack([z, z, z])))["zentropy"])


def test_cloud_metrics_options_and_edge_cases():
    c = random_cloud(4, 500)
    # Without attributes only the height and cover metrics exist.
    bare = als.cloud_metrics(PointCloud(c.xyz))
    assert "itot" not in bare and "pground" not in bare and "cover" in bare
    assert bare["cover"] == pytest.approx((c.z > 2).mean() * 100)
    # A height attribute, or an array of heights.
    shifted = PointCloud(c.xyz + [0, 0, 100], {**c.attrs, "height": c.z.copy()})
    assert als.cloud_metrics(shifted, height="height") == als.cloud_metrics(c)
    assert als.cloud_metrics(shifted, height=c.z) == als.cloud_metrics(c)
    # min_height, noise and NaN heights leave points out.
    cls = c.attrs["classification"].copy()
    cls[:50] = 7
    noisy = PointCloud(c.xyz, {**c.attrs, "classification": cls})
    assert als.cloud_metrics(noisy)["n"] == 450
    assert als.cloud_metrics(noisy, drop_noise=False)["n"] == 500
    assert als.cloud_metrics(c, min_height=2.0)["zq5"] >= 2.0
    h = c.z.copy()
    h[:10] = np.nan
    assert als.cloud_metrics(c, height=h)["n"] == 490
    other = als.cloud_metrics(c, threshold=5.5, cover_break=10.0, entropy_bin=2.0)
    assert other["pzabove5.5"] == pytest.approx((c.z > 5.5).mean() * 100)
    assert other["zentropy"] == pytest.approx(ref_entropy(c.z, 2.0))
    # Empty and single-point clouds.
    empty = als.cloud_metrics(PointCloud(np.zeros((0, 3))))
    assert empty["n"] == 0 and all(np.isnan(v) for k, v in empty.items() if k != "n")
    one = als.cloud_metrics(PointCloud([[0.0, 0.0, 3.0]]))
    assert one["zmax"] == 3.0 and np.isnan(one["zsd"]) and one["zq50"] == 3.0
    for bad in ({"entropy_bin": 0}, {"threshold": np.nan}, {"min_height": np.inf},
                {"height": "nope"}, {"height": np.zeros(3)}):
        with pytest.raises(ValueError):
            als.cloud_metrics(c, **bad)


# ------------------------------------------------------------------ catalogue

def _scene(seed=3, n_trees=12, size=SIZE):
    rng = np.random.default_rng(seed)
    trees = [(float(x), float(y), 0.3, float(h)) for x, y, h in
             zip(rng.uniform(6, size - 6, n_trees), rng.uniform(6, size - 6, n_trees),
                 rng.uniform(10, 22, n_trees), strict=True)]
    return synthetic.forest(trees, size=size, ground_points=100, margin=0.0, seed=seed)


@pytest.fixture(scope="module")
def tiles(tmp_path_factory):
    """A synthetic flight as 2 x 2 tiles of 30 m."""
    flight = synthetic.als_flight(_scene(), pulse_rate=12_000, line_spacing=30.0,
                                  bounds=(0.0, 0.0, SIZE, SIZE), seed=1)
    return flight.write_tiles(tmp_path_factory.mktemp("tiles"), size=30.0, epsg=32755)


@pytest.fixture(scope="module")
def merged(tiles):
    return tiles.read()


def _cell_clouds(cloud, grid: Raster):
    rows, cols = grid.cell_index(cloud.x, cloud.y)
    key = rows * grid.shape[1] + cols
    order = np.argsort(key, kind="stable")
    cuts = np.flatnonzero(np.diff(key[order])) + 1
    for idx in np.split(order, cuts):
        yield rows[idx[0]], cols[idx[0]], cloud[idx]


def test_grid_equals_the_merged_cloud_and_ignores_chunking(tiles, merged):
    res = 7.0   # cells straddle the 30 m tile edges
    one = als.grid_metrics(tiles, res, dtm=None, workers=1)
    many = als.grid_metrics(tiles, res, dtm=None, chunk_size=17.0, buffer=0.0, workers=3)
    assert list(one) == als.metric_names()
    for k in one:
        np.testing.assert_array_equal(one[k].data, many[k].data, err_msg=k)
    r = one["zmax"]
    assert r.crs == "EPSG:32755"
    assert (r.xmin, r.ymin, r.resolution) == (0.0, 0.0, res)
    seen = 0
    for row, col, cell in _cell_clouds(merged, r):
        want = als.cloud_metrics(cell)
        for k in ("n", "zmean", "zsd", "zq95", "zentropy", "zpcum3", "cover", "iskew", "p2th"):
            np.testing.assert_allclose(one[k].data[row, col], want[k], rtol=1e-12, atol=1e-12,
                                       err_msg=f"{k} at {row} {col}")
        seen += 1
    assert seen == np.isfinite(one["n"].data).sum()


def test_grid_with_heights_agrees_with_chm(tiles):
    m = als.grid_metrics(tiles, 2.0, ["zmax", "n", "pground"], workers=2)
    chm = als.chm(tiles, 2.0, workers=2)
    assert m["zmax"].shape == chm.shape and m["zmax"].xmin == chm.xmin
    has = m["zmax"].data >= 0
    assert has.mean() > 0.9
    np.testing.assert_array_equal(m["zmax"].data[has], chm.data[has])
    assert 20 < np.nanmax(m["zmax"].data) < 30
    # One name gives one raster.
    single = als.grid_metrics(tiles, 2.0, "zmax", workers=2)
    assert isinstance(single, Raster)
    np.testing.assert_array_equal(single.data, m["zmax"].data)
    with pytest.raises(ValueError, match="unknown metric"):
        als.grid_metrics(tiles, 2.0, ["zmx"])
    with pytest.raises(ValueError):
        als.grid_metrics(tiles, 0.0)
    with pytest.raises(ValueError):
        als.grid_metrics(tiles, 5.0, dtm=3)


def test_a_python_function_gives_what_the_builtin_gives(tiles):
    def f(c):
        return {"zmean": c.z.mean(), "n": len(c), "imax": c.attrs["intensity"].max()}

    mine = als.grid_metrics(tiles, 10.0, func=f, dtm=None)
    builtin = als.grid_metrics(tiles, 10.0, ["zmean", "n", "imax"], dtm=None)
    for k in builtin:
        np.testing.assert_allclose(mine[k].data, builtin[k].data, rtol=1e-12, err_msg=k)
    chunked = als.grid_metrics(tiles, 10.0, func=f, dtm=None, chunk_size=25.0)
    for k in mine:
        np.testing.assert_array_equal(mine[k].data, chunked[k].data)
    # A plain number is a "value" raster; metrics picks results.
    v = als.grid_metrics(tiles, 10.0, func=lambda c: c.z.max(), dtm=None, metrics="value")
    np.testing.assert_array_equal(v.data, als.grid_metrics(tiles, 10.0, "zmax", dtm=None).data)
    with pytest.raises(ValueError, match="returned no"):
        als.grid_metrics(tiles, 10.0, func=f, dtm=None, metrics=["zq95"])


def _inside_circle(cloud, x, y, r):
    return cloud[(cloud.x - x) ** 2 + (cloud.y - y) ** 2 <= r * r]


def test_plots_match_the_points_inside_them(tiles, merged):
    centres = np.array([[30.0, 30.0], [12.0, 44.0], [59.0, 2.0], [300.0, 300.0]])
    t = als.plot_metrics(tiles, centres, radius=[8.0, 5.0, 4.0, 5.0], dtm=None,
                         ids=["a", "b", "c", "far"])
    assert len(t) == 4 and list(t["plot"]) == [0, 1, 2, 3]
    assert list(t["id"]) == ["a", "b", "c", "far"]
    assert t.names == als.metric_names()
    for k, ((x, y), r) in enumerate(zip(centres[:3], [8.0, 5.0, 4.0], strict=True)):
        want = als.cloud_metrics(_inside_circle(merged, x, y, r))
        got = t.row(k)
        for name in t.names:
            np.testing.assert_allclose(got[name], want[name], rtol=1e-12, atol=1e-12,
                                       err_msg=f"plot {k} {name}")
    assert t["n"][3] == 0 and np.isnan(t["zmax"][3])
    # A square with a hole across the four tiles; boundary points are inside.
    ring = np.array([[20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0]])
    hole = np.array([[28.0, 28.0], [32.0, 28.0], [32.0, 32.0], [28.0, 32.0]])
    poly = masks.Polygon(ring, [hole])
    p = als.plot_metrics(tiles, [poly], metrics=["n", "zq50", "cover"], dtm=None)
    inside = merged[masks.inside_polygons(merged, poly)]
    want = als.cloud_metrics(inside)
    assert p["n"][0] == want["n"] == len(inside)
    assert p["zq50"][0] == pytest.approx(want["zq50"], rel=1e-12)
    # The same plots with a Python function.
    f = als.plot_metrics(tiles, centres, radius=5.0, dtm=None,
                         func=lambda c: {"n": len(c), "top": c.z.max() if len(c) else np.nan})
    b = als.plot_metrics(tiles, centres, radius=5.0, dtm=None, metrics=["n", "zmax"])
    np.testing.assert_array_equal(f["n"], b["n"])
    np.testing.assert_array_equal(f["top"], b["zmax"])
    for bad in ({"radius": 0.0}, {"radius": [1.0, 2.0]}, {"ids": ["x"]}):
        with pytest.raises(ValueError):
            als.plot_metrics(tiles, centres, **{"radius": 5.0, **bad})
    with pytest.raises(ValueError):
        als.plot_metrics(tiles, [[[0, 0], [1, 1]]])   # a ring of two vertices


def test_plot_heights_from_a_dtm_or_an_attribute(tiles, tmp_path):
    norm = als.normalize(tiles, tmp_path / "norm")       # adds a "height" attribute
    centres = [[15.0, 15.0], [45.0, 40.0]]
    by_attr = als.plot_metrics(norm, centres, radius=9.0, dtm="height")
    auto = als.plot_metrics(tiles, centres, radius=9.0)
    for k in ("n", "zmax", "zq95", "cover"):
        np.testing.assert_allclose(by_attr[k], auto[k], atol=2e-3, err_msg=k)
    dtm = als.dtm(tiles, 1.0)
    given = als.plot_metrics(tiles, centres, radius=9.0, dtm=dtm)
    np.testing.assert_allclose(given["zmax"], auto["zmax"], atol=0.05)
    grid = als.grid_metrics(norm, 10.0, ["zmax"], dtm="height")
    assert 20 < np.nanmax(grid["zmax"].data) < 30


def test_csv_and_table(tiles, tmp_path):
    t = als.plot_metrics(tiles, [[10.0, 10.0], [999.0, 999.0]], radius=5.0,
                         metrics=["n", "zmax"], dtm=None, ids=["p,1", "p2"])
    t.to_csv(tmp_path / "t.csv")
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert lines[0] == "plot,id,n,zmax"
    assert lines[1].startswith('0,"p,1",')
    assert lines[2] == "1,p2,0.0,"
    assert "2 plots" in repr(t) and t.as_dict()["n"][1] == 0


def test_command_line(tiles, tmp_path, capsys):
    d = str(tiles.paths[0]).rsplit("/", 1)[0]
    cli.main(["--no-progress", "als-metrics", d, str(tmp_path / "m"), "--resolution", "10",
              "--metrics", "zmax,cover,zq95", "--normalized"])
    want = als.grid_metrics(tiles, 10.0, ["zmax", "cover", "zq95"], dtm=None)
    for k, r in want.items():
        back = Raster.from_ascii_grid(tmp_path / "m" / f"{k}.asc")
        np.testing.assert_allclose(back.data, r.data, atol=1e-4)
    (tmp_path / "plots.csv").write_text("id,x,y\nA,20,20\nB,40,35\n")
    cli.main(["--no-progress", "als-plot-metrics", d, str(tmp_path / "plots.csv"),
              str(tmp_path / "p.csv"), "--radius", "6", "--metrics", "n,zmean"])
    rows = (tmp_path / "p.csv").read_text().splitlines()
    assert rows[0] == "plot,id,n,zmean" and rows[1].startswith("0,A,")
    t = als.plot_metrics(tiles, [[20, 20], [40, 35]], radius=6.0, metrics=["n"])
    assert float(rows[2].split(",")[2]) == t["n"][1]
    assert "2 plots" in capsys.readouterr().out
