# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Area-based metrics: argument checks, the plot table's other forms, and the
plot inputs of ``sylva als-plot-metrics``."""

import json

import numpy as np
import pytest

from sylva import PointCloud, Raster, als, cli, io


@pytest.fixture(scope="module")
def flat_tile(tmp_path_factory):
    """One 40 m tile: ground at z = 10 (class 2) and vegetation 0-15 m above it."""
    rng = np.random.default_rng(21)
    d = tmp_path_factory.mktemp("flat")
    g = np.column_stack([rng.uniform(0, 40, (4000, 2)), np.full(4000, 10.0)])
    v = np.column_stack([rng.uniform(0, 40, (4000, 2)), 10.0 + rng.uniform(0, 15, 4000)])
    xyz = np.vstack([g, v])
    cls = np.repeat(np.array([2, 5], np.uint8), 4000)
    io.write(PointCloud(xyz, {"classification": cls, "height": xyz[:, 2] - 10.0}), d / "t.las")
    return als.catalog(d)


@pytest.fixture(scope="module")
def unclassified_tile(tmp_path_factory):
    rng = np.random.default_rng(22)
    d = tmp_path_factory.mktemp("unclassified")
    io.write(PointCloud(rng.uniform(0, 20, (500, 3))), d / "t.las")
    return als.catalog(d)


# --------------------------------------------------------------------------- #
# argument checks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kwargs, message", [
    ({"dtm": "auto", "dtm_resolution": 0}, "dtm_resolution must be a positive number, got 0"),
    ({"dtm": "auto", "dtm_resolution": float("nan")}, "dtm_resolution must be a positive number"),
    ({"dtm": ""}, "dtm must be 'auto', None, a Raster or an attribute name"),
    ({"dtm": 3}, "dtm must be 'auto', None, a Raster or an attribute name, got int"),
    ({"metrics": []}, "metrics must name at least one metric, or be None for all"),
    ({"func": "zmax"}, "func must be callable as func(cloud)"),
])
def test_grid_metrics_rejects_bad_arguments(flat_tile, kwargs, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        als.grid_metrics(flat_tile, 10.0, **kwargs)


def test_a_function_must_return_what_is_asked(flat_tile):
    with pytest.raises(ValueError, match="func returned no 'zmax'; it returned n"):
        als.grid_metrics(flat_tile, 20.0, ["zmax"], dtm=None, func=lambda c: {"n": len(c)})
    with pytest.raises(ValueError, match="func returned no 'top'; it returned n"):
        als.plot_metrics(flat_tile, [[10.0, 10.0]], radius=3.0, metrics=["top"], dtm=None,
                         func=lambda c: {"n": len(c)})


def test_a_function_on_unclassified_tiles_needs_a_dtm(unclassified_tile):
    with pytest.raises(ValueError, match=r"no chunk has 3 ground points \(classification 2\)"):
        als.grid_metrics(unclassified_tile, 10.0, func=lambda c: {"n": len(c)})
    # With z as height it runs, and counts every point once.
    n = als.grid_metrics(unclassified_tile, 10.0, dtm=None, func=lambda c: {"n": len(c)})["n"]
    assert np.nansum(n.data) == 500


def test_plot_metrics_rejects_bad_plots(flat_tile):
    with pytest.raises(ValueError, match=r"with a radius, plots must be \(N, 2\) centres, "
                                         r"got shape \(2, 3\)"):
        als.plot_metrics(flat_tile, np.zeros((2, 3)), radius=5.0)
    with pytest.raises(ValueError, match="buffer must be zero or more, got -1"):
        als.plot_metrics(flat_tile, [[10.0, 10.0]], radius=5.0, buffer=-1)
    with pytest.raises(ValueError, match=r"func must be callable as func\(cloud\)"):
        als.plot_metrics(flat_tile, [[10.0, 10.0]], radius=5.0, func=1.0)


# --------------------------------------------------------------------------- #
# plots and tables
# --------------------------------------------------------------------------- #


def test_one_centre_is_one_plot(flat_tile):
    one = als.plot_metrics(flat_tile, (20.0, 20.0), radius=6.0, metrics=["n", "zmax"])
    many = als.plot_metrics(flat_tile, [(20.0, 20.0)], radius=6.0, metrics=["n", "zmax"])
    assert len(one) == 1
    np.testing.assert_array_equal(one["n"], many["n"])
    # Heights above the flat ground at z = 10: the vegetation reaches 15 m.
    cloud = io.read(flat_tile.paths[0])
    inside = (cloud.x - 20) ** 2 + (cloud.y - 20) ** 2 <= 36
    assert one["n"][0] == inside.sum()
    assert one["zmax"][0] == pytest.approx(cloud.z[inside].max() - 10.0, abs=1e-3)


def test_to_pandas_is_the_table(flat_tile):
    pd = pytest.importorskip("pandas")
    t = als.plot_metrics(flat_tile, [(10.0, 10.0), (30.0, 25.0)], radius=5.0,
                         metrics=["n", "zmean"], ids=["a", "b"])
    df = t.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["plot", "id", "n", "zmean"]
    np.testing.assert_array_equal(df["zmean"].to_numpy(), t["zmean"])
    assert list(df["id"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #


def _cli(*argv):
    cli.main(["--no-progress", *map(str, argv)])


def _fails(capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exc:
        _cli(*argv)
    assert exc.value.code == 1
    return capsys.readouterr().err


def test_plot_csv_radius_column_and_errors(flat_tile, tmp_path, capsys):
    d = flat_tile.paths[0].rsplit("/", 1)[0]
    plots = tmp_path / "plots.csv"
    plots.write_text("X,Y,Radius,name\n10,10,3,p1\n30,30,6,p2\n")
    _cli("als-plot-metrics", d, plots, tmp_path / "t.csv", "--metrics", "n", "--id-field", "name")
    rows = (tmp_path / "t.csv").read_text().splitlines()
    want = als.plot_metrics(flat_tile, [[10, 10], [30, 30]], radius=[3.0, 6.0], metrics=["n"])
    assert rows == ["plot,id,n", f"0,p1,{want['n'][0]:.1f}", f"1,p2,{want['n'][1]:.1f}"]
    capsys.readouterr()
    cases = [
        ("", f"{tmp_path / 'bad.csv'} has no plots"),
        ("x,z\n1,2\n", f"{tmp_path / 'bad.csv'} needs columns x and y, found x, z"),
        ("x,y\n1,2\n", f"{tmp_path / 'bad.csv'} has no radius column; give --radius"),
    ]
    for text, message in cases:
        (tmp_path / "bad.csv").write_text(text)
        err = _fails(capsys, "als-plot-metrics", d, tmp_path / "bad.csv", tmp_path / "t.csv")
        assert err == f"sylva: error: {message}\n"
    (tmp_path / "bad.csv").write_text("x,y\n1,2\n")
    err = _fails(capsys, "als-plot-metrics", d, tmp_path / "bad.csv", tmp_path / "t.csv",
                 "--radius", 2, "--id-field", "plot_name")
    assert err == f"sylva: error: {tmp_path / 'bad.csv'} has no 'plot_name' column\n"


def test_plot_polygons_with_ids(flat_tile, tmp_path, capsys):
    d = flat_tile.paths[0].rsplit("/", 1)[0]
    square = [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0], [5.0, 5.0]]
    features = [{"type": "Feature", "properties": {"name": name},
                 "geometry": {"type": "Polygon", "coordinates": [[[x + dx, y] for x, y in square]]}}
                for name, dx in (("west", 0.0), ("east", 20.0))]
    path = tmp_path / "plots.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    _cli("als-plot-metrics", d, path, tmp_path / "t.csv", "--metrics", "n", "--id-field", "name")
    rows = (tmp_path / "t.csv").read_text().splitlines()
    cloud = io.read(flat_tile.paths[0])
    for row, x0 in zip(rows[1:], (5.0, 25.0), strict=True):
        inside = (cloud.x >= x0) & (cloud.x <= x0 + 10) & (cloud.y >= 5) & (cloud.y <= 15)
        assert row.split(",")[1:] == ["west" if x0 == 5 else "east", f"{inside.sum():.1f}"]
    capsys.readouterr()
    err = _fails(capsys, "als-plot-metrics", d, path, tmp_path / "t.csv", "--id-field", "code")
    assert err == "sylva: error: 2 plot(s) have no 'code' attribute\n"


def test_metrics_heights_from_an_attribute_or_a_dtm_file(flat_tile, tmp_path):
    d = flat_tile.paths[0].rsplit("/", 1)[0]
    _cli("als-metrics", d, tmp_path / "attr", "--resolution", 20, "--metrics", "zmax,n",
         "--height-attribute", "height")
    dtm = Raster(np.full((5, 5), 10.0), -5.0, -5.0, 10.0)
    dtm.to_ascii_grid(tmp_path / "dtm.asc")
    _cli("als-metrics", d, tmp_path / "file", "--resolution", 20, "--metrics", "zmax,n",
         "--dtm", tmp_path / "dtm.asc")
    cloud = io.read(flat_tile.paths[0])
    for sub in ("attr", "file"):
        zmax = Raster.from_ascii_grid(tmp_path / sub / "zmax.asc")
        assert zmax.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                cell = ((cloud.x >= 20 * j) & (cloud.x < 20 * (j + 1))
                        & (cloud.y >= 20 * i) & (cloud.y < 20 * (i + 1)))
                assert zmax.data[i, j] == pytest.approx(cloud.z[cell].max() - 10.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# negative heights and noise
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def noisy_tile(tmp_path_factory):
    """Ground returns just under the DTM (height -5 to 0 cm), vegetation 0-15 m,
    and a few noise returns (class 7) 60 m up."""
    rng = np.random.default_rng(23)
    d = tmp_path_factory.mktemp("noisy")
    n = 3000
    xy = rng.uniform(0, 40, (3 * n + 20, 2))
    h = np.concatenate([rng.uniform(-0.05, 0.0, n), rng.uniform(0, 15, 2 * n), np.full(20, 60.0)])
    cls = np.concatenate([np.full(n, 2), np.full(2 * n, 5), np.full(20, 7)]).astype(np.uint8)
    io.write(PointCloud(np.column_stack([xy, 10.0 + h]),
                        {"classification": cls, "height": h}), d / "t.las")
    return als.catalog(d)


def test_zentropy_warns_until_negative_heights_are_clamped(noisy_tile):
    with pytest.warns(UserWarning, match="clamp_negative=True"):
        lidr = als.grid_metrics(noisy_tile, 10.0, ["n", "zentropy"], dtm="height")
    assert np.isnan(lidr["zentropy"].data).all()
    clamped = als.grid_metrics(noisy_tile, 10.0, ["n", "zentropy", "zq5"], dtm="height",
                               clamp_negative=True)
    assert np.isfinite(clamped["zentropy"].data).all()
    np.testing.assert_array_equal(clamped["n"].data, lidr["n"].data)   # kept, as ground
    assert np.nanmin(clamped["zq5"].data) == 0.0
    dropped = als.grid_metrics(noisy_tile, 10.0, ["n", "zentropy"], dtm="height", min_height=0.0)
    assert (dropped["n"].data < lidr["n"].data).all()
    one = als.cloud_metrics(io.read(noisy_tile.paths[0]), height="height", clamp_negative=True)
    assert np.isfinite(one["zentropy"])


def test_chm_and_gap_profile_leave_out_noise(noisy_tile):
    dtm = Raster(np.full((1, 1), 10.0), 0.0, 0.0, 100.0)
    assert np.nanmax(als.chm(noisy_tile, 2.0, dtm=dtm).data) <= 15.0
    noisy = als.chm(noisy_tile, 2.0, dtm=dtm, drop_noise=False)
    assert np.nanmax(noisy.data) == pytest.approx(60.0, abs=0.01)
    cloud = io.read(noisy_tile.paths[0])
    heights = PointCloud(np.column_stack([cloud.x, cloud.y, cloud.attrs["height"]]),
                         dict(cloud.attrs))
    kw = {"resolution": 20.0, "angles": "none", "dtm": None, "weighting": "all"}
    clean = als.gap_profile(heights, **kw)
    assert clean.weight.shape[0] - 2 <= 15
    kept = als.gap_profile(heights, drop_noise=False, top_quantile=1.0, **kw)
    assert kept.weight.shape[0] - 2 == 60
