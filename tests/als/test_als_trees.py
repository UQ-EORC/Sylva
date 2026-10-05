# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Tree tops, crowns and labelled tiles from airborne lidar (sylva.als)."""

import json
import warnings

import numpy as np
import pytest

from sylva import PointCloud, Raster, als, cli, synthetic


def cones(res=0.5):
    """Two cones, 10 and 8 m high, 12 m apart; returns the CHM and the true heights."""
    n_r, n_c = int(20 / res), int(30 / res)
    y, x = (np.mgrid[0:n_r, 0:n_c] + 0.5) * res
    a = 10.0 - 1.2 * np.hypot(x - 10.25, y - 10.25)
    b = 8.0 - 1.2 * np.hypot(x - 22.25, y - 10.25)
    return Raster(np.maximum(np.maximum(a, b), 0.0), 0.0, 0.0, res), a, b


# ---------------------------------------------------------------- tree tops

def test_local_maxima_on_a_chm():
    chm, _, _ = cones()
    t = als.locate_trees(chm, window=3.0)
    assert np.allclose(t.x, [10.25, 22.25]) and np.allclose(t.y, [10.25, 10.25])
    assert np.allclose(t.height, [10.0, 8.0])
    assert t.xyz.shape == (2, 3) and len(t) == 2
    # A window wider than the spacing keeps only the taller; so does hmin.
    assert len(als.locate_trees(chm, window=30.0, shape="square")) == 1
    assert len(als.locate_trees(chm, window=3.0, hmin=9.0)) == 1
    # Height-dependent windows: a function, and the same as a LinearWindow.
    w = als.LinearWindow(-91.0, 11.5, 1.0, 30.0)   # 1 m at 8 m, 24 m at 10 m
    assert float(w(8.0)) == 1.0 and float(w(10.0)) == 24.0
    assert len(als.locate_trees(chm, window=w)) == 2
    assert len(als.locate_trees(chm, window=lambda h: np.clip(-91 + 11.5 * h, 1, 30))) == 2
    assert len(als.locate_trees(chm, window=lambda h: 30.0)) == 1


def test_equal_maxima_keep_the_first_in_x():
    d = np.zeros((3, 4))
    d[1, 1] = d[1, 2] = 5.0
    t = als.locate_trees(Raster(d, 0.0, 0.0, 1.0), window=3.0)
    assert list(t.x) == [1.5]
    cloud = PointCloud(np.array([[1.0, 1.0, 5.0], [0.0, 1.0, 5.0], [5.0, 5.0, 1.0]]))
    assert list(als.locate_trees(cloud, window=3.0).index) == [1]
    assert list(als.locate_trees(cloud, window=1.5).index) == [1, 0]


def test_point_maxima_match_brute_force():
    rng = np.random.default_rng(4)
    xyz = np.column_stack([rng.uniform(0, 30, (2000, 2)), rng.uniform(0, 30, 2000)])
    cloud = PointCloud(xyz)
    for window, shape in [(2.5, "circular"), (als.LinearWindow(0.5, 0.2, 1.0, 6.0), "square")]:
        got = als.locate_trees(cloud, window=window, shape=shape).index
        ws = np.full(2000, window) if isinstance(window, float) else window(xyz[:, 2])
        want = []
        for i in range(2000):
            if xyz[i, 2] < 2.0:
                continue
            d = xyz[:, :2] - xyz[i, :2]
            r = ws[i] / 2
            inside = (np.hypot(d[:, 0], d[:, 1]) <= r) if shape == "circular" else \
                (np.abs(d) <= r).all(axis=1)
            if not (inside & (xyz[:, 2] > xyz[i, 2])).any():
                want.append(i)
        want = sorted(want, key=lambda i: xyz[i, 0])
        assert list(got) == want


def test_heights_from_an_attribute_and_bad_input():
    cloud = PointCloud(np.array([[0.0, 0.0, 100.0], [3.0, 0.0, 100.0]]),
                       {"height": np.array([10.0, 5.0])})
    t = als.locate_trees(cloud, window=10.0, heights="height")
    assert list(t.index) == [0] and t.height[0] == 10.0
    with pytest.raises(ValueError, match="positive"):
        als.locate_trees(cloud, window=0.0)
    with pytest.raises(ValueError, match="positive"):
        als.locate_trees(cloud, window=lambda h: -h)
    with pytest.raises(ValueError, match="shape"):
        als.locate_trees(cloud, shape="hexagon")
    with pytest.raises(ValueError, match="attribute"):
        als.locate_trees(cloud, heights="nope")
    with pytest.raises(ValueError, match="one value per point"):
        als.locate_trees(cloud, heights=[1.0])
    with pytest.raises(ValueError, match="Raster"):
        als.locate_trees(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="min <= max"):
        als.locate_trees(cloud, window=als.LinearWindow(1.0, 0.1, 5.0, 2.0))
    # Empty and NaN input.
    assert len(als.locate_trees(PointCloud(np.zeros((0, 3))))) == 0
    nan = PointCloud(np.array([[0.0, 0.0, np.nan], [1.0, 1.0, 5.0]]))
    assert list(als.locate_trees(nan).index) == [1]
    assert len(als.locate_trees(Raster(np.full((4, 4), np.nan), 0, 0, 1.0))) == 0


# ---------------------------------------------------------------- crowns on a CHM

def test_watershed_splits_at_the_valley():
    chm, a, b = cones()
    crowns = als.segment_crowns(chm, als.locate_trees(chm, window=3.0), method="watershed")
    want = np.where(np.maximum(a, b) <= 2.0, np.nan, np.where(a >= b, 1.0, 2.0))
    clear = (np.abs(a - b) > 0.8) | np.isnan(want)
    np.testing.assert_array_equal(crowns.data[clear], want[clear])
    assert crowns.resolution == chm.resolution and crowns.xmin == chm.xmin


def test_dalponte_thresholds_and_max_cr():
    chm, _, _ = cones()
    tops = np.array([[10.25, 10.25, 10.0], [22.25, 10.25, 8.0]])
    crowns = als.segment_crowns(chm, tops, max_cr=100)
    for k, seed_h in ((1, 10.0), (2, 8.0)):
        z = chm.data[crowns.data == k]
        assert z.min() > 0.45 * seed_h and z.min() > 2.0 and z.max() <= 1.05 * seed_h
    small = als.segment_crowns(chm, tops, max_cr=3)
    rows, cols = np.nonzero(small.data == 1)
    assert np.all(np.abs(rows - 20) < 3) and np.all(np.abs(cols - 20) < 3)
    with pytest.raises(ValueError, match="between 0 and 1"):
        als.segment_crowns(chm, tops, th_seed=1.5)
    with pytest.raises(ValueError, match="unknown method"):
        als.segment_crowns(chm, tops, method="li2012")
    with pytest.raises(ValueError, match="tops"):
        als.segment_crowns(chm, np.zeros((2, 2)))
    assert np.isnan(als.segment_crowns(chm, np.zeros((0, 3))).data).all()


# ---------------------------------------------------------------- points

def two_clusters(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    cx, top = np.where(k % 2 == 0, 0.0, 8.0), np.where(k % 2 == 0, 20.0, 16.0)
    r, a = 3.0 * np.sqrt(rng.uniform(size=n)), rng.uniform(0, 2 * np.pi, n)
    return PointCloud(np.column_stack([cx + r * np.cos(a), r * np.sin(a), top - 2 * r]))


def test_li2012_separates_two_clusters():
    cloud = two_clusters()
    ids = als.li2012(cloud)
    np.testing.assert_array_equal(ids, np.where(np.arange(len(cloud)) % 2 == 0, 1, 2))
    # hmin above everything: nothing.
    assert (als.li2012(cloud, hmin=50.0) == 0).all()
    with pytest.raises(ValueError, match="positive"):
        als.li2012(cloud, dt1=0.0)


def test_crown_hulls():
    x, y = np.meshgrid(np.arange(0, 10.01, 0.5), np.arange(0, 10.01, 0.5))
    keep = ~((x > 3) & (x < 7) & (y > 4))
    xy = np.column_stack([x[keep], y[keep]])

    def area(r):
        x, y = r[:, 0], r[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    assert area(als.crown_hull(xy)) == pytest.approx(100.0)
    concave = area(als.crown_hull(xy, "concave", 1.0))
    assert 70.0 < concave < 85.0
    assert als.crown_hull(xy[:2], "concave").shape == (2, 2)
    with pytest.raises(ValueError, match="hull"):
        als.crown_hull(xy, "alpha")
    with pytest.raises(ValueError, match="concavity"):
        als.crown_hull(xy, "concave", 0.0)


# ---------------------------------------------------------------- whole segmentations

@pytest.fixture(scope="module")
def stand():
    """Ten well-spaced trees with solid crowns, flown at about 30 points/m²."""
    trees = [(x + 6.0, y + 6.0, d, h) for x, y, d, h in
             synthetic.stand(10, size=48.0, min_spacing=15.0, heights=(12, 22), seed=2)]
    scene = synthetic.crown_forest(trees, size=60.0, ground_points=100, margin=0.0, seed=2)
    truth = synthetic.forest_trees(scene)
    flight = synthetic.als_flight(scene, pulse_rate=6_000, line_spacing=30.0,
                                  bounds=(0, 0, 60, 60), seed=2)
    p = flight.points
    norm = PointCloud(np.column_stack([p.x, p.y, p.z - synthetic.terrain_height(p.x, p.y)]),
                      dict(p.attrs))
    return trees, truth, flight, norm


def test_synthetic_truth(stand):
    trees, truth, _, _ = stand
    assert len(trees) == 10 and list(truth["tree_id"]) == list(range(1, 11))
    for t, x, y, h, a in zip(trees, truth["stem_x"], truth["stem_y"], truth["height"],
                             truth["crown_area"], strict=True):
        assert abs(x - t[0]) < 0.05 and abs(y - t[1]) < 0.05
        assert t[3] - 0.6 < h <= t[3] + 0.1
        assert 0.85 < a / (np.pi * (0.25 * t[3]) ** 2) < 1.0
    assert len(truth["crowns"]) == 10
    with pytest.raises(ValueError, match="could not place"):
        synthetic.stand(100, size=10.0, min_spacing=5.0)
    with pytest.raises(ValueError, match="crown shape"):
        synthetic.crown_forest(trees, shape="sphere")
    with pytest.raises(ValueError, match="tree_id"):
        synthetic.forest_trees(PointCloud(np.zeros((3, 3))))


@pytest.mark.parametrize("method", ["dalponte2016", "watershed", "li2012"])
def test_segment_trees_finds_every_tree(stand, method):
    trees, _, _, norm = stand
    tr = als.segment_trees(norm, method=method, window=als.LinearWindow(0, 0.3, 2, 20), max_cr=20,
                           dt1=3.0, dt2=3.5)
    assert len(tr) == 10, tr
    true_xy = np.array([t[:2] for t in trees])
    true_h = np.array([t[3] for t in trees])
    for k in range(10):
        j = np.argmin(np.hypot(*(true_xy - [tr.x[k], tr.y[k]]).T))
        assert np.hypot(*(true_xy[j] - [tr.x[k], tr.y[k]])) < 1.0
        assert abs(tr.height[k] - true_h[j]) < 0.5
        assert abs(tr.crown_area[k] / (np.pi * (0.25 * true_h[j]) ** 2) - 1) < 0.2
    # Points: each tree's returns carry one id, and that tree's alone.
    tid = norm.attrs["tree_id"]
    veg = (norm.z >= 0.5) & (tid > 0)
    agree = np.mean([np.bincount(tr.tree_id[veg & (tid == i)]).argmax() > 0 for i in range(1, 11)])
    assert agree == 1.0
    ids = tr.tree_id[veg]
    pairs = set(zip(tid[veg].tolist(), ids.tolist(), strict=True))
    assert len({p for p in pairs if p[1] > 0}) <= 12
    assert list(tr.id) == list(range(1, 11)) and np.all(np.diff(tr.x) >= 0)
    assert all(c.shape[1] == 2 for c in tr.crowns) and tr.n_points.sum() == (tr.tree_id > 0).sum()


def test_segment_trees_edge_cases(stand):
    *_, norm = stand
    tops = als.segment_trees(norm, method="tops", window=6.0)
    assert np.isnan(tops.crown_area).all() and (tops.tree_id == 0).all()
    concave = als.segment_trees(norm, hull="concave", concavity=1.5)
    convex = als.segment_trees(norm)
    assert np.all(concave.crown_area <= convex.crown_area + 1e-9)
    empty = als.segment_trees(PointCloud(np.zeros((0, 3))))
    assert len(empty) == 0 and len(empty.tree_id) == 0
    with pytest.raises(ValueError, match="unknown method"):
        als.segment_trees(norm, method="silva2016")
    with pytest.raises(ValueError, match="tops_from"):
        als.segment_trees(norm, tops_from="lidar")
    with pytest.raises(ValueError, match="smooth"):
        als.segment_trees(norm, smooth=-1)
    with pytest.raises(ValueError, match="resolution"):
        als.segment_trees(norm, resolution=0.0)
    smoothed = als.segment_trees(norm, smooth=1)
    assert len(smoothed) > 0


# ---------------------------------------------------------------- catalogues

@pytest.fixture(scope="module")
def tiles(stand, tmp_path_factory):
    *_, norm = stand
    d = tmp_path_factory.mktemp("trees")
    return als.write_tiles(norm, d / "norm", 20.0, origin=(0.0, 0.0), epsg=32755)


@pytest.mark.parametrize("method", ["dalponte2016", "li2012"])
def test_a_catalogue_gives_the_trees_of_the_merged_tiles(tiles, tmp_path, method):
    merged = tiles.read()
    whole = als.segment_trees(merged, method=method, window=als.LinearWindow(0, 0.3, 2, 20),
                              max_cr=20, dt1=3.0, dt2=3.5)
    key = {k: i for i, k in enumerate(zip(*(np.round(v, 6) for v in (
        merged.x, merged.y, merged.z, merged.attrs["gps_time"])), strict=True))}
    for chunk_size, workers in [(None, 1), (25.0, 3)]:
        out = tmp_path / f"{method}-{chunk_size}"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = als.find_trees(tiles, out=out, method=method, dtm=None, chunk_size=chunk_size,
                                 workers=workers, window=als.LinearWindow(0, 0.3, 2, 20),
                                 max_cr=20, dt1=3.0, dt2=3.5, attribute="als_tree",
                                 format="las")
        for c in ("id", "x", "y", "height", "crown_area", "n_points"):
            np.testing.assert_array_equal(getattr(got, c), getattr(whole, c))
        for a, b in zip(got.crowns, whole.crowns, strict=True):
            np.testing.assert_array_equal(a, b)
        back = got.catalog.read()
        assert len(back) == len(merged) and got.unmatched_points == 0 and got.at_edge == 0
        idx = np.array([key[k] for k in zip(*(np.round(v, 6) for v in (
            back.x, back.y, back.z, back.attrs["gps_time"])), strict=True)])
        np.testing.assert_array_equal(back.attrs["als_tree"], whole.tree_id[idx])
        assert got.crs == "EPSG:32755"


def test_catalogue_outputs_and_errors(tiles, tmp_path):
    t = als.find_trees(tiles, method="tops", dtm=None, window=lambda h: 0.3 * h + 1)
    assert len(t) == 10 and t.catalog is None
    t.to_csv(tmp_path / "t.csv")
    rows = (tmp_path / "t.csv").read_text().splitlines()
    assert rows[0] == "id,x,y,height,crown_area,n_points" and len(rows) == 11
    t.to_geojson(tmp_path / "tops.geojson", geometry="tops")
    fc = json.loads((tmp_path / "tops.geojson").read_text())
    assert len(fc["features"]) == 10 and fc["features"][0]["geometry"]["type"] == "Point"
    c = als.find_trees(tiles, dtm=None, window=6.0)
    c.to_geojson(tmp_path / "crowns.geojson")
    fc = json.loads((tmp_path / "crowns.geojson").read_text())
    ring = fc["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and "EPSG::32755" in fc["crs"]["properties"]["name"]
    with pytest.raises(ValueError, match="need crowns"):
        als.find_trees(tiles, out=tmp_path / "x", method="tops", dtm=None)
    with pytest.raises(ValueError, match="geometry"):
        c.to_geojson(tmp_path / "x.geojson", geometry="stems")
    # A buffer too narrow for the crowns is reported.
    with pytest.warns(UserWarning, match="wider buffer"):
        als.find_trees(tiles, dtm=None, window=6.0, buffer=1.0)


def test_heights_from_ground(stand, tmp_path):
    _, _, flight, norm = stand
    cat = flight.write_tiles(tmp_path / "raw", size=30.0, epsg=32755)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = als.find_trees(cat, out=tmp_path / "lab", window=als.LinearWindow(0, 0.3, 2, 20),
                           max_cr=20)
    assert t.unmatched_points == 0 and t.catalog.n_points == cat.n_points
    want = als.segment_trees(norm, window=als.LinearWindow(0, 0.3, 2, 20), max_cr=20)
    assert len(t) == len(want) == 10
    np.testing.assert_allclose(t.height, want.height, atol=0.15)


def test_command_line(tiles, tmp_path, capsys):
    d = str(tiles.paths[0]).rsplit("/", 1)[0]
    cli.main(["--no-progress", "als-trees", d, str(tmp_path / "trees.csv"), "--normalized",
              "--window-linear", "0", "0.3", "2", "20", "--max-cr", "20",
              "--crowns", str(tmp_path / "crowns.geojson"), "--labelled", str(tmp_path / "lab"),
              "--workers", "2"])
    out = capsys.readouterr().out
    assert "10 trees" in out
    rows = (tmp_path / "trees.csv").read_text().splitlines()
    want = als.find_trees(tiles, dtm=None, window=als.LinearWindow(0, 0.3, 2, 20), max_cr=20)
    assert len(rows) == 11
    np.testing.assert_allclose([float(r.split(",")[3]) for r in rows[1:]], want.height)
    lab = als.catalog(tmp_path / "lab")
    assert lab.n_points == tiles.n_points and "tree_id" in lab.read().attrs
    assert len(json.loads((tmp_path / "crowns.geojson").read_text())["features"]) == 10
    cli.main(["--no-progress", "als-trees", d, str(tmp_path / "tops.csv"), "--normalized",
              "--method", "tops", "--window", "6"])
    assert len((tmp_path / "tops.csv").read_text().splitlines()) == 11
