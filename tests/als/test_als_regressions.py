"""Regressions in sylva.als: the tiled plot workflow, tree stores, plot tables and the CLI."""

import pickle
import threading
import time
import warnings

import numpy as np
import pytest

from sylva import PointCloud, als, synthetic
from sylva.als import tiles, tiles_trees
from sylva.als.metrics import _read_plot_csv
from sylva.cli import main


def _scans():
    """A 20 m synthetic plot of eight trees scanned from four positions."""
    rng = np.random.default_rng(3)
    rows = [(x, y, d, h) for x, y, d, h in zip(rng.uniform(3, 17, 8), rng.uniform(3, 17, 8),
                                               rng.uniform(0.2, 0.5, 8), rng.uniform(12, 20, 8),
                                               strict=True)]
    scene = synthetic.forest(rows, size=20.0, ground_points=int(70 * 400), margin=2.0, seed=3)
    return [synthetic.scan(scene, origin=(x, y, 1.5), resolution_deg=0.3).to_pointcloud()
            for x, y in [(4, 4), (16, 16), (4, 16), (16, 4)]]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A small tree store: three clumps of points of different sizes, labelled 1 to 3."""
    rng = np.random.default_rng(0)
    parts, labels = [], []
    for tree_id, (n, x) in enumerate([(300, 2.0), (900, 5.0), (600, 8.0)], start=1):
        parts.append(np.column_stack([rng.normal(x, 0.3, n), rng.normal(5.0, 0.3, n),
                                      rng.uniform(0.0, 6.0, n)]))
        labels.append(np.full(n, tree_id, dtype=np.int32))
    xyz = np.vstack(parts)
    cloud = PointCloud(xyz, {"tree_id": np.concatenate(labels), "height": xyz[:, 2].copy()})
    d = tmp_path_factory.mktemp("store")
    cat = als.write_tiles(cloud, d / "tiles", 10.0)
    return tiles.split_trees(cat, d / "trees"), cat


# ------------------------------------------------------------------ run_plot


def test_run_plot_segments_over_the_margin_and_keeps_the_trees_of_the_plot(tmp_path):
    """The trees of a plot are those the segmentation of everything inside
    the bounds gives, picked by where their stem stands: an edge tree keeps
    the crown that hangs over the plot's edge."""
    plot = (0.0, 0.0, 10.0, 20.0)
    kw = dict(bounds=(-2, -2, 22, 22), tile_size=10.0, voxel_size=0.02, sor={"k": 6, "std_ratio": 2.0},
              ground={"method": "pmf", "buffer": 5.0}, stems={}, buffer=10.0, workers=2, log=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = tiles.run_plot(_scans(), tmp_path / "run", plot=plot, leaf_wood=False,
                             qsm_options={"stem_radius_cap": 1.5, "min_points": 500}, **kw)
        heights = run.catalogs["heights"]
        stems = tiles.detect_stems(heights, workers=2)
        whole, _ = tiles.segment_trees(heights, stems, tmp_path / "seg", buffer=10.0, merge={"voxel_size": 0.1},
                                       percentile=99.0, prune={"min_height": 2.0}, workers=2, voxel_size=0.1)
    inside = [t for t in whole if plot[0] <= t.x <= plot[2] and plot[1] <= t.y <= plot[3]]
    assert 0 < len(inside) < len(whole)

    def rows(ts):
        return [(t.tree_id, t.x, t.y, t.dbh, t.height, t.n_points) for t in ts]

    assert rows(run.trees) == rows(inside)
    assert [r["tree_id"] for r in run.table] == [t.tree_id for t in inside]
    # A crown reaches over the plot's edge, into tiles that are not the plot's.
    assert max(run.store.bounds(t.tree_id)[3] for t in run.trees) > plot[2] + 0.5
    # The trees of the margin are segmented and stored, but not modelled.
    kept = {t.tree_id for t in run.trees}
    assert set(run.store.ids) > kept
    assert set(run.qsms.models) | set(run.qsms.skipped) <= kept
    for i in set(run.store.ids) - kept:
        assert not (run.store.path / f"tree_{i}" / "qsm.pkl").exists()


def test_run_plot_takes_an_iterator_of_scans(tmp_path, monkeypatch):
    given = []

    class Reached(Exception):
        pass

    def from_scans(scans, *a, **k):
        given.extend(scans)
        raise Reached

    monkeypatch.setattr(tiles, "from_scans", from_scans)
    a, b = PointCloud(np.zeros((3, 3))), PointCloud(np.ones((3, 3)))
    with pytest.raises(Reached):
        tiles.run_plot(iter([a, b]), tmp_path / "run", log=None)
    assert given == [a, b]


def test_run_plot_refuses_one_file_path(tmp_path):
    with pytest.raises(ValueError, match="list of scan files"):
        tiles.run_plot(str(tmp_path / "scan.laz"), tmp_path / "run", log=None)
    with pytest.raises(ValueError, match="plot must be"):
        tiles.run_plot([PointCloud(np.zeros((3, 3)))], tmp_path / "bad", plot=(0, 0, 0, 5), log=None,
                       sor=False)


# ------------------------------------------------------------------ caches of the store


def _fake_qsm(calls):
    def fit(st, tree_id, *a):
        calls.append(tree_id)
        return {"models": [], "buttresses": [], "skipped": [(tree_id, "test")], "points": [],
                "heights": []}
    return fit


def test_saved_qsms_are_reused_only_with_the_same_settings(store, monkeypatch):
    st, _ = store
    calls = []
    monkeypatch.setattr(tiles_trees, "_qsm_one", _fake_qsm(calls))
    stems = [tiles_trees.Tree(1, 2.0, 5.0, dbh=0.3)]
    tiles.build_qsms(st, stems, stem_radius_cap=1.5, workers=1)
    assert sorted(calls) == [1, 2, 3]
    calls.clear()
    tiles.build_qsms(st, stems, stem_radius_cap=1.5, resume=True, workers=1)
    assert calls == []
    # Another QSM setting: every tree is fitted again, none taken from the store.
    tiles.build_qsms(st, stems, stem_radius_cap=2.0, resume=True, workers=1)
    assert sorted(calls) == [1, 2, 3]
    calls.clear()
    # Another thinning, and another stem for tree 1 only.
    tiles.build_qsms(st, stems, stem_radius_cap=2.0, voxel_size=0.02, resume=True, workers=1)
    assert sorted(calls) == [1, 2, 3]
    calls.clear()
    moved = [tiles_trees.Tree(1, 2.1, 5.0, dbh=0.3)]
    tiles.build_qsms(st, moved, stem_radius_cap=2.0, voxel_size=0.02, resume=True, workers=1)
    assert calls == [1]
    calls.clear()
    # A model saved without its settings (an earlier version) is not trusted.
    with open(st.path / "tree_2" / "qsm.pkl", "wb") as f:
        pickle.dump(_fake_qsm([])(st, 2), f)
    tiles.build_qsms(st, moved, stem_radius_cap=2.0, voxel_size=0.02, resume=True, workers=1)
    assert calls == [2]


def test_stored_leaf_wood_is_reused_only_with_the_same_settings(store, tmp_path, monkeypatch):
    st, cat = store
    calls = []

    def classify(cloud, **kw):
        calls.append(len(cloud))
        return np.zeros(len(cloud), dtype=bool)

    monkeypatch.setattr(tiles_trees.leaves, "classify_leaf_wood", classify)
    tiles.classify_leaf_wood(st, cat, tmp_path / "a", name="lw", voxel_size=0.02, workers=1)
    assert len(calls) == 3
    calls.clear()
    tiles.classify_leaf_wood(st, cat, tmp_path / "b", name="lw", voxel_size=0.02, resume=True, workers=1)
    assert calls == []
    tiles.classify_leaf_wood(st, cat, tmp_path / "c", name="lw", voxel_size=0.05, resume=True, workers=1)
    assert len(calls) == 3
    calls.clear()
    tiles.classify_leaf_wood(st, cat, tmp_path / "d", name="lw", voxel_size=0.05, method="passage",
                             resume=True, workers=1)
    assert len(calls) == 3
    # Only the trees asked for are classified; the others are written as -1.
    calls.clear()
    out = tiles.classify_leaf_wood(st, cat, tmp_path / "e", name="other", ids=[2], workers=1).read()
    assert calls == [900]
    got = {int(t): set(np.unique(out.attrs["other"][out.attrs["tree_id"] == t]).tolist()) for t in (1, 2, 3)}
    assert got == {1: {-1}, 2: {0}, 3: {-1}}


# ------------------------------------------------------------------ trees at once


def test_trees_are_taken_largest_first(store):
    st, _ = store
    order = []
    st.map(lambda i, cloud: order.append((i, len(cloud))), workers=1)
    assert order == [(2, 900), (3, 600), (1, 300)]
    assert list(st.map(lambda i, cloud: len(cloud), workers=1)) == [1, 2, 3]  # results in id order


def test_a_failing_tree_stops_the_trees_not_started():
    started = []
    lock = threading.Lock()

    def job(i):
        with lock:
            started.append(i)
        if i == 0:
            raise RuntimeError("tree 0 failed")
        time.sleep(0.05)
        return i

    with pytest.raises(RuntimeError, match="tree 0 failed"):
        tiles_trees._map_trees(list(range(40)), job, {i: 40 - i for i in range(40)}, 2, "test")
    assert len(started) < 20, started


# ------------------------------------------------------------------ tables


@pytest.mark.parametrize("header, radius, want_ids, want_radius", [
    ("id,x,y,radius", None, ["A", "B"], [5.0, 7.0]),
    ("radius,x,y,id", 11.28, ["A", "B"], [5.0, 7.0]),
])
def test_plot_csv_from_a_spreadsheet_keeps_its_first_column(tmp_path, header, radius, want_ids,
                                                            want_radius):
    """A byte order mark before the first column name ("CSV UTF-8") must not hide the column."""
    names = header.split(",")
    rows = [{"id": "A", "x": "1", "y": "2", "radius": "5"}, {"id": "B", "x": "3", "y": "4", "radius": "7"}]
    path = tmp_path / "plots with bom.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(header + "\n" + "".join(",".join(r[n] for n in names) + "\n" for r in rows))
    xy, rad, ids = _read_plot_csv(str(path), radius, None)
    np.testing.assert_array_equal(xy, [[1.0, 2.0], [3.0, 4.0]])
    assert ids == want_ids
    np.testing.assert_array_equal(rad, want_radius)


def test_tiles_stems_writes_the_header_without_stems(tmp_path, capsys):
    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(0, 10, 5000), rng.uniform(0, 10, 5000), rng.normal(0, 0.02, 5000)])
    als.write_tiles(PointCloud(xyz, {"height": xyz[:, 2].copy()}), tmp_path / "tiles", 10.0)
    out = tmp_path / "stems.csv"
    main(["--no-progress", "tiles-stems", str(tmp_path / "tiles"), str(out)])
    assert capsys.readouterr().out.startswith("0 stems")
    lines = out.read_text().splitlines()
    assert len(lines) == 1 and lines[0].split(",")[:4] == ["tree_id", "x", "y", "dbh"]


# ------------------------------------------------------------------ trajectory


def test_week_seconds_of_an_array_is_that_of_each_time():
    t = np.concatenate([np.random.default_rng(0).uniform(-2e9, 2e9, 1998), [0.0, -1e9, 604800.0 - 1e9]])
    got = als.week_seconds(t)
    assert got.shape == t.shape
    np.testing.assert_array_equal(got, [als.week_seconds(float(v)) for v in t])
    assert isinstance(als.week_seconds(3.5e8), float)
    assert als.week_seconds(t.reshape(-1, 3)).shape == (len(t) // 3, 3)
    # A whole survey's times in well under a second.
    start = time.perf_counter()
    als.week_seconds(np.linspace(3e8, 3.1e8, 5_000_000))
    assert time.perf_counter() - start < 2.0
