"""Tiled tree segmentation, tree stores and per-tree models against the whole plot."""

import pickle
import warnings

import numpy as np
import pytest

from sylva import PointCloud, als, ground, io, leaves, qsm, synthetic, trees
from sylva.als import tiles, tiles_trees
from sylva.cli import main

POSITIONS = [(5, 5), (15, 15), (25, 5), (5, 25), (25, 25), (15, 3), (3, 15)]
ORIGIN = (0.0, 0.0, 0.0)
MERGE = {"voxel_size": 0.1}
SEGMENT = {"voxel_size": 0.1}
PRUNE = {"min_height": 2.0, "min_slenderness": 10.0}


def _scene(size=30.0, n_trees=12, seed=1):
    rng = np.random.default_rng(seed)
    rows = [(x, y, d, h) for x, y, d, h in zip(rng.uniform(3, size - 3, n_trees),
                                               rng.uniform(3, size - 3, n_trees),
                                               rng.uniform(0.2, 0.5, n_trees),
                                               rng.uniform(12, 20, n_trees), strict=True)]
    return synthetic.forest(rows, size=size, ground_points=int(70 * size * size), margin=2.0,
                            seed=seed)


def _scans(scene, positions, resolution=0.25):
    out = []
    for i, (x, y) in enumerate(positions):
        pc = synthetic.scan(scene, origin=(x, y, 1.5), resolution_deg=resolution).to_pointcloud()
        pc = PointCloud(pc.xyz, {"truth": np.asarray(pc.attrs["tree_id"], dtype=np.int32)})
        out.append(pc.with_attrs(pid=np.arange(len(pc), dtype=np.int64) + 10_000_000 * i))
    return out


def _by_pid(cloud, name):
    order = np.argsort(cloud.attrs["pid"])
    return np.asarray(cloud.attrs[name])[order]


def _whole(cloud, stems, merge=MERGE, segment=SEGMENT, prune=PRUNE, percentile=99.0):
    """The whole-plot sequence on one cloud, on the grid the tiles use."""
    kept, _ = trees.merge_branches(cloud, stems, voxel_origin=ORIGIN, **merge)
    labels = trees.segment_trees(cloud, kept, voxel_origin=ORIGIN, **segment)
    trees.tree_heights(cloud, labels, kept, percentile=percentile)
    return trees.prune_trees(kept, labels, **prune)


def _rows(ts):
    def v(x):
        return None if isinstance(x, float) and np.isnan(x) else x
    return [tuple(v(x) for x in (t.tree_id, t.x, t.y, t.dbh, t.height, t.n_points, t.quality)) for t in ts]


@pytest.fixture(scope="module")
def plot(tmp_path_factory):
    """A 30 m synthetic plot scanned from seven positions, in 10 m tiles with heights and stems."""
    d = tmp_path_factory.mktemp("tree_tiles")
    scans = _scans(_scene(), POSITIONS)
    cat = tiles.from_scans(scans, d / "tiles", tile_size=10.0, voxel_size=0.02, scale=0.0001)
    g = tiles.classify_ground(cat, d / "ground", method="pmf", buffer=6.0)
    h = tiles.normalize(g, d / "heights", dtm_resolution=0.5, buffer=5.0)
    stems = tiles.detect_stems(h, buffer=2.0)
    whole = h.read()
    ref_trees, ref_labels = _whole(whole, stems)
    seg, seg_cat = tiles.segment_trees(h, stems, d / "seg", merge=MERGE, prune=PRUNE, percentile=99.0,
                                       **SEGMENT)
    return {"dir": d, "scans": scans, "cat": h, "stems": stems, "whole": whole,
            "ref": (ref_trees, ref_labels), "seg": (seg, seg_cat)}


# ------------------------------------------------------------------ segmentation


def test_segmentation_matches_the_whole_plot(plot):
    ref_trees, ref_labels = plot["ref"]
    seg, seg_cat = plot["seg"]
    assert len(plot["stems"]) >= 10 and len(ref_trees) >= 8
    assert _rows(seg) == _rows(ref_trees)
    got = seg_cat.read()
    assert got.attrs["tree_id"].dtype == np.int32
    whole = plot["whole"].with_attrs(tree_id=ref_labels)
    np.testing.assert_array_equal(_by_pid(got, "tree_id"), _by_pid(whole, "tree_id"))
    assert not any(t.extra["crown_at_edge"] for t in seg)
    # The trees are the ones planted: most of each tree's points are its own.
    truth = got.attrs["truth"]
    purity = []
    for t in seg:
        mine = truth[got.attrs["tree_id"] == t.tree_id]
        purity.append(np.mean(mine == np.bincount(mine[mine > 0]).argmax()))
    assert np.mean(purity) > 0.8, purity


@pytest.mark.parametrize("tile_size,workers,buffer", [(7.5, 3, 20.0), (15.0, 8, 12.0), (10.0, 2, 3.0)])
def test_segmentation_does_not_depend_on_tiles_or_workers(plot, tmp_path, tile_size, workers, buffer):
    cat = plot["cat"] if tile_size == 10.0 else als.retile(plot["cat"], tmp_path / "re", tile_size)
    whole = cat.read()
    ref_trees, ref_labels = _whole(whole, plot["stems"])
    seg, seg_cat = tiles.segment_trees(cat, plot["stems"], tmp_path / "seg", buffer=buffer,
                                       max_buffer=40.0, merge=MERGE, prune=PRUNE, percentile=99.0,
                                       workers=workers, **SEGMENT)
    assert _rows(seg) == _rows(ref_trees)
    np.testing.assert_array_equal(_by_pid(seg_cat.read(), "tree_id"),
                                  _by_pid(whole.with_attrs(tree_id=ref_labels), "tree_id"))
    if buffer < 5:
        # Crowns wider than the buffer: tiles were read again, wider.
        assert tiles.last_run().rereads > 0


def test_without_merging_or_pruning(plot, tmp_path):
    whole, stems = plot["whole"], plot["stems"]
    labels = trees.segment_trees(whole, stems, voxel_origin=ORIGIN, k=8, power=4.0, **SEGMENT)
    ref = trees.tree_heights(whole, labels, [trees.Tree(**vars(t)) for t in stems])
    seg, cat = tiles.segment_trees(plot["cat"], stems, tmp_path, merge=False, prune=False, k=8,
                                   power=4.0, attribute="label", workers=4, **SEGMENT)
    assert _rows(seg) == _rows(ref)
    np.testing.assert_array_equal(_by_pid(cat.read(), "label"),
                                  _by_pid(whole.with_attrs(label=labels), "label"))


def test_crowns_at_the_buffer_edge_are_reported(plot, tmp_path):
    with pytest.warns(UserWarning, match="reach the edge") as caught:
        seg, _ = tiles.segment_trees(plot["cat"], plot["stems"], tmp_path, buffer=1.0, max_buffer=1.0,
                                     merge=MERGE, prune=PRUNE, **SEGMENT)
    flagged = sorted(t.tree_id for t in seg if t.extra["crown_at_edge"])
    assert flagged and str(flagged) in str(caught[0].message)
    assert tiles.last_run().rereads == 0


def test_segmentation_edge_cases(plot, tmp_path):
    cat, stems = plot["cat"], plot["stems"]
    seg, out = tiles.segment_trees(cat, [], tmp_path / "none", **SEGMENT)
    assert seg == [] and np.all(out.read().attrs["tree_id"] == -1)
    with pytest.raises(ValueError, match="unique"):
        tiles.segment_trees(cat, [stems[0], stems[0]], tmp_path / "dup", **SEGMENT)
    with pytest.raises(ValueError, match="buffer"):
        tiles.segment_trees(cat, stems, tmp_path / "b", buffer=-1.0)
    with pytest.raises(ValueError, match="max_buffer"):
        tiles.segment_trees(cat, stems, tmp_path / "b", buffer=10.0, max_buffer=5.0)
    with pytest.raises(ValueError, match="unknown segment_trees parameter"):
        tiles.segment_trees(cat, stems, tmp_path / "p", radius_of_everything=1.0)
    with pytest.raises(ValueError, match="unknown merge_branches parameter"):
        tiles.segment_trees(cat, stems, tmp_path / "p", merge={"gravity": 1.0})
    with pytest.raises(ValueError, match="unknown prune_trees parameter"):
        tiles.segment_trees(cat, stems, tmp_path / "p", prune={"tallest": 1.0})
    with pytest.raises(ValueError, match="merge must be"):
        tiles.segment_trees(cat, stems, tmp_path / "p", merge="yes")
    with pytest.raises(ValueError, match="voxel_origin"):
        tiles.segment_trees(cat, stems, tmp_path / "p", voxel_origin=(0, np.nan, 0))


def test_voxel_origin_leaves_the_default_unchanged(plot):
    whole, stems = plot["whole"], plot["stems"][:4]
    lo = whole.xyz[whole.heights("height") >= 0.25].min(axis=0)
    a = trees.segment_trees(whole, stems, voxel_size=0.1)
    b = trees.segment_trees(whole, stems, voxel_size=0.1, voxel_origin=lo)
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ per tree


@pytest.fixture(scope="module")
def store(plot):
    return tiles.split_trees(plot["seg"][1], plot["dir"] / "store", workers=3)


def test_tree_store_reads_each_tree_as_the_plot_has_it(plot, store):
    ref_trees, ref_labels = plot["ref"]
    whole = plot["whole"]
    assert store.ids == sorted(t.tree_id for t in ref_trees)
    for t in ref_trees:
        idx = np.flatnonzero(ref_labels == t.tree_id)
        one = store.read(t.tree_id)
        np.testing.assert_array_equal(one.xyz, whole.xyz[idx])
        for name in ("pid", "height", "scan_id"):
            np.testing.assert_array_equal(one.attrs[name], whole.attrs[name][idx])
        assert store.n_points(t.tree_id) == len(idx) == t.n_points
        lo, hi = whole.xyz[idx].min(axis=0), whole.xyz[idx].max(axis=0)
        np.testing.assert_array_equal(store.bounds(t.tree_id), [*lo, *hi])
        from_tiles = tiles.read_tree(plot["seg"][1], t.tree_id, bounds=(lo[0], lo[1], hi[0], hi[1]))
        np.testing.assert_array_equal(from_tiles.xyz, one.xyz)
    np.testing.assert_array_equal(tiles.read_tree(store.path, store.ids[0]).xyz, store.read(store.ids[0]).xyz)
    assert "TreeStore" in repr(store) and store.ids[0] in store and 10_000 not in store
    assert set(store.tiles_of(store.ids[0])) <= set(store.tiles)
    with pytest.raises(ValueError, match="no tree"):
        store.read(10_000)
    with pytest.raises(ValueError, match="points"):
        store.set_values(store.ids[0], "x", np.zeros(3))
    with pytest.raises(ValueError, match="plain word"):
        store.set_values(store.ids[0], "a/b", np.zeros(store.n_points(store.ids[0])))
    with pytest.raises(ValueError, match="tree store"):
        tiles.TreeStore(plot["dir"])


def test_leaf_wood_per_tree_goes_back_to_the_tiles(plot, store, tmp_path):
    ref_trees, ref_labels = plot["ref"]
    whole = plot["whole"]
    wood = np.full(len(whole), -1, np.int8)
    for t in ref_trees:
        idx = np.flatnonzero(ref_labels == t.tree_id)
        if len(idx) >= 100:
            wood[idx] = leaves.classify_leaf_wood(whole[idx]).astype(np.int8)
    out = tiles.classify_leaf_wood(store, plot["seg"][1], tmp_path, workers=3)
    got = out.read()
    assert got.attrs["wood"].dtype == np.int8
    np.testing.assert_array_equal(_by_pid(got, "wood"), _by_pid(whole.with_attrs(wood=wood), "wood"))
    np.testing.assert_array_equal(_by_pid(got, "tree_id"), _by_pid(plot["seg"][1].read(), "tree_id"))
    assert 0.1 < np.mean(wood[wood >= 0]) < 0.9


def test_qsms_per_tree_match_build_plot(plot, store):
    ref_trees, ref_labels = plot["ref"]
    kw = {"stem_radius_cap": 1.5, "min_points": 1000}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = qsm.build_plot(plot["whole"], ref_labels, ref_trees, **kw)
        for workers in (1, 4):
            got = tiles.build_qsms(store, ref_trees, workers=workers, **kw)
            assert got.models.keys() == ref.models.keys() and len(got) >= 5
            for t in ref.models:
                np.testing.assert_array_equal(got.models[t].cylinders, ref.models[t].cylinders)
            assert got.skipped == ref.skipped
            assert got.table() == ref.table()
            assert got.total_volume == ref.total_volume
    # Saved models are taken up again.
    again = tiles.build_qsms(store, ref_trees, resume=True, **kw)
    assert again.table() == ref.table()
    first = next(iter(ref.models))
    one = tiles.build_qsms(store, ref_trees, ids=[first], resume=True, **kw)
    assert list(one.models) == [first]
    with pytest.raises(ValueError, match="no tree"):
        tiles.build_qsms(store, ref_trees, ids=[10_000], **kw)


def test_crown_metrics_per_tree(plot, store):
    _, ref_labels = plot["ref"]
    ref = trees.crown_metrics_all(plot["whole"], ref_labels)
    got = tiles.crown_metrics(store, workers=3)
    assert got == ref and len(got) == len(store)


def test_tree_work_is_bounded_by_the_largest_tree(store, monkeypatch):
    largest = max(store.n_points(i) for i in store.ids)
    monkeypatch.setenv("SYLVA_MEM_BUDGET", str(3.5 * largest * tiles_trees.BYTES_PER_TREE_POINT / 1e9))
    busy, peak = [0], [0]
    import threading
    lock = threading.Lock()

    def job(i, cloud):
        with lock:
            busy[0] += 1
            peak[0] = max(peak[0], busy[0])
        n = len(cloud)
        with lock:
            busy[0] -= 1
        return n

    counts = store.map(job, workers=8)
    assert counts == {i: store.n_points(i) for i in store.ids}
    assert peak[0] <= 3


# ------------------------------------------------------------------ memory


def test_segmentation_holds_a_few_tiles(tmp_path):
    """On a 60 m plot in 10 m tiles, no stage of the segmentation holds more than a few tiles."""
    rng = np.random.default_rng(4)
    rows = [(x, y, d, h) for x, y, d, h in zip(rng.uniform(3, 57, 30), rng.uniform(3, 57, 30),
                                               rng.uniform(0.2, 0.5, 30), rng.uniform(12, 20, 30), strict=True)]
    scene = synthetic.forest(rows, size=60.0, ground_points=250_000, margin=2.0, seed=4)
    pos = [(x, y) for x in (8, 30, 52) for y in (8, 30, 52)]
    scans = _scans(scene, pos, resolution=0.35)
    cat = tiles.from_scans(scans, tmp_path / "t", tile_size=10.0, voxel_size=0.025)
    h = tiles.normalize(tiles.classify_ground(cat, tmp_path / "g", method="pmf", buffer=3.0),
                        tmp_path / "h", buffer=3.0)
    n = h.n_points
    stems = tiles.detect_stems(h, buffer=2.0)
    # The largest read is a tile with its buffer: 26 m square of graph nodes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seg, seg_cat = tiles.segment_trees(h, stems, tmp_path / "s", buffer=8.0, max_buffer=8.0,
                                           merge=MERGE, prune=PRUNE, **SEGMENT)
    info = tiles.last_run()
    assert len(seg) >= 20
    # A 26 m square is a fifth of the plot; its graph nodes are fewer than its points.
    assert info.max_points < n / 5, (info, n)
    store = tiles.split_trees(seg_cat, tmp_path / "store")
    assert tiles.last_run().max_points < n / 8
    assert max(store.n_points(i) for i in store.ids) < n / 8


# ------------------------------------------------------------------ the workflow


def _small_plot():
    scene = _scene(size=20.0, n_trees=5, seed=3)
    return _scans(scene, [(4, 4), (16, 16), (4, 16), (16, 4)], resolution=0.3)


def test_run_plot_resumes_after_an_interruption(tmp_path, monkeypatch):
    scans = _small_plot()
    kw = dict(bounds=(-2, -2, 22, 22), tile_size=10.0, voxel_size=0.02, sor={"k": 6, "std_ratio": 2.0},
              ground={"method": "pmf", "buffer": 5.0}, stems={}, buffer=10.0,
              qsm_options={"stem_radius_cap": 1.5, "min_points": 500}, workers=2, log=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = tiles.run_plot(scans, tmp_path / "full", **kw)
    assert len(full.trees) >= 3 and len(full.table) == len(full.trees)
    assert {"tiles", "sor", "heights", "segmented", "wood"} <= set(full.catalogs)
    assert (tmp_path / "full" / "trees.csv").exists() and (tmp_path / "full" / "qsm_table.csv").exists()
    # Heights come from the DTM at full precision, not from its ASCII grid.
    dtm = tiles.dtm(tiles.catalog(tmp_path / "full" / "ground"), resolution=0.5, buffer=5.0)
    np.testing.assert_array_equal(full.dtm.data, dtm.data)
    h = full.catalogs["heights"].read()
    np.testing.assert_array_equal(h.attrs["height"],
                                  ground.normalize_height(PointCloud(h.xyz), dtm).attrs["height"])

    # Interrupted while fitting QSMs, after two trees.
    calls = []
    real = tiles_trees._qsm_one

    def failing(*a, **k):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append(a[1])
        return real(*a, **k)

    monkeypatch.setattr(tiles_trees, "_qsm_one", failing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(KeyboardInterrupt):
            tiles.run_plot(scans, tmp_path / "run", **{**kw, "workers": 1})
    monkeypatch.setattr(tiles_trees, "_qsm_one", lambda *a, **k: (calls.append(a[1]), real(*a, **k))[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = tiles.run_plot(scans, tmp_path / "run", **{**kw, "workers": 1})
    # Nothing done twice: stages before the QSMs came from the first run, and
    # so did the two trees it had fitted.
    assert all(again.timings[s] == 0.0 for s in ("tiles", "sor", "ground", "heights", "stems",
                                                 "segmentation", "split", "leaf/wood"))
    assert again.timings["QSMs"] > 0 and len(calls) == len(set(calls)) == len(full.store)
    assert again.table == full.table
    assert again.qsms.table() == full.qsms.table()
    np.testing.assert_array_equal(_by_pid(again.catalogs["wood"].read(), "wood"),
                                  _by_pid(full.catalogs["wood"].read(), "wood"))
    # A third run does nothing again.
    third = tiles.run_plot(scans, tmp_path / "run", **{**kw, "workers": 1})
    assert all(v == 0.0 for v in third.timings.values())
    with open(tmp_path / "run" / "trees.pkl", "rb") as f:
        assert [t.tree_id for t in pickle.load(f)] == [t.tree_id for t in full.trees]


def test_run_plot_checks_its_input(tmp_path):
    scans = _small_plot()[:2]
    with pytest.raises(ValueError, match="use flags"):
        tiles.run_plot(scans, tmp_path, use=[True], log=None)
    with pytest.raises(ValueError, match="transforms"):
        tiles.run_plot(scans, tmp_path, transforms=[None], log=None)


def test_cli_runs_the_workflow(tmp_path, capsys):
    paths = []
    for i, s in enumerate(_small_plot()):
        p = tmp_path / f"s{i}.laz"
        io.write(s, p)
        paths.append(str(p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(["--no-progress", "tiles-plot", *paths, str(tmp_path / "run"), "--bounds", "-2", "-2", "22",
              "22", "--ground", "pmf", "--min-arc", "0", "--buffer", "10", "--workers", "2"])
    out = capsys.readouterr().out
    assert "trees" in out and (tmp_path / "run" / "trees.csv").exists()
