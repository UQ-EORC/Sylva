"""Tiled point operations (sylva.als.tiles) against the same operations on the whole cloud."""

import csv

import numpy as np
import pytest

from sylva import PointCloud, als, filters, ground, io, synthetic, trees
from sylva.als import tiles
from sylva.cli import main

POSITIONS = [(5, 5), (15, 15), (25, 5), (5, 25), (25, 25), (15, 3), (3, 15)]


def _scene(size=30.0, n_trees=12, seed=1):
    rng = np.random.default_rng(seed)
    rows = [(x, y, d, h) for x, y, d, h in zip(rng.uniform(3, size - 3, n_trees),
                                               rng.uniform(3, size - 3, n_trees),
                                               rng.uniform(0.2, 0.5, n_trees),
                                               rng.uniform(12, 20, n_trees))]
    return synthetic.forest(rows, size=size, ground_points=int(70 * size * size), margin=2.0,
                            seed=seed)


def _scans(scene, positions, resolution=0.25):
    out = []
    for i, (x, y) in enumerate(positions):
        pc = synthetic.scan(scene, origin=(x, y, 1.5), resolution_deg=resolution).to_pointcloud()
        n = len(pc)
        # Attributes a plot carries through its processing, of several types.
        out.append(pc.with_attrs(pid=np.arange(n, dtype=np.int64) + 10_000_000 * i,
                                 wood=(pc.attrs["classification"] == 5).astype(np.int8),
                                 height=pc.z.astype(np.float64)))
    return out


def _key(cloud):
    return np.sort(np.asarray(cloud.attrs["pid"], dtype=np.int64))


def _by_pid(cloud):
    return cloud[np.argsort(cloud.attrs["pid"])]


@pytest.fixture(scope="module")
def plot(tmp_path_factory):
    """Scans of a 30 m synthetic plot, and their tiles thinned to 2 cm on 10 m tiles."""
    d = tmp_path_factory.mktemp("tiles")
    scans = _scans(_scene(), POSITIONS)
    cat = tiles.from_scans(scans, d / "tiles", tile_size=10.0, voxel_size=0.02, scale=0.0001)
    return {"dir": d, "scans": scans, "cat": cat, "whole": cat.read()}


# ------------------------------------------------------------------ building tiles


def _concat(scans, transforms=None, bounds=None):
    parts = []
    for i, s in enumerate(scans):
        if transforms is not None and transforms[i] is not None:
            s = s.transform(transforms[i])
        parts.append(s.with_attrs(scan_id=np.full(len(s), i, np.uint32)))
    c = PointCloud.concatenate(parts)
    if bounds is not None:
        x0, y0, x1, y1 = bounds
        c = c[(c.x >= x0) & (c.x <= x1) & (c.y >= y0) & (c.y <= y1)]
    return c


@pytest.mark.parametrize("tile_size,workers", [(5.0, 1), (7.5, 3), (15.0, 8)])
def test_from_scans_is_the_thinned_concatenation(plot, tmp_path, tile_size, workers):
    scans = plot["scans"]
    t = np.eye(4)
    t[:3, 3] = (0.37, -0.21, 0.05)
    c, s = np.cos(0.2), np.sin(0.2)
    t[:2, :2] = [[c, -s], [s, c]]
    transforms = [None, t, None, None, np.eye(4), None, None]
    bounds = (-1.0, 0.5, 28.0, 31.0)
    cat = tiles.from_scans(scans, tmp_path, tile_size=tile_size, voxel_size=0.05,
                           transforms=transforms, bounds=bounds, scale=0.0001, workers=workers)
    ref = filters.voxel_downsample(_concat(scans, transforms, bounds), 0.05, origin=(0, 0, 0))
    got = cat.read()
    assert len(got) == len(ref) == cat.n_points
    key = lambda c: np.sort(c.attrs["scan_id"].astype(np.int64) * 10**9 + c.attrs["pid"])  # noqa: E731
    np.testing.assert_array_equal(key(got), key(ref))
    # Coordinates are the reference's up to the quantisation, attributes exactly.
    g, r = _by_pid(got), _by_pid(ref)
    assert np.abs(g.xyz - r.xyz).max() <= 0.5e-4 + 1e-9
    for name in ("tree_id", "wood", "height", "scan_id"):
        np.testing.assert_array_equal(g.attrs[name], r.attrs[name])
    # Tiles sit on multiples of the tile size and never split a voxel.
    for tile in cat.tiles:
        x0, y0 = (float(v) for v in tile.path.rsplit("/", 1)[1].rsplit(".", 1)[0].split("_"))
        assert x0 % tile_size == pytest.approx(0) or x0 % tile_size == pytest.approx(tile_size)
        assert tile.bounds[0] >= x0 - 1e-6 and tile.bounds[3] <= x0 + tile_size + 1e-6
    assert tiles.last_run().max_points <= max(len(s) for s in scans) + len(got)


def test_from_scans_reads_files_and_keeps_every_point_without_a_voxel(plot, tmp_path):
    scans = plot["scans"][:3]
    paths = []
    for i, s in enumerate(scans):
        p = tmp_path / f"scan{i}.laz"
        io.write(s, p)
        paths.append(p)
    back = [io.read(p) for p in paths]
    cat = tiles.from_scans(paths, tmp_path / "t", tile_size=10.0, voxel_size=0.02)
    ref = filters.voxel_downsample(_concat(back), 0.02, origin=(0, 0, 0))
    np.testing.assert_array_equal(_key(cat.read()), _key(ref))
    every = tiles.from_scans([paths[0], scans[1]], tmp_path / "all", tile_size=10.0)
    assert every.n_points == len(back[0]) + len(scans[1])
    np.testing.assert_array_equal(np.unique(every.read().attrs["scan_id"]), [0, 1])


def test_from_scans_refuses_bad_input(plot, tmp_path):
    s = plot["scans"][0]
    with pytest.raises(ValueError, match="whole number of voxels"):
        tiles.from_scans([s], tmp_path, tile_size=1.0, voxel_size=0.3)
    with pytest.raises(ValueError, match="4x4"):
        tiles.from_scans([s], tmp_path, transforms=[np.eye(3)])
    with pytest.raises(ValueError, match="transforms"):
        tiles.from_scans([s, s], tmp_path, transforms=[np.eye(4)])
    with pytest.raises(ValueError, match="no scans"):
        tiles.from_scans([], tmp_path)
    with pytest.raises(ValueError, match="path or a PointCloud"):
        tiles.from_scans([42], tmp_path)
    with pytest.raises(ValueError, match="bounds"):
        tiles.from_scans([s], tmp_path, bounds=(10, 10, 0, 0))
    bad = PointCloud(np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        tiles.from_scans([bad], tmp_path)
    with pytest.raises(OSError):
        tiles.from_scans([tmp_path / "missing.laz"], tmp_path)
    assert not (tmp_path / ".sylva-scans").exists()
    empty = tiles.from_scans([PointCloud(np.zeros((0, 3)))], tmp_path / "e", tile_size=5.0)
    assert len(empty) == 0


# ------------------------------------------------------------------ point operations


@pytest.mark.parametrize("voxel,workers", [(0.05, 1), (0.13, 4)])
def test_voxel_thinning_on_a_global_grid(plot, tmp_path, voxel, workers):
    out = tiles.voxel_downsample(plot["cat"], tmp_path, voxel, workers=workers)
    ref = filters.voxel_downsample(plot["whole"], voxel, origin=(0, 0, 0))
    np.testing.assert_array_equal(_key(out.read()), _key(ref))
    # Other tile sizes of the same points give the same thinning.
    re = als.retile(plot["cat"], tmp_path / "re", 7.0)
    out = tiles.voxel_downsample(re, tmp_path / "re_thin", voxel, workers=workers)
    np.testing.assert_array_equal(_key(out.read()), _key(filters.voxel_downsample(
        re.read(), voxel, origin=(0, 0, 0))))


@pytest.mark.parametrize("k,std_ratio,buffer,workers", [(6, 1.0, 0.0, 1), (6, 1.0, 1.0, 4),
                                                        (8, 2.0, 0.2, 3)])
def test_sor_is_exact_in_two_passes(plot, tmp_path, k, std_ratio, buffer, workers):
    cat, whole = plot["cat"], plot["whole"]
    out = tiles.statistical_outlier_removal(cat, tmp_path, k=k, std_ratio=std_ratio,
                                            buffer=buffer, workers=workers)
    ref = filters.statistical_outlier_removal(whole, k=k, std_ratio=std_ratio)
    assert 0 < len(ref) < len(whole)
    np.testing.assert_array_equal(_key(out.read()), _key(ref))
    info = tiles.last_run()
    if buffer == 0.0:
        assert info.rereads > 0 and info.widened_points > 0
    assert not (tmp_path / ".sylva-sor").exists()


def test_sor_needs_the_global_statistics(plot, tmp_path):
    # One pass per tile (statistics of each tile and its buffer, as als.filter
    # does) keeps a different set; the two-pass tiling keeps the whole cloud's.
    cat, whole = plot["cat"], plot["whole"]
    ref = _key(filters.statistical_outlier_removal(whole, k=6, std_ratio=1.0))
    local = als.filter(cat, tmp_path / "local", method="sor", k=6, std_ratio=1.0, buffer=1.0)
    assert not np.array_equal(_key(local.read()), ref)
    two = tiles.statistical_outlier_removal(cat, tmp_path / "two", k=6, std_ratio=1.0)
    np.testing.assert_array_equal(_key(two.read()), ref)
    # Classifying rather than removing marks exactly the removed points.
    cls = _by_pid(tiles.statistical_outlier_removal(cat, tmp_path / "cls", k=6, std_ratio=1.0,
                                                    classify=True).read())
    assert len(cls) == len(whole)
    np.testing.assert_array_equal(np.sort(cls.attrs["pid"][cls.attrs["classification"] != 7]), ref)


@pytest.mark.parametrize("buffer", [None, 0.0])
def test_ror_is_exact(plot, tmp_path, buffer):
    out = tiles.radius_outlier_removal(plot["cat"], tmp_path, 0.08, min_neighbors=4,
                                       buffer=buffer, workers=2)
    ref = filters.radius_outlier_removal(plot["whole"], 0.08, min_neighbors=4)
    assert len(ref) < len(plot["whole"])
    np.testing.assert_array_equal(_key(out.read()), _key(ref))


def test_normals_and_shape_features_are_exact(plot, tmp_path):
    whole = _by_pid(plot["whole"])
    got = _by_pid(tiles.estimate_normals(plot["cat"], tmp_path / "n", k=12, buffer=0.1).read())
    normals = np.c_[got.attrs["normal_x"], got.attrs["normal_y"], got.attrs["normal_z"]]
    np.testing.assert_array_equal(normals, filters.estimate_normals(whole, k=12))
    assert tiles.last_run().rereads > 0
    got = _by_pid(tiles.planarity_linearity(plot["cat"], tmp_path / "s", k=20, workers=1).read())
    p, lin = filters.planarity_linearity(whole, k=20)
    np.testing.assert_array_equal(got.attrs["planarity"], p)
    np.testing.assert_array_equal(got.attrs["linearity"], lin)


def test_attributes_survive_every_step(plot, tmp_path):
    whole = _by_pid(plot["whole"])
    out = _by_pid(tiles.estimate_normals(plot["cat"], tmp_path, k=8).read())
    for name, dtype in [("tree_id", None), ("wood", np.int8), ("height", np.float64),
                        ("scan_id", np.uint32), ("pid", np.int64)]:
        if dtype is not None:
            assert out.attrs[name].dtype == dtype, name
        np.testing.assert_array_equal(out.attrs[name], whole.attrs[name])


# ------------------------------------------------------------------ ground and heights


@pytest.fixture(scope="module")
def ground_tiles(plot):
    d = plot["dir"]
    g = tiles.classify_ground(plot["cat"], d / "pmf", method="pmf", buffer=6.0)
    return g, _by_pid(g.read())


def test_pmf_ground_is_exact_and_csf_close(plot, ground_tiles, tmp_path):
    whole = _by_pid(plot["whole"])
    _, g = ground_tiles
    ref = ground.classify_ground_pmf(whole)
    np.testing.assert_array_equal(g.attrs["classification"], ref.attrs["classification"])
    csf = _by_pid(tiles.classify_ground(plot["cat"], tmp_path, method="csf", buffer=5.0).read())
    agree = np.mean(csf.attrs["classification"] == ground.classify_ground_csf(whole)
                    .attrs["classification"])
    assert agree > 0.995


def test_dtm_and_heights_match_the_whole_cloud(ground_tiles, tmp_path):
    gcat, g = ground_tiles
    ref = ground.make_dtm(g, resolution=0.5)
    for workers in (1, 3):
        dtm = tiles.dtm(gcat, resolution=0.5, buffer=3.0, workers=workers)
        c0 = round((ref.xmin - dtm.xmin) / 0.5)
        r0 = round((ref.ymin - dtm.ymin) / 0.5)
        sub = dtm.data[r0:r0 + ref.data.shape[0], c0:c0 + ref.data.shape[1]]
        np.testing.assert_array_equal(sub, ref.data)
    heights = _by_pid(tiles.normalize(gcat, tmp_path, dtm_resolution=0.5, buffer=3.0).read())
    href = ground.normalize_height(g, ref).attrs["height"]
    gp = g[g.attrs["classification"] == 2]
    lo, hi = gp.xyz.min(axis=0) + 2.0, gp.xyz.max(axis=0) - 2.0
    inside = (g.x > lo[0]) & (g.x < hi[0]) & (g.y > lo[1]) & (g.y < hi[1])
    assert inside.mean() > 0.7
    np.testing.assert_allclose(heights.attrs["height"][inside], href[inside], rtol=0, atol=1e-9)


# ------------------------------------------------------------------ stems


@pytest.fixture(scope="module")
def normalised(plot, ground_tiles):
    gcat, _ = ground_tiles
    n = tiles.normalize(gcat, plot["dir"] / "norm", dtm_resolution=0.5, buffer=5.0)
    return n, n.read()


@pytest.mark.parametrize("tile_size,buffer,workers", [(10.0, 2.0, 1), (6.0, 2.0, 4),
                                                      (10.0, 1.0, 3)])
def test_stems_match_the_whole_cloud(normalised, tmp_path, tile_size, buffer, workers):
    ncat, whole = normalised
    cat = ncat if tile_size == 10.0 else als.retile(ncat, tmp_path, tile_size)
    got = tiles.detect_stems(cat, buffer=buffer, workers=workers)
    ref = trees.detect_stems(cat.read(), cluster_seeds=True)
    assert len(ref) >= 10
    assert [(t.tree_id, t.x, t.y, t.dbh, t.n_slices) for t in got] == \
        [(t.tree_id, t.x, t.y, t.dbh, t.n_slices) for t in ref]


def test_stems_refuse_unknown_parameters(normalised):
    with pytest.raises(ValueError, match="unknown stem detection parameter"):
        tiles.detect_stems(normalised[0], radius_of_everything=3)


# ------------------------------------------------------------------ memory


def test_no_step_holds_the_whole_plot(tmp_path):
    """On a 60 m plot in 10 m tiles, no step holds more than a few tiles' worth."""
    scene = _scene(size=60.0, n_trees=30, seed=4)
    pos = [(x, y) for x in (8, 30, 52) for y in (8, 30, 52)]
    scans = _scans(scene, pos, resolution=0.35)
    largest_scan = max(len(s) for s in scans)
    cat = tiles.from_scans(scans, tmp_path / "t", tile_size=10.0, voxel_size=0.025)
    n = cat.n_points
    assert len(cat) >= 36
    assert tiles.last_run().max_points <= largest_scan + n // 8
    steps = [
        lambda: tiles.voxel_downsample(cat, tmp_path / "v", 0.05),
        lambda: tiles.statistical_outlier_removal(cat, tmp_path / "s", k=6, std_ratio=1.0),
        lambda: tiles.radius_outlier_removal(cat, tmp_path / "r", 0.1),
        lambda: tiles.estimate_normals(cat, tmp_path / "n", k=10),
    ]
    for step in steps:
        step()
        info = tiles.last_run()
        assert info.chunks == len(cat)
        assert info.max_points < n / 8, info
    g = tiles.classify_ground(cat, tmp_path / "g", method="pmf", buffer=3.0)
    h = tiles.normalize(g, tmp_path / "h", buffer=3.0)
    tiles.detect_stems(h, buffer=2.0)
    assert tiles.last_run().max_points < n / 8


# ------------------------------------------------------------------ checks and CLI


def test_bad_arguments_raise(plot, tmp_path):
    cat = plot["cat"]
    with pytest.raises(ValueError, match="buffer"):
        tiles.statistical_outlier_removal(cat, tmp_path, buffer=-1)
    with pytest.raises(ValueError, match="k must"):
        tiles.statistical_outlier_removal(cat, tmp_path, k=0)
    with pytest.raises(ValueError, match="radius"):
        tiles.radius_outlier_removal(cat, tmp_path, 0.0)
    with pytest.raises(ValueError, match="voxel"):
        tiles.voxel_downsample(cat, tmp_path, 0.0)
    with pytest.raises(ValueError, match="k must"):
        tiles.estimate_normals(cat, tmp_path, k=0)
    with pytest.raises(ValueError, match="catalogue"):
        tiles.voxel_downsample(cat, cat.tiles[0].path.rsplit("/", 1)[0], 0.1)
    with pytest.raises(ValueError, match="origin"):
        filters.voxel_downsample(plot["whole"], 0.1, method="centroid", origin=(0, 0, 0))


def test_voxel_downsample_default_grid_is_unchanged(plot):
    c = plot["scans"][0]
    np.testing.assert_array_equal(filters.voxel_downsample(c, 0.1).xyz,
                                  filters.voxel_downsample(c, 0.1, origin=c.xyz.min(axis=0)).xyz)


def test_shared_engine_objects():
    assert tiles.Catalog is als.Catalog and tiles.apply is als.apply
    assert tiles.retile is als.retile and tiles.write_tiles is als.write_tiles


def test_cli(plot, tmp_path, capsys):
    scans = []
    for i, s in enumerate(plot["scans"][:3]):
        p = tmp_path / f"s{i}.laz"
        io.write(s, p)
        scans.append(str(p))
    np.save(tmp_path / "t.npy", np.stack([np.eye(4)] * 3))
    main(["--no-progress", "tiles-from-scans", *scans, str(tmp_path / "t"), "--voxel", "0.02",
          "--tile-size", "10", "--transforms", str(tmp_path / "t.npy")])
    assert "points held at once" in capsys.readouterr().out
    main(["--no-progress", "tiles-thin", str(tmp_path / "t"), str(tmp_path / "v"), "--voxel", "0.04"])
    main(["--no-progress", "tiles-filter", str(tmp_path / "v"), str(tmp_path / "f"), "--k", "6",
          "--std-ratio", "1"])
    main(["--no-progress", "tiles-features", str(tmp_path / "f"), str(tmp_path / "x"),
          "--feature", "shape"])
    main(["--no-progress", "tiles-stems", str(tmp_path / "t"), str(tmp_path / "stems.csv"),
          "--height-attribute", "height"])
    out = capsys.readouterr().out
    assert "stems from" in out and (tmp_path / "stems.csv").exists()
    n_stems = int(out.split("stems from")[0].split()[-1].replace(",", ""))
    with open(tmp_path / "stems.csv", newline="") as f:      # written without pandas, which is optional
        rows = list(csv.DictReader(f))
    assert len(rows) == n_stems and (n_stems == 0 or {"x", "y"} <= set(rows[0]))
    assert "planarity" in tiles.catalog(tmp_path / "x").read().attrs
