# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""ALS tiles: the catalogue, the chunk engine and the whole-area operations.

The reference for every whole-area result is the same single-cloud function
run on all the tiles merged into one cloud (or on the one tile): tiling must
not show in the result away from the outer edge of the survey.
"""

import time

import numpy as np
import pytest

from sylva import PointCloud, Raster, als, filters, ground, io, limits, progress, synthetic

SIZE = 60.0


def _scene(seed=3, n_trees=10, size=SIZE):
    rng = np.random.default_rng(seed)
    trees = [(float(x), float(y), 0.3, float(h)) for x, y, h in
             zip(rng.uniform(6, size - 6, n_trees), rng.uniform(6, size - 6, n_trees),
                 rng.uniform(10, 18, n_trees), strict=True)]
    return synthetic.forest(trees, size=size, ground_points=100, margin=0.0, seed=seed)


@pytest.fixture(scope="module")
def flight():
    return synthetic.als_flight(_scene(), pulse_rate=12_000, line_spacing=30.0,
                                bounds=(0.0, 0.0, SIZE, SIZE), seed=1)


@pytest.fixture(scope="module")
def tiles(flight, tmp_path_factory):
    """The flight as 2 x 2 tiles of 30 m, with a CRS."""
    d = tmp_path_factory.mktemp("tiles")
    cat = flight.write_tiles(d, size=30.0, epsg=32755)
    assert len(cat) == 4
    return cat


@pytest.fixture(scope="module")
def merged(tiles):
    """All the tiles as one cloud, quantised as the files are."""
    return tiles.read()


def _interior(r: Raster, margin_cells: int) -> tuple[slice, slice]:
    m = margin_cells
    return slice(m, r.shape[0] - m), slice(m, r.shape[1] - m)


def _same_grid_dtm(cloud, dtm: Raster, **kw):
    """ground.make_dtm of a cloud on the catalogue DTM's grid."""
    bounds = (dtm.xmin, dtm.ymin, dtm.xmax - dtm.resolution / 2, dtm.ymax - dtm.resolution / 2)
    return ground.make_dtm(cloud, dtm.resolution, bounds=bounds, **kw)


# ------------------------------------------------------------------ catalogue

def test_catalog_reads_headers_only(tiles, flight):
    assert tiles.n_points == len(flight.points)
    assert tiles.crs == "EPSG:32755"
    assert all(t.point_format == 6 and t.version == (1, 4) for t in tiles.tiles)
    assert not any(t.spatial_index for t in tiles.tiles)
    b = tiles.bounds
    assert b[0] >= 0 and b[3] <= SIZE and b[2] < b[5]
    assert tiles.issues() == []
    tiles.validate()
    text = tiles.report()
    assert "4 tiles" in text and "EPSG:32755" in text and "no problems found" in text
    s = tiles.summary()
    assert s["n_tiles"] == 4 and s["point_formats"] == {6: 4} and s["density"] > 10
    assert "Catalog(4 tiles" in repr(tiles)


def test_catalog_from_a_directory_a_file_or_a_list(tiles, tmp_path):
    d = str(tiles.tiles[0].path).rsplit("/", 1)[0]
    assert als.catalog(d).paths == sorted(tiles.paths)
    assert len(als.catalog(tiles.paths[0])) == 1
    assert len(als.catalog(tiles.paths[:2])) == 2
    with pytest.raises(ValueError, match="no files"):
        als.catalog(tmp_path)
    with pytest.raises(ValueError, match="no files"):
        als.catalog(d, pattern="*.ply")


def test_problems_are_reported(tiles, flight, tmp_path):
    pts = flight.points
    # Two tiles in another CRS and point format, one overlapping the others.
    als.write_tiles(pts, tmp_path, 30.0, epsg=32755)
    first = sorted(tmp_path.iterdir())[0]
    io.write(io.read(first), tmp_path / "copy.laz")         # same extent: overlap, no CRS
    als.write_tiles(pts[pts.x > 45], tmp_path / "other", 30.0, epsg=28355, point_format=1)
    cat = als.catalog([*sorted(tmp_path.glob("*.laz")), *sorted((tmp_path / "other").glob("*.laz")),
                       tmp_path / "missing.laz"])
    kinds = [k for k, _ in cat.issues()]
    assert kinds == ["missing", "mixed_crs", "mixed_point_format", "overlap"], cat.issues()
    assert cat.missing == [str(tmp_path / "missing.laz")]
    assert cat.overlaps()
    assert cat.crs is None
    with pytest.raises(ValueError, match="missing"):
        cat.validate()
    # Processing refuses a catalogue with missing files.
    with pytest.raises(ValueError, match="missing"):
        als.dtm(cat, 1.0)


def test_holes_in_the_coverage_are_found(flight, tmp_path):
    cat = als.write_tiles(flight.points, tmp_path, 20.0)
    assert len(cat) == 9
    middle = [p for p in cat.paths if p.endswith("/20_20.laz")]
    holed = als.catalog([p for p in cat.paths if p not in middle])
    gaps = holed.gaps()
    assert len(gaps) == 1
    (x0, y0, x1, y1), area = gaps[0]
    assert 18 < x0 < 22 and 38 < x1 < 42 and 250 < area < 450
    assert [k for k, _ in holed.issues()] == ["no_crs", "gap"]


def test_unreadable_files_are_recorded(tiles, tmp_path):
    bad = tmp_path / "bad.laz"
    bad.write_bytes(b"not a las file at all")
    cat = als.catalog([*tiles.paths, str(bad)])
    assert len(cat) == 4 and cat.unreadable[0][0] == str(bad)
    assert cat.issues()[0][0] == "unreadable"
    with pytest.raises(ValueError, match="could not be read"):
        cat.chunks()


def test_a_catalogue_pickles(tiles):
    import pickle

    back = pickle.loads(pickle.dumps(tiles))
    assert back == tiles


# ------------------------------------------------------------------ chunks and reading

def test_chunks_and_their_buffers(tiles, merged):
    per_tile = tiles.chunks(buffer=5.0)
    assert [c.own for c in per_tile] == [0, 1, 2, 3]
    assert per_tile[0].buffer == 5.0
    assert all(set(c.files) == {0, 1, 2, 3} for c in per_tile)
    grid = tiles.chunks(chunk_size=20.0, buffer=2.0)
    # Half-open cores: points at x or y = 60 exactly start a fourth row or column.
    assert len(grid) in (9, 12, 16)
    middle = next(c for c in grid if c.name == "20_20")
    assert middle.core == (20.0, 20.0, 40.0, 40.0) and middle.outer == (18.0, 18.0, 42.0, 42.0)
    assert [c.index for c in grid] == list(range(len(grid)))
    with pytest.raises(ValueError, match="chunk size"):
        tiles.chunks(chunk_size=-1.0)
    with pytest.raises(ValueError, match="buffer"):
        tiles.chunks(buffer=-1.0)
    # Reading a box gives the merged cloud's points in it.
    box = (10.0, 12.0, 35.0, 41.0)
    got = tiles.read(box)
    x, y = merged.x, merged.y
    expect = (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
    assert len(got) == expect.sum()
    assert np.array_equal(np.sort(got.attrs["gps_time"]), np.sort(merged.attrs["gps_time"][expect]))
    with pytest.raises(ValueError, match="bounds"):
        tiles.read((5.0, 5.0, 1.0, 1.0))


def test_apply_sees_each_point_once_as_core(tiles, merged):
    seen = als.apply(tiles, lambda c, ch: c, chunk_size=25.0, buffer=3.0)
    assert len(seen) == len(merged)
    assert "buffer" not in seen.attrs
    assert np.array_equal(np.sort(seen.attrs["gps_time"]), np.sort(merged.attrs["gps_time"]))
    counts = als.apply(tiles, lambda c, ch: (int((~c.attrs["buffer"]).sum()), len(c)), buffer=4.0)
    assert [n for n, _ in counts] == [t.n_points for t in tiles.tiles]
    assert all(total > own for own, total in counts)


def test_apply_does_not_depend_on_the_workers(tiles):
    def lowest(cloud, chunk):
        return cloud[cloud.z < np.quantile(cloud.z, 0.5)]

    one = als.apply(tiles, lowest, chunk_size=20.0, workers=1)
    four = als.apply(tiles, lowest, chunk_size=20.0, workers=4)
    assert np.array_equal(one.xyz, four.xyz)


def test_apply_mosaics_rasters_seamlessly(tiles, merged):
    grid_dtm = als.dtm(tiles, 1.0)

    def chm(cloud, chunk):
        x0, y0, x1, y1 = chunk.outer
        return ground.make_chm(cloud, 1.0, bounds=(np.floor(x0), np.floor(y0), x1, y1))

    got = als.apply(tiles, chm, chunk_size=20.0, buffer=2.0)
    whole = ground.make_chm(merged, 1.0, bounds=(grid_dtm.xmin, grid_dtm.ymin,
                                                 grid_dtm.xmax - 0.5, grid_dtm.ymax - 0.5))
    assert got.shape == whole.shape and got.crs == "EPSG:32755"
    assert np.array_equal(got.data, whole.data)

    def misaligned(cloud, chunk):
        x0, y0, x1, y1 = chunk.outer
        return ground.make_chm(cloud, 1.0, bounds=(np.floor(x0) + 0.3, np.floor(y0), x1, y1))

    with pytest.raises(ValueError, match="not aligned"):
        als.apply(tiles, misaligned, chunk_size=30.0)


def test_apply_writes_point_outputs(tiles, tmp_path):
    out = als.apply(tiles, lambda c, ch: c[c.attrs["return_number"] == 1], out=tmp_path,
                    format="las")
    assert isinstance(out, als.Catalog) and len(out) == 4
    assert all(p.endswith(".las") for p in out.paths)
    assert out.crs == "EPSG:32755"
    first = io.read(out.paths[0])
    assert (first.attrs["return_number"] == 1).all()
    assert "buffer" not in first.attrs


def test_apply_errors_stop_the_run(tiles):
    calls = []

    def boom(cloud, chunk):
        calls.append(chunk.index)
        raise RuntimeError(f"chunk {chunk.index}")

    with pytest.raises(RuntimeError, match="chunk 0"):
        als.apply(tiles, boom, chunk_size=10.0, workers=1)
    assert len(calls) < len(tiles.chunks(chunk_size=10.0))
    with pytest.raises(ValueError, match="callable"):
        als.apply(tiles, 3)
    with pytest.raises(ValueError, match="workers"):
        als.apply(tiles, lambda c, ch: None, workers=0)


def test_chunks_too_big_for_memory_are_refused(tiles):
    limits.set_budget(1e-6)
    try:
        with pytest.raises(ValueError, match="smaller chunk_size"):
            als.apply(tiles, lambda c, ch: None)
        with pytest.raises(ValueError, match="SYLVA_MEM_BUDGET"):
            als.dtm(tiles, 1.0)
    finally:
        limits.set_budget(None)


def test_apply_reports_progress(tiles):
    seen = set()

    def slow(cloud, chunk):
        time.sleep(0.2)

    with progress.bar(lambda stages: seen.update(s[0] for s in stages), interval=0.02):
        als.apply(tiles, slow, workers=1)
    assert "processing ALS chunks" in seen


# ------------------------------------------------------------------ DTM, CHM, normalising

def test_dtm_of_tiles_equals_dtm_of_the_merged_cloud(tiles, merged):
    per_tile = als.dtm(tiles, 1.0, buffer=10.0)
    whole = _same_grid_dtm(merged, per_tile)
    assert per_tile.shape == whole.shape and per_tile.crs == "EPSG:32755"
    inner = _interior(per_tile, 5)
    assert np.array_equal(per_tile.data[inner], whole.data[inner])
    # Neither the chunking nor the workers change it.
    grid = als.dtm(tiles, 1.0, chunk_size=25.0, buffer=10.0, workers=3)
    assert np.array_equal(grid.data[inner], whole.data[inner])
    assert np.array_equal(als.dtm(tiles, 1.0, buffer=10.0, workers=1).data, per_tile.data)
    # And it is the terrain the flight was flown over.
    truth = synthetic.terrain_height(*per_tile.cell_centers())
    assert np.abs(per_tile.data - truth)[inner].max() < 0.2


def test_interpolated_dtm_is_seamless(tiles, merged):
    tin = als.dtm(tiles, 2.0, method="tin", buffer=10.0)
    whole = _same_grid_dtm(merged, tin, method="tin")
    inner = _interior(tin, 3)
    np.testing.assert_allclose(tin.data[inner], whole.data[inner], atol=1e-9)
    with pytest.raises(ValueError, match="unknown method"):
        als.dtm(tiles, 1.0, method="kriging")
    with pytest.raises(ValueError, match="resolution"):
        als.dtm(tiles, 0.0)


def test_chm_of_tiles_equals_chm_of_the_merged_cloud(tiles, merged):
    per_tile = als.chm(tiles, 1.0, dtm_resolution=1.0, buffer=10.0)
    dtm = _same_grid_dtm(merged, per_tile)
    whole = ground.make_chm(ground.normalize_height(merged, dtm), 1.0,
                            bounds=(per_tile.xmin, per_tile.ymin, per_tile.xmax - 0.5,
                                    per_tile.ymax - 0.5))
    inner = _interior(per_tile, 5)
    np.testing.assert_allclose(per_tile.data[inner], whole.data[inner], atol=1e-9)
    assert 10 < np.nanmax(per_tile.data) < 25
    # A given DTM, or z as it is (a surface model).
    given = als.chm(tiles, 1.0, dtm=dtm)
    np.testing.assert_allclose(given.data[inner], whole.data[inner], atol=1e-9)
    dsm = als.chm(tiles, 1.0, dtm=None)
    assert np.nanmax(dsm.data) > np.nanmax(per_tile.data)
    with pytest.raises(ValueError, match="dtm must be"):
        als.chm(tiles, 1.0, dtm="lidr")


def test_single_tile_results_equal_the_single_cloud_functions(tiles, tmp_path):
    one = als.catalog(tiles.paths[0])
    cloud = io.read(tiles.paths[0])
    # Ground: the very same classes.
    out = als.classify_ground(one, tmp_path / "g", method="csf", cloth_resolution=1.0)
    got = io.read(out.paths[0])
    want = ground.classify_ground_csf(cloud, cloth_resolution=1.0)
    assert np.array_equal(got.attrs["classification"], want.attrs["classification"])
    assert np.array_equal(got.xyz, cloud.xyz)
    out = als.classify_ground(one, tmp_path / "p", method="pmf", cell_size=1.0)
    want = ground.classify_ground_pmf(cloud, cell_size=1.0)
    assert np.array_equal(io.read(out.paths[0]).attrs["classification"], want.attrs["classification"])
    # Normalising: the heights of normalize_height with make_dtm, away from the edge.
    norm = io.read(als.normalize(one, tmp_path / "n", dtm_resolution=1.0).paths[0])
    want = ground.normalize_height(cloud, ground.make_dtm(cloud, 1.0))
    b = one.bounds
    inside = (cloud.x > b[0] + 3) & (cloud.x < b[3] - 3) & (cloud.y > b[1] + 3) & (cloud.y < b[4] - 3)
    np.testing.assert_allclose(norm.attrs["height"][inside], want.attrs["height"][inside], atol=1e-9)
    # CHM: make_chm of the normalised cloud.
    chm = als.chm(one, 1.0, dtm_resolution=1.0)
    want = ground.make_chm(want, 1.0, bounds=(chm.xmin, chm.ymin, chm.xmax - 0.5, chm.ymax - 0.5))
    inner = _interior(chm, 3)
    np.testing.assert_allclose(chm.data[inner], want.data[inner], atol=1e-9)


def test_ground_classification_over_tiles_finds_the_ground(tiles, merged, tmp_path):
    out = als.classify_ground(tiles, tmp_path, method="csf", cloth_resolution=1.0, workers=2)
    assert len(out) == 4 and out.crs == "EPSG:32755"
    got = out.read()
    truth = merged.attrs["classification"] == 2
    order_got, order_want = np.lexsort(got.xyz.T), np.lexsort(merged.xyz.T)
    agree = (got.attrs["classification"][order_got] == 2) == truth[order_want]
    assert agree.mean() > 0.95
    # Outputs never overwrite the inputs.
    d = str(tiles.paths[0]).rsplit("/", 1)[0]
    with pytest.raises(ValueError, match="catalogue"):
        als.classify_ground(tiles, d)
    with pytest.raises(ValueError, match="unknown method"):
        als.classify_ground(tiles, tmp_path, method="mcc")


def test_normalize_replaces_z_or_adds_height(tiles, tmp_path):
    added = als.normalize(tiles, tmp_path / "h", buffer=10.0)
    replaced = als.normalize(tiles, tmp_path / "z", replace_z=True, buffer=10.0, format="las")
    a, r = added.read(), replaced.read()
    assert "height" in a.attrs and "elevation" in r.attrs
    np.testing.assert_allclose(r.z, a.attrs["height"], atol=1e-3)
    np.testing.assert_allclose(r.attrs["elevation"], a.z, atol=1e-9)
    ground_h = r.z[r.attrs["classification"] == 2]
    assert np.abs(ground_h).max() < 0.3
    # A CHM from the replaced z equals the CHM made on the fly.
    on_the_fly = als.chm(tiles, 1.0, buffer=10.0)
    from_z = als.chm(replaced, 1.0, dtm=None, buffer=10.0)
    np.testing.assert_allclose(from_z.data, on_the_fly.data, atol=2e-3)
    with pytest.raises(ValueError, match="needs"):
        als.normalize(tiles, tmp_path / "x", dtm=None)


def test_normalize_needs_ground(flight, tmp_path):
    cat = als.write_tiles(flight.points.without("classification"), tmp_path / "t", 30.0)
    with pytest.raises(ValueError, match="classification"):
        als.normalize(cat, tmp_path / "n")
    with pytest.raises(ValueError, match="classification"):
        als.dtm(cat, 1.0)


# ------------------------------------------------------------------ noise, retiling, thinning

def test_noise_filter_matches_the_merged_cloud(flight, tmp_path):
    rng = np.random.default_rng(5)
    birds = rng.uniform([0, 0, 40], [SIZE, SIZE, 60], (30, 3))
    pts = flight.points
    noisy = PointCloud(np.vstack([pts.xyz, birds]),
                       {"gps_time": np.concatenate([pts.attrs["gps_time"], np.full(30, -1.0)])})
    cat = als.write_tiles(noisy, tmp_path / "in", 30.0)
    out = als.filter(cat, tmp_path / "out", method="ror", radius=2.0, min_neighbors=3, buffer=3.0)
    got = out.read()
    merged = cat.read()
    want = filters.radius_outlier_removal(merged, 2.0, 3)
    assert len(got) == len(want)
    assert (got.attrs["gps_time"] == -1).sum() == 0
    flagged = als.filter(cat, tmp_path / "cls", classify=True, buffer=3.0).read()
    assert len(flagged) == len(merged)
    assert (flagged.attrs["classification"] == 7).sum() == len(merged) - len(want)
    sor = als.filter(cat, tmp_path / "sor", method="sor", k=8, std_ratio=3.0).read()
    assert (sor.attrs["gps_time"] == -1).sum() == 0 and len(sor) > 0.95 * len(merged)
    with pytest.raises(ValueError, match="radius"):
        als.filter(cat, tmp_path / "bad", radius=0.0)


def test_retile_keeps_every_point_once(tiles, merged, tmp_path):
    re = als.retile(tiles, tmp_path / "re", 25.0, origin=(0.0, 0.0))
    assert len(re) == 9 and re.n_points == len(merged)
    assert re.crs == "EPSG:32755"
    assert re.issues() == []
    names = sorted(p.rsplit("/", 1)[1] for p in re.paths)
    assert "25_25.laz" in names
    buffered = als.retile(tiles, tmp_path / "rb", 30.0, buffer=5.0)
    core = sum(int((io.read(p).attrs["buffer"] == 0).sum()) for p in buffered.paths)
    assert core == len(merged) and buffered.n_points > len(merged)


def test_decimate(tiles, merged, tmp_path):
    half = als.decimate(tiles, tmp_path / "r", fraction=0.5, seed=2)
    assert abs(half.n_points - 0.5 * len(merged)) <= 4
    again = als.decimate(tiles, tmp_path / "r2", fraction=0.5, seed=2, workers=1)
    assert np.array_equal(half.read().xyz, again.read().xyz)
    vox = als.decimate(tiles, tmp_path / "v", method="voxel", size=2.0).read()
    keys = np.floor(vox.xyz / 2.0)
    assert len(np.unique(keys, axis=0)) >= len(vox) - 4 * 30   # duplicates only across tile edges
    top = als.decimate(tiles, tmp_path / "t", method="highest", size=1.0).read()
    assert top.z.mean() > merged.z.mean()
    with pytest.raises(ValueError, match="fraction"):
        als.decimate(tiles, tmp_path / "x", fraction=1.5)
    with pytest.raises(ValueError, match="unknown method"):
        als.decimate(tiles, tmp_path / "x", method="poisson")


def test_write_tiles_round_trips_the_attributes(flight, tmp_path):
    cat = flight.write_tiles(tmp_path, size=30.0, format="las")
    back = cat.read()
    order = np.lexsort((back.attrs["return_number"], back.attrs["gps_time"]))
    pts = flight.points
    want = np.lexsort((pts.attrs["return_number"], pts.attrs["gps_time"]))
    for k in ("return_number", "number_of_returns", "point_source_id", "intensity",
              "classification"):
        assert np.array_equal(back.attrs[k][order], pts.attrs[k][want]), k
    np.testing.assert_allclose(back.attrs["scan_angle"][order], pts.attrs["scan_angle"][want],
                               atol=0.0061)   # LAS stores 0.006 degree steps
    assert np.abs(back.xyz[order] - pts.xyz[want]).max() <= 0.0005 + 1e-9
    with pytest.raises(ValueError, match="size"):
        als.write_tiles(pts, tmp_path / "x", 0.0)
    with pytest.raises(ValueError, match="format"):
        als.write_tiles(pts, tmp_path / "x", 10.0, format="ply")
    assert len(als.write_tiles(pts[:0], tmp_path / "empty", 10.0)) == 0

