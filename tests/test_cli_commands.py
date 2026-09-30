# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The TLS commands of the command line on synthetic scenes with known answers,
and the errors they report."""

import numpy as np
import pytest

from sylva import PointCloud, Raster, Shots, canopy, cli, io, voxels


def run(*argv):
    cli.main(["--no-progress", *map(str, argv)])


def fails(capsys, *argv) -> str:
    """Run a command that must fail; return its message."""
    with pytest.raises(SystemExit) as exc:
        run(*argv)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("sylva: error: ")
    return err


# --------------------------------------------------------------------------- #
# convert, info and the errors every command reports the same way
# --------------------------------------------------------------------------- #


@pytest.fixture
def small_cloud():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0, 2, (2000, 3)) + [500.0, 7000.0, 30.0]
    return PointCloud(xyz, {"intensity": rng.integers(0, 60000, len(xyz)).astype(np.uint16)})


def test_convert_keeps_points_and_attributes(small_cloud, tmp_path, capsys):
    src, dst = tmp_path / "a.laz", tmp_path / "a.ply"
    io.write(small_cloud, src)
    run("convert", src, dst)
    assert capsys.readouterr().out == f"wrote 2,000 points to {dst}\n"
    back = io.read(dst)
    np.testing.assert_allclose(back.xyz, small_cloud.xyz, atol=0.0005 + 1e-9)  # LAS millimetres
    np.testing.assert_array_equal(back.attrs["intensity"], small_cloud.attrs["intensity"])


def test_convert_voxel_keeps_one_point_per_occupied_voxel(small_cloud, tmp_path):
    src, dst = tmp_path / "a.ply", tmp_path / "thin.xyz"
    io.write(small_cloud, src)
    run("convert", src, dst, "--voxel", 0.5)
    back = io.read(dst)
    # The grid starts at the cloud's minimum corner (filters.voxel_downsample).
    xyz = small_cloud.xyz
    keys = np.floor((xyz - xyz.min(axis=0)) / 0.5).astype(np.int64)
    assert len(back) == len(np.unique(keys, axis=0)) == 64
    got = np.floor((back.xyz - xyz.min(axis=0)) / 0.5 + 1e-6).astype(np.int64)
    assert len(np.unique(got, axis=0)) == len(back), "two points kept in one voxel"


def test_info_prints_an_uninterpretable_crs_as_stored(small_cloud, tmp_path, capsys):
    src = tmp_path / "odd_crs.las"
    odd = 'LOCAL_CS["somewhere",UNIT["metre",1]]'
    io.write(PointCloud(small_cloud.xyz, crs=odd), src)
    run("info", src)
    out = capsys.readouterr().out
    assert f"{src}: 2,000 points" in out
    assert f"  crs: {odd}" in out


def test_missing_input_exits_with_status_1(tmp_path, capsys):
    for name in ("nothing.laz", "nothing.ply", "nothing.xyz"):
        err = fails(capsys, "convert", tmp_path / name, tmp_path / "out.ply")
        assert err.startswith(f"sylva: error: {tmp_path / name}: "), "the message names the file"


def test_unknown_output_format_is_an_error(small_cloud, tmp_path, capsys):
    src = tmp_path / "a.ply"
    io.write(small_cloud, src)
    err = fails(capsys, "convert", src, tmp_path / "out.docx")
    assert "docx" in err
    assert not (tmp_path / "out.docx").exists()


def test_reproject_without_a_crs_asks_for_one(small_cloud, tmp_path, capsys):
    src = tmp_path / "nocrs.ply"
    io.write(small_cloud, src)
    err = fails(capsys, "reproject", src, tmp_path / "out.laz", "--to", "EPSG:4326")
    assert err == f"sylva: error: {src} declares no CRS; give it with --from\n"


def test_progress_bar_path_runs_the_command(small_cloud, tmp_path):
    src, dst = tmp_path / "a.ply", tmp_path / "b.ply"
    io.write(small_cloud, src)
    cli.main(["convert", str(src), str(dst)])        # without --no-progress
    assert len(io.read(dst)) == len(small_cloud)


# --------------------------------------------------------------------------- #
# ground and chm on a sloping plane with a shrub layer at a known height
# --------------------------------------------------------------------------- #


def _terrain(a=0.02, b=-0.01, c=50.0):
    return lambda x, y: a * x + b * y + c


@pytest.fixture(scope="module")
def sloping_plot():
    rng = np.random.default_rng(11)
    surface = _terrain()
    xy = rng.uniform(0, 20, (30000, 2))
    ground = np.column_stack([xy, surface(xy[:, 0], xy[:, 1]) + rng.normal(0, 0.005, len(xy))])
    # A layer of vegetation 2 m above the terrain over half the plot.
    vxy = rng.uniform([0, 0], [10, 20], (6000, 2))
    veg = np.column_stack([vxy, surface(vxy[:, 0], vxy[:, 1]) + 2.0])
    return PointCloud(np.vstack([ground, veg]))


@pytest.mark.parametrize("method", ["csf", "pmf"])
def test_ground_writes_heights_and_dtm_beside_the_input(sloping_plot, tmp_path, capsys, method):
    src = tmp_path / "plot.laz"
    io.write(sloping_plot, src)
    run("ground", src, "--method", method, "--resolution", 1.0, "--dtm", tmp_path / "dtm.asc")
    out = io.read(tmp_path / "plot_norm.laz")
    n_ground = int((out.attrs["classification"] == 2).sum())
    written = tmp_path / "plot_norm.laz"
    assert capsys.readouterr().out == f"{n_ground:,} ground points; wrote {written}\n"
    veg = np.arange(len(out)) >= 30000
    np.testing.assert_allclose(out.attrs["height"][veg], 2.0, atol=0.1)
    assert np.abs(out.attrs["height"][~veg]).max() < 0.1
    # The DTM follows the plane at its cell centres.
    dtm = Raster.from_ascii_grid(tmp_path / "dtm.asc")
    ys, xs = np.mgrid[0:dtm.data.shape[0], 0:dtm.data.shape[1]]
    cx = dtm.xmin + (xs + 0.5) * dtm.resolution
    cy = dtm.ymin + (ys + 0.5) * dtm.resolution
    inside = (cx > 1) & (cx < 19) & (cy > 1) & (cy < 19)
    np.testing.assert_allclose(dtm.data[inside], _terrain()(cx, cy)[inside], atol=0.06)


def test_chm_is_the_highest_height_per_cell(tmp_path, capsys):
    rng = np.random.default_rng(12)
    xy = rng.uniform(0, 10, (5000, 2))
    h = np.where(xy[:, 0] < 5, rng.uniform(0, 12, len(xy)), rng.uniform(0, 1.5, len(xy)))
    cloud = PointCloud(np.column_stack([xy, h + 100]), {"height": h})
    src = tmp_path / "norm.laz"
    io.write(cloud, src)
    run("chm", src, "--resolution", 1.0)
    chm = Raster.from_ascii_grid(tmp_path / "norm_chm.asc")
    # Brute force: the maximum of the (LAS-rounded) heights in each 1 m cell.
    back = io.read(src)
    i = np.floor((back.xyz[:, 0] - chm.xmin) / chm.resolution).astype(int)
    j = np.floor((back.xyz[:, 1] - chm.ymin) / chm.resolution).astype(int)
    expected = np.full(chm.data.shape, -np.inf)
    np.maximum.at(expected, (j, i), back.attrs["height"])
    expected[np.isinf(expected)] = 0.0                          # cells without points
    np.testing.assert_allclose(chm.data, expected, atol=1e-3)   # the grid is written to 1 mm
    cover = np.mean(expected > 2)
    assert capsys.readouterr().out.endswith(f"cover(>2m)={cover:.2f}\n")


def test_pad_prints_the_profile_of_the_library(tmp_path, capsys):
    rng = np.random.default_rng(13)
    xyz = rng.uniform([0, 0, 0], [4, 4, 6], (4000, 3))
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    src = tmp_path / "c.ply"
    io.write(cloud, src)
    run("pad", src, "--voxel", 1.0)
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines[0] == "height,pad"
    rows = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    z, pad = canopy.pad_profile_voxel(io.read(src), voxel_size=1.0)
    np.testing.assert_allclose(rows[:, 0], z, atol=0.005)
    np.testing.assert_allclose(rows[:, 1], pad, atol=5e-5)
    assert err == f"# PAI = {np.sum(pad) * 1.0:.3f}\n"


# --------------------------------------------------------------------------- #
# shots and voxel: a ray cloud through a turbid slab
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ray_cloud():
    """raycloudtools convention: nx, ny, nz point from each end back to the sensor."""
    rng = np.random.default_rng(14)
    n = 4000
    xy = rng.uniform(0.05, 1.95, (n, 2))
    origin = np.column_stack([xy, np.full(n, 5.0)])
    free = rng.exponential(1 / 0.8, n)
    depth = np.where(free < 1.0, 3.0 + free, 5.0)        # slab z in [1, 2] over a floor at 0
    ends = origin - [0, 0, 1] * depth[:, None]
    to_sensor = origin - ends
    alpha = np.full(n, 255, np.uint8)
    return PointCloud(ends, {"nx": to_sensor[:, 0], "ny": to_sensor[:, 1], "nz": to_sensor[:, 2],
                             "alpha": alpha})


def test_shots_converts_a_ray_cloud(ray_cloud, tmp_path, capsys):
    src = tmp_path / "rays.ply"
    io.write(ray_cloud, src)
    run("shots", src, "--double")
    assert capsys.readouterr().out == f"4000 pulses, 4000 echoes -> {tmp_path / 'rays.parquet'}\n"
    shots = Shots.load(tmp_path / "rays.parquet")
    np.testing.assert_allclose(shots.echo_xyz(), ray_cloud.xyz, atol=1e-6)
    np.testing.assert_allclose(shots.origin[:, 2], 5.0, atol=1e-6)


def test_voxel_writes_what_the_library_writes(ray_cloud, tmp_path, capsys):
    src = tmp_path / "rays.ply"
    io.write(ray_cloud, src)
    run("shots", src, "--double")                  # exact angles and ranges, as in the ray cloud
    parquet = tmp_path / "rays.parquet"
    capsys.readouterr()
    bounds = (0, 0, 0, 2, 2, 3)
    run("voxel", parquet, "--voxel", 1.0, "--bounds", *bounds, "--neighbour-priors", 0)
    out = capsys.readouterr().out
    assert out.startswith("4000 pulses -> ") and out.endswith(f"to {tmp_path / 'rays.vox'}\n")
    grid = voxels.ray_voxelize(str(parquet), 1.0, ((0, 0, 0), (2, 2, 3)))
    grid.write(tmp_path / "api.vox")
    assert (tmp_path / "rays.vox").read_text() == (tmp_path / "api.vox").read_text()
    # A ray cloud is loaded whole, and gives the same grid.
    run("voxel", src, tmp_path / "from_ply.vox", "--voxel", 1.0, "--bounds", *bounds)
    assert (tmp_path / "from_ply.vox").read_text() == (tmp_path / "rays.vox").read_text()
    # The slab's attenuation is recovered from the file (FPL, no leaf-size correction).
    run("voxel", parquet, tmp_path / "fpl.vox", "--voxel", 1.0, "--bounds", *bounds,
        "--average-leaf-area", 0)
    g = voxels.ray_voxelize(str(parquet), 1.0, ((0, 0, 0), (2, 2, 3)), average_leaf_area=0)
    np.testing.assert_allclose(g["attenuation_fpl"][1], 0.8, atol=0.1)


def test_voxel_rejects_bad_options_with_a_message(ray_cloud, tmp_path, capsys):
    src = tmp_path / "rays.ply"
    io.write(ray_cloud, src)
    err = fails(capsys, "voxel", src, "--voxel", 0)
    assert "voxel_size must be positive" in err


# --------------------------------------------------------------------------- #
# trees: the segmented cloud beside the input
# --------------------------------------------------------------------------- #


def test_trees_segment_flag_writes_the_labelled_cloud(tmp_path, capsys):
    from test_cli import _two_tree_plot

    src = tmp_path / "plot.laz"
    io.write(_two_tree_plot(), src)
    run("trees", src, "--segment")
    assert capsys.readouterr().out == f"2 trees -> {tmp_path / 'plot_trees.csv'}\n"
    seg = io.read(tmp_path / "plot_segmented.laz")
    ids = seg.attrs["tree_id"]
    # Each stem's points carry one tree id (or -1, no tree), different for the two stems.
    left, right = ids[seg.xyz[:, 0] < 3], ids[seg.xyz[:, 0] > 3]
    ids_left, ids_right = set(np.unique(left)) - {-1}, set(np.unique(right)) - {-1}
    assert len(ids_left) == 1 and len(ids_right) == 1 and ids_left != ids_right
    assert np.mean(left >= 0) > 0.9 and np.mean(right >= 0) > 0.9
    rows = (tmp_path / "plot_trees.csv").read_text().splitlines()
    assert rows[0].startswith("tree_id,") and len(rows) == 3


# --------------------------------------------------------------------------- #
# coreg: two simulated scans given as files
# --------------------------------------------------------------------------- #


def test_coreg_registers_scan_files(small_survey, tmp_path, capsys):
    from sylva.coreg import invert, transform_difference
    from sylva.coreg.pipeline import load_transforms

    paths = []
    for k, cloud in enumerate(small_survey.clouds):
        paths.append(tmp_path / f"scan{k}.laz")
        io.write(cloud, paths[-1])
    out = tmp_path / "result"
    run("coreg", *paths, "-o", out, "--quiet", "--reference", "scan0", "--merged",
        tmp_path / "merged.laz", "--voxel", 0.1)
    msg = capsys.readouterr().out
    assert msg.startswith(f"2 of 2 scans registered -> {out}; merged ")
    assert (out / "report.txt").read_text().strip()
    saved = load_transforms(out / "transforms.json")
    assert sorted(saved) == ["scan0", "scan1"]
    # scan1.dat maps scan 1 into scan 0's frame: compare with the truth.
    got = np.loadtxt(out / "scan1.dat")
    np.testing.assert_allclose(got, saved["scan1"], atol=1e-11)          # written to 12 decimals
    truth = invert(small_survey.true_transforms[0]) @ small_survey.true_transforms[1]
    dt, dr = transform_difference(got, truth)
    assert dt < 0.05 and dr < 0.2
    np.testing.assert_allclose(np.loadtxt(out / "scan0.dat"), np.eye(4), atol=1e-12)
    merged = io.read(tmp_path / "merged.laz")
    assert 0 < len(merged) < sum(len(c) for c in small_survey.clouds)


def test_coreg_project_without_scans_is_an_error(tmp_path, capsys):
    from test_riscan import RSP

    root = tmp_path / "Empty.RiSCAN"
    root.mkdir()
    (root / "project.rsp").write_text(RSP.replace("<file>s1.rxp</file>", ""))
    err = fails(capsys, "coreg", root, "--quiet")
    assert err == f"sylva: error: no scans found in {root}\n"


def test_qsm_plot_reports_the_trees_it_skips(tmp_path, capsys):
    from test_cli import _two_tree_plot

    src = tmp_path / "plot.laz"
    io.write(_two_tree_plot(), src)
    run("qsm-plot", src, "--no-wood", "--min-points", 10**6)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("0 QSMs, 2 skipped, 0.000 m3 of wood")
    assert [line.split(":")[0] for line in out[1:]] == ["  skipped 1", "  skipped 2"]
