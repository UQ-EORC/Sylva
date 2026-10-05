# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Small public functions, properties and methods, each checked against an
analytic answer or an independent computation."""

import sys

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from sylva import PointCloud, Raster, als, canopy, io, trees, waveform
from sylva.als import metrics as als_metrics
from sylva.coreg import simulate
from sylva.coreg.icp import ICPConfig, ICPTarget, plane_information
from sylva.coreg.matching import MatchResult
from sylva.coreg.pipeline import ScanFeatures
from sylva.coreg.stems import Stem, StemMap

# --------------------------------------------------------------------------- #
# trees.convex_hull_area
# --------------------------------------------------------------------------- #


def test_convex_hull_area_matches_scipy():
    rng = np.random.default_rng(0)
    for n in (3, 10, 1000):
        xy = rng.normal(size=(n, 2)) * [3.0, 1.0] + [500.0, -20.0]
        assert trees.convex_hull_area(xy) == pytest.approx(ConvexHull(xy).volume, rel=1e-12)


def test_convex_hull_area_of_a_regular_polygon_with_interior_points():
    n, r = 12, 2.0
    t = 2 * np.pi * np.arange(n) / n
    ring = np.column_stack([r * np.cos(t), r * np.sin(t)])
    inner = np.random.default_rng(1).uniform(-1, 1, (200, 2))
    area = 0.5 * n * r**2 * np.sin(2 * np.pi / n)
    assert trees.convex_hull_area(np.vstack([inner, ring])) == pytest.approx(area, rel=1e-12)


def test_convex_hull_area_ignores_non_finite_rows_and_needs_three_points():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [np.nan, 5.0], [np.inf, 0.0]])
    assert trees.convex_hull_area(square) == pytest.approx(1.0)
    assert np.isnan(trees.convex_hull_area(np.array([[0.0, 0.0], [1.0, 1.0]])))
    assert np.isnan(trees.convex_hull_area(np.array([[0.0, 0.0]] * 5 + [[1.0, 1.0]])))
    assert np.isnan(trees.convex_hull_area(np.zeros((0, 2))))


# --------------------------------------------------------------------------- #
# Raster.to_geotiff
# --------------------------------------------------------------------------- #


def test_to_geotiff_round_trip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    data = np.arange(12, dtype=float).reshape(3, 4)
    data[1, 2] = np.nan
    r = Raster(data, 500.0, 7000.0, 2.0, crs="EPSG:7855")
    r.to_geotiff(tmp_path / "r.tif")
    with rasterio.open(tmp_path / "r.tif") as src:
        back = src.read(1)
        assert src.crs.to_epsg() == 7855
        assert src.transform.c == 500.0 and src.transform.f == 7006.0 and src.transform.a == 2.0
    # The file is north-up; Raster row 0 is the southern edge.
    np.testing.assert_array_equal(np.flipud(back), data.astype(np.float32))


def test_to_geotiff_needs_no_rasterio(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rasterio", None)       # makes `import rasterio` fail
    r = Raster(np.zeros((2, 2)), 0.0, 0.0, 1.0, crs="EPSG:28356")
    r.to_geotiff(tmp_path / "r.tif")
    assert (tmp_path / "r.tif").read_bytes()[:4] == b"II*\x00"
    with pytest.raises(ValueError):
        r.to_geotiff(tmp_path / "bad.tif", crs="not a crs")


def test_ascii_grid_keeps_its_crs_in_a_prj(tmp_path):
    r = Raster(np.arange(6.0).reshape(2, 3), 500.0, 7000.0, 2.0, crs="EPSG:28356")
    r.to_ascii_grid(tmp_path / "r.asc")
    assert "MGA zone 56" in (tmp_path / "r.prj").read_text()
    back = Raster.from_ascii_grid(tmp_path / "r.asc")
    assert back.crs == "EPSG:28356"
    np.testing.assert_array_equal(back.data, r.data)
    Raster(r.data, 500.0, 7000.0, 2.0).to_ascii_grid(tmp_path / "plain.asc")
    assert not (tmp_path / "plain.prj").exists()
    assert Raster.from_ascii_grid(tmp_path / "plain.asc").crs is None


# --------------------------------------------------------------------------- #
# coreg: plane information, match results, stem maps, simulation, scan features
# --------------------------------------------------------------------------- #


def _grid_plane(n=40, size=10.0):
    u = np.linspace(0, size, n)
    x, y = np.meshgrid(u, u)
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def test_plane_information_of_flat_ground_leaves_three_directions_free():
    ground = _grid_plane()
    cfg = ICPConfig(voxel_sizes=(0.5,), max_distances=(1.0,), max_points=100_000)
    info = plane_information(ground, ground, np.eye(4), cfg)
    assert info is not None and info.n >= 10
    assert info.sigma == pytest.approx(0.0, abs=1e-9)
    assert info.hessian.shape == (6, 6)
    np.testing.assert_allclose(info.hessian, info.hessian.T, atol=1e-9)
    # xi = [omega, v]: a horizontal plane fixes height (v_z), roll and pitch
    # (omega_x, omega_y) and leaves x, y and yaw free.
    h = info.hessian
    for free in (2, 3, 4):
        assert np.allclose(h[free], 0.0, atol=1e-9)
    assert np.linalg.matrix_rank(h[np.ix_([0, 1, 5], [0, 1, 5])], tol=1e-6) == 3


def test_plane_information_of_a_corner_pins_every_direction():
    g = _grid_plane(30, 6.0)
    walls = [g, g[:, [2, 0, 1]] + [0.0, 0.0, 0.0], g[:, [0, 2, 1]]]      # z = 0, x = 0, y = 0
    rng = np.random.default_rng(2)
    pts = np.vstack(walls) + rng.normal(0, 1e-4, (sum(len(w) for w in walls), 3))
    cfg = ICPConfig(voxel_sizes=(0.4,), max_distances=(1.0,), max_points=100_000)
    info = plane_information(pts, pts, np.eye(4), cfg)
    assert np.linalg.matrix_rank(info.hessian, tol=1e-3 * np.abs(info.hessian).max()) == 6
    # A prepared target gives the same answer as the raw points.
    again = plane_information(pts, ICPTarget(pts, cfg), np.eye(4), cfg)
    np.testing.assert_allclose(again.hessian, info.hessian, rtol=1e-9, atol=1e-12)


def test_plane_information_is_none_with_too_few_correspondences():
    few = _grid_plane(3, 1.0)                                  # 9 points
    cfg = ICPConfig(voxel_sizes=(0.01,), max_distances=(0.1,))
    assert plane_information(few, few, np.eye(4), cfg) is None


def test_inlier_fraction_is_over_the_smaller_map():
    m = MatchResult(np.eye(4), n_inliers=6, inlier_rmse=0.01, score=1.0, n_source=8, n_target=20)
    assert m.inlier_fraction == 6 / 8
    # Empty maps: no division by zero.
    assert MatchResult(np.eye(4), 0, 0.0, 0.0).inlier_fraction == 0.0


def test_sorted_by_quality_is_best_first_with_stable_ties():
    rng = np.random.default_rng(3)
    stems = [Stem(float(i), 0.0, 0.0, 0.3, n_slices=int(rng.integers(0, 10)),
                  rmse=float(rng.choice([0.0, 0.01, 0.02])), coverage=float(rng.choice([0.5, 1.0])))
             for i in range(40)]
    sm = StemMap(stems, name="scan")
    # stem quality = clip(1 / (1 + rmse / 0.01) * coverage * min(n_slices / 6, 1), 0, 1)
    q = np.array([1 / (1 + s.rmse / 0.01) * s.coverage * min(s.n_slices / 6, 1) for s in stems])
    out = sm.sorted_by_quality()
    assert out.name == "scan" and len(out) == 40
    np.testing.assert_allclose([s.quality for s in out], np.sort(q)[::-1])
    assert [s.x for s in out] == [stems[i].x for i in np.argsort(-q)]
    assert [s.x for s in sm.top(5)] == [s.x for s in out.stems[:5]]


def _tree(lean_deg=10.0, azimuth_deg=30.0):
    t, a = np.radians(lean_deg), np.radians(azimuth_deg)
    lean = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])
    return simulate.Tree(x=2.0, y=-1.0, base_z=100.0, dbh=0.4, height=20.0, lean=lean, taper=0.01,
                         crown_base=10.0)


def test_tree_axis_point_rises_by_h_and_leans_by_tan():
    tree = _tree(lean_deg=10.0, azimuth_deg=30.0)
    h = np.array([0.0, 1.3, 10.0])
    p = tree.axis_point(h)
    np.testing.assert_allclose(p[:, 2], 100.0 + h)
    offset = np.hypot(p[:, 0] - 2.0, p[:, 1] + 1.0)
    np.testing.assert_allclose(offset, h * np.tan(np.radians(10.0)), atol=1e-12)
    np.testing.assert_allclose(np.degrees(np.arctan2(p[2, 1] + 1.0, p[2, 0] - 2.0)), 30.0)
    assert tree.axis_point(5.0).shape == (1, 3)


def test_tree_radius_tapers_linearly_from_breast_height_to_a_floor():
    tree = _tree()
    assert tree.radius_at(1.3) == pytest.approx(0.2)
    assert tree.radius_at(11.3) == pytest.approx(0.1)
    assert tree.radius_at(0.3) == pytest.approx(0.21)
    np.testing.assert_allclose(tree.radius_at([30.0, 100.0]), 0.01)   # never below 1 cm


def test_overlap_stems_counts_trees_visible_from_both_scans(small_survey):
    v = small_survey.visible
    assert small_survey.overlap_stems(0, 1) == small_survey.overlap_stems(1, 0)
    assert small_survey.overlap_stems(0, 1) == int(np.sum(v[0] & v[1]))
    assert small_survey.overlap_stems(0, 0) == int(v[0].sum())
    assert 0 < small_survey.overlap_stems(0, 1) <= min(v[0].sum(), v[1].sum())


def test_scan_location_moves_the_origin_by_the_pose():
    f = ScanFeatures("s", 0, None, StemMap([]), np.zeros((0, 3)), origin=np.array([1.0, 2.0, 3.0]))
    yaw = np.radians(90)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    pose[:3, 3] = [10.0, 20.0, 30.0]
    np.testing.assert_allclose(f.location(pose), [10.0 - 2.0, 20.0 + 1.0, 33.0], atol=1e-12)
    np.testing.assert_allclose(f.location(np.eye(4)), [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# canopy.GapProfile.zenith, als.Tile.xy_bounds, als_metrics.pixel_metrics
# --------------------------------------------------------------------------- #


def test_gap_profile_zenith_is_the_ring_centres():
    np.testing.assert_allclose(canopy.GapProfile.empty().zenith, np.arange(7.5, 70.1, 5.0))
    prof = canopy.GapProfile.empty(zenith_edges=[0.0, 10.0, 40.0, 90.0])
    np.testing.assert_allclose(prof.zenith, [5.0, 25.0, 65.0])
    assert prof.zenith.shape == (prof.hits.shape[0],)


def test_tile_xy_bounds_are_the_header_extent(tmp_path):
    rng = np.random.default_rng(4)
    xyz = rng.uniform([300.0, 6000.0, 10.0], [400.0, 6050.0, 40.0], (500, 3))
    io.write(PointCloud(xyz), tmp_path / "t.las")
    tile = als.catalog(tmp_path).tiles[0]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    np.testing.assert_allclose(tile.xy_bounds, [lo[0], lo[1], hi[0], hi[1]], atol=0.001)
    assert tile.xy_bounds == (tile.bounds[0], tile.bounds[1], tile.bounds[3], tile.bounds[4])


def test_pixel_metrics_is_lidr_name_for_grid_metrics():
    assert als_metrics.pixel_metrics is als_metrics.grid_metrics
    assert als.pixel_metrics is als_metrics.grid_metrics


# --------------------------------------------------------------------------- #
# waveform: Waveform.times, Waveforms.n_waveforms, Echoes.cross_section
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def few_waveforms():
    from sylva import Shots, synthetic

    n = 7
    count = np.ones(n, dtype=np.int64)
    shots = Shots(np.tile([0.0, 0.0, 50.0], (n, 1)), np.tile([0.0, 0.0, -1.0], (n, 1)),
                  np.arange(n), count, np.linspace(20, 40, n), {"amplitude": np.full(n, 80.0)})
    return synthetic.waveforms(shots, pulse_width=1.5, noise=0.5, seed=1)


def test_waveform_times_and_count(few_waveforms):
    wf, truth = few_waveforms
    assert wf.n_waveforms == len(wf) == len(wf.anchor) == 7
    w = wf[3]
    t = w.times()
    np.testing.assert_allclose(t, np.arange(len(w.samples)) * w.interval)
    # Sample k lies at anchor + direction * metres_per_ns * (offset + t_k).
    expected = w.anchor + np.outer(w.metres_per_ns * (w.offset + t), w.direction)
    np.testing.assert_allclose(w.positions(), expected, atol=1e-9)


def test_echo_cross_section_is_c_r4_amplitude_width():
    e = waveform.Echoes(waveform=[0, 1, 2], time=[1.0, 2.0, 3.0], amplitude=[10.0, 20.0, 5.0],
                        width=[1.5, 2.0, 1.0], xyz=np.zeros((3, 3)), range=[10.0, np.nan, 20.0])
    expected = 3.0 * np.array([10.0, np.nan, 20.0]) ** 4 * e.amplitude * e.width
    np.testing.assert_allclose(e.cross_section(3.0), expected)
    assert np.isnan(e.cross_section()[1])
    # Ranges from elsewhere replace the unknown one.
    np.testing.assert_allclose(e.cross_section(range=[1.0, 2.0, 3.0]),
                               np.array([1.0, 16.0, 81.0]) * e.amplitude * e.width)
