import numpy as np
import pytest

from sylva import PointCloud, Raster, Shots, voxels
from sylva.qsm import QSM


def down_shots(xy, ranges, z0=10.0):
    """Single-echo pulses straight down from ``z0`` (``ranges`` NaN = no echo)."""
    n = len(xy)
    hit = np.isfinite(ranges)
    count = hit.astype(np.int64)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    return Shots(np.column_stack([xy, np.full(n, z0)]), np.tile([0.0, 0.0, -1.0], (n, 1)),
                 start, count, ranges[hit])


@pytest.fixture(scope="module")
def slab():
    """A turbid 1 m slab (z in [1, 2]) with attenuation 0.8 over a floor at z = 0."""
    rng = np.random.default_rng(1)
    n = 30000
    xy = rng.uniform(0, 2, (n, 2))
    free = rng.exponential(1 / 0.8, n)
    ranges = np.where(free < 1.0, 8.0 + free, 10.0)
    shots = down_shots(xy, ranges)
    shots.echo_attrs["classification"] = np.where(ranges >= 10.0, 2, 5).astype(np.uint8)
    return shots


BOUNDS = ((0, 0, 0), (2, 2, 3))


def test_estimators_recover_attenuation(slab):
    g = voxels.ray_voxelize(slab, 1.0, BOUNDS, ground_class=2, laser="VZ-400",
                            attenuation=["fpl", "ppl", "transmittance"], average_leaf_area=0)
    assert g.shape == (2, 2, 3) and g.num_hits.shape == (3, 2, 2)
    assert g.num_hits[0].sum() == 0 and g.num_hits[2].sum() == 0
    assert np.all(g.state[1] == voxels.STATES["filled"])
    assert np.all(g.state[2] == voxels.STATES["empty"])
    for name in ("attenuation_fpl", "attenuation_ppl", "attenuation_transmittance"):
        np.testing.assert_allclose(g[name][1], 0.8, atol=0.05)
        assert np.all(g[name][2] == 0)
    np.testing.assert_allclose(g.pad_ppl[1], 1.6, atol=0.1)  # spherical G = 0.5
    np.testing.assert_allclose(g.transmittance[1], np.exp(-0.8), atol=0.03)
    np.testing.assert_allclose(g.mean_zenith_angle[1], 180, atol=0.05)  # float32 sums
    assert g.profile("pad_ppl")[1] == pytest.approx(1.6, abs=0.1)
    assert "bs_entering" in g.fields and "ppl_lambda" in g.fields
    assert not hasattr(g, "not_a_metric")


def test_ground_from_dtm_and_leaf_wood(slab):
    rng = np.random.default_rng(2)
    slab.echo_attrs["classification"][:] = np.where(
        slab.echo_range >= 10.0, 2, rng.choice([4, 6], slab.n_echoes, p=[0.75, 0.25]))
    dtm = Raster(np.zeros((4, 4)), -1.0, -1.0, 1.0)
    g = voxels.ray_voxelize(slab, 1.0, BOUNDS, dtm=dtm, leaf_classes=[4], wood_classes=[6],
                            attenuation="transmittance", occlusion=True)
    assert g.num_hits[0].sum() == 0, "floor echoes are within ground_distance of the DTM"
    assert g.num_hit_leaf.sum() + g.num_hit_wood.sum() == g.num_hits.sum()
    np.testing.assert_allclose(g.distance_from_ground[:, 0, 0], [0.5, 1.5, 2.5])
    ratio = g.wad_transmittance[1] / g.pad_transmittance[1]
    np.testing.assert_allclose(ratio, 0.25, atol=0.03)
    assert g.num_beams_occluded[1].sum() > 0
    assert g.num_beams_occluded[2].sum() == 0


def test_misses_cross_the_whole_grid():
    xy = np.full((10, 2), 0.5)
    g = voxels.ray_voxelize(down_shots(xy, np.full(10, np.nan)), 1.0, ((0, 0, 0), (1, 1, 3)))
    assert np.all(g.num_beams.ravel() == 10)
    assert np.all(g.num_unbound_rays.ravel() == 10)
    assert g.num_hits.sum() == 0
    np.testing.assert_allclose(g.path_length.ravel(), 10.0, rtol=1e-5)


def test_inclination_wood_volume_and_files(tmp_path):
    rng = np.random.default_rng(3)
    n = 4000
    xy = rng.uniform(0, 4, (n, 2))
    shots = down_shots(xy, np.full(n, 8.5))  # a horizontal sheet at z = 1.5
    shots.echo_attrs["tree_id"] = (xy[:, 0] > 2).astype(np.int32) + 7
    g = voxels.ray_voxelize(shots, 1.0, ((0, 0, 0), (4, 4, 3)), inclination=True, subvoxel_split=2)
    iad = g.tree_iad
    assert sorted(iad) == [7, 8]
    assert iad[7]["piad"][0] > 0.99 and iad[7]["piad_de_wit"] == "planophile"
    assert iad[7]["g_plant"] > 0.95
    assert set(np.unique(g.predominant_tree)) == {-1, 7, 8}
    assert g.pad_fpl[1].min() > 0
    assert g.exploration_rate.max() == 1.0

    cyl = np.array([[2.0, 2.0, 0.1, 0, 0, 1, 2.8, 0.25, -1, 0, 0, 0]], dtype=float)
    g.add_wood_volume(QSM(cyl))
    assert g.wood_volume.sum() == pytest.approx(np.pi * 0.25**2 * 2.8, rel=1e-4)
    assert g.wood_volume_density.max() <= 1.0

    n_vox = g.write(tmp_path / "plot.vox")
    lines = (tmp_path / "plot.vox").read_text().splitlines()
    assert lines[0] == "VOXEL SPACE"
    header = next(i for i, line in enumerate(lines) if line.startswith("i j k"))
    assert "#split:4 4 3" in lines[:header]
    assert "pad_fpl" in lines[header] and "wood_volume_density" in lines[header]
    assert len(lines) - header - 1 == n_vox == int(g.observed.sum())
    assert g.write(tmp_path / "plot.txt", filled_only=True) == int((g.num_hits > 0).sum())
    g.write_iad_csv(tmp_path / "iad.csv")
    assert (tmp_path / "iad.csv").read_text().splitlines()[1].startswith("7,planophile,")


def test_ray_cloud_pulses_are_regrouped():
    # Two pulses with two returns each, interleaved in the file, plus one miss.
    start = np.array([[0, 0, 10.0]] * 5)
    end = np.array([[0.5, 0.5, 6.0], [1.5, 0.5, 5.0], [0, 0, 0], [0, 0, 0], [0.5, 1.5, 0.0]])
    end[2] = start[2] + 2.0 * (end[0] - start[0])  # second returns lie on the first ones' rays
    end[3] = start[3] + 1.8 * (end[1] - start[1])
    pc = PointCloud(end, {
        "sx": (start - end)[:, 0], "sy": (start - end)[:, 1], "sz": (start - end)[:, 2],
        "gps_time": np.array([1.0, 2.0, 1.0, 2.0, 3.0]),
        "number_of_returns": np.array([2, 2, 2, 2, 1], dtype=np.uint8),
        "bound": np.array([1, 1, 1, 1, 0], dtype=np.uint8),
    })
    shots = Shots.from_ray_cloud(pc)
    assert shots.n_shots == 3
    assert list(shots.echo_count) == [2, 2, 0]
    np.testing.assert_allclose(shots.echo_range[1] / shots.echo_range[0], 2.0)
    np.testing.assert_allclose(shots.echo_xyz(), end[[0, 2, 1, 3]], atol=1e-9)


def test_leaf_projection():
    theta = np.radians([0, 30, 57.5, 80])
    np.testing.assert_allclose(voxels.leaf_projection(theta), 0.5)
    # All distributions cross near G = 0.5 at the hinge angle.
    for lad in ("planophile", "erectophile", "uniform"):
        assert voxels.leaf_projection(theta, lad)[2] == pytest.approx(0.5, abs=0.04)
    assert voxels.leaf_projection(0.0, "planophile")[0] > 0.8
    assert voxels.leaf_projection(0.0, "erectophile")[0] < 0.45
    np.testing.assert_allclose(voxels.leaf_projection(theta, "ellipsoidal", [1.0]), 0.5, atol=5e-3)
    assert voxels.laser_spec("vz-400") == (0.007, 0.00035)
    with pytest.raises(ValueError):
        voxels.laser_spec("nope")


def test_streaming_a_shots_file_matches_in_memory(slab, tmp_path):
    rng = np.random.default_rng(5)
    slab.echo_attrs["classification"][:] = np.where(
        slab.echo_range >= 10.0, 2, rng.choice([4, 6], slab.n_echoes))
    path = tmp_path / "slab.parquet"
    slab.save(path, double=True, origin_tolerance=0, row_group_size=4000)
    assert Shots.file_info(path)["n_groups"] == 8
    kw = dict(ground_class=2, leaf_classes=[4], wood_classes=[6], laser="VZ-400",
              attenuation=["fpl", "ppl"], inclination=True, occlusion=True, flat_top=True)
    mem = voxels.ray_voxelize(Shots.load(path), 0.5, BOUNDS, **kw)
    streamed = voxels.ray_voxelize(path, 0.5, BOUNDS, **kw)
    for name in ("num_hits", "num_hit_leaf", "num_beams", "num_beams_occluded", "predominant_tree"):
        np.testing.assert_array_equal(streamed[name], mem[name])
    for name in ("path_length", "free_path_length", "bs_entering", "attenuation_ppl", "lad_fpl"):
        np.testing.assert_allclose(streamed[name], mem[name], rtol=1e-5, atol=1e-6)
    # Without bounds the grid comes from the echo bounds in the file header.
    assert voxels.ray_voxelize(path, 0.5).shape == voxels.ray_voxelize(Shots.load(path), 0.5).shape
    with pytest.raises(ValueError):
        voxels.ray_voxelize(path, 0.5, ground=np.zeros(slab.n_echoes, bool))


def _pulses(origins, ranges_per_pulse, direction=(0.0, 0.0, -1.0)):
    count = np.array([len(r) for r in ranges_per_pulse], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64)
    echo_range = np.array([x for r in ranges_per_pulse for x in r], dtype=float)
    n = len(ranges_per_pulse)
    return Shots(np.asarray(origins, float).reshape(n, 3), np.tile(direction, (n, 1)), start, count, echo_range)


COLUMN = ((0, 0, 0), (1, 1, 4))


def test_first_weighting_traces_misses():
    # Pulses without an echo still sample every voxel they cross.
    shots = _pulses([[0.5, 0.5, 10.0]] * 10, [[]] * 10)
    for weighting in ("equal", "first"):
        g = voxels.ray_voxelize(shots, 1.0, COLUMN, weighting=weighting, attenuation="transmittance")
        assert np.all(g.num_beams[:, 0, 0] == 10), weighting
        np.testing.assert_allclose(g.num_beams_weighted[:, 0, 0], 10), weighting


def test_auto_bounds_keep_the_outermost_echo():
    # Echo extent an exact multiple of the voxel size on every axis.
    origins = [[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 1.0, 10.0], [1.0, 1.0, 10.0]]
    shots = _pulses(origins, [[7.0], [8.0], [9.0], [10.0]])
    g = voxels.ray_voxelize(shots, 0.5, attenuation="transmittance")
    assert g.num_hits.sum() == 4


def test_malformed_shots_raise_value_error():
    bad = Shots(np.array([[0.0, 0.0, 10.0]]), np.array([[0.0, 0.0, -1.0]]),
                np.array([0]), np.array([3]), np.array([7.0]))
    with pytest.raises(ValueError, match="echo_range"):
        voxels.ray_voxelize(bad, 1.0, COLUMN, attenuation="transmittance")
