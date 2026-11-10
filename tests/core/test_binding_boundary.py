# What the compiled layer does with arrays the Python wrappers would not
# normally hand it: Fortran-ordered coordinates, per-point arrays of the
# wrong length, sizes of zero. Each is a ValueError or the right answer,
# never a panic and never scrambled rows.
import numpy as np
import pytest

from sylva import PointCloud, _core, io
from sylva.geo import masks


@pytest.fixture()
def xyz():
    return np.random.default_rng(7).uniform(0.0, 5.0, (500, 3))


@pytest.fixture()
def tree():
    return [dict(tree_id=1, x=2.5, y=2.5, dbh=0.3, height=5.0)]


# ----------------------------------------------------------- Fortran order


def test_fortran_ordered_coordinates_are_read_row_by_row(xyz, tmp_path):
    f = np.asfortranarray(xyz)
    assert f.flags.f_contiguous and not f.flags.c_contiguous

    # interpolate_py.rs
    np.testing.assert_array_equal(_core.interp_nearest_indices(f, xyz, None), np.arange(len(xyz)))
    # change_points_py.rs
    np.testing.assert_allclose(_core.change_c2c(xyz, f, None), 0.0, atol=1e-12)
    # lib.rs, xyz_ref_from_py: the batched LAS writer
    path = tmp_path / "fortran.las"
    _core.write(str(path), f, None)
    np.testing.assert_allclose(_core.read(str(path))[0], xyz, atol=2e-3)
    # lib.rs, rows_from_py: (n, 2) positions on a ground grid
    elevation = np.add.outer(np.arange(6.0), 10.0 * np.arange(6.0))
    xy = xyz[:, :2].copy()
    np.testing.assert_array_equal(
        _core.coreg_ground_height(elevation, 0.0, 0.0, 1.0, np.asfortranarray(xy)),
        _core.coreg_ground_height(elevation, 0.0, 0.0, 1.0, xy),
    )


def test_a_cloud_given_fortran_coordinates_after_construction_still_writes_them(xyz, tmp_path):
    cloud = PointCloud(xyz.copy())
    cloud.xyz = np.asfortranarray(xyz)
    path = tmp_path / "assigned.las"
    io.write(cloud, path)
    np.testing.assert_allclose(io.read(path).xyz, xyz, atol=2e-3)


def test_masks_refuse_fortran_ordered_coordinates(xyz):
    # masks_py.rs borrows without a copy, so it says what it needs.
    with pytest.raises(ValueError, match="C-contiguous"):
        _core.mask_near(np.asfortranarray(xyz), xyz, 0.01, False)
    near = masks.near(PointCloud(xyz), PointCloud(xyz[:10]), 0.01)
    assert near[:10].all()


# ----------------------------------------------------------------- lengths


@pytest.mark.parametrize(
    "call",
    [
        lambda x, h, lab, t: _core.make_chm(x, h[:100]),
        lambda x, h, lab, t: _core.stem_root_support(x, h[:100], t, np.array([0.5, 1.0])),
        lambda x, h, lab, t: _core.dbh_profile(x, h[:100], 2.5, 2.5, np.array([1.3])),
        lambda x, h, lab, t: _core.segment_trees(x, h[:100], t),
        lambda x, h, lab, t: _core.merge_branches(x, h[:100], t),
        lambda x, h, lab, t: _core.tree_heights(h, lab[:100], t),
        lambda x, h, lab, t: _core.tree_heights(h[:100], lab, t),
        lambda x, h, lab, t: _core.crown_metrics(x, h[:100], lab, 0),
        lambda x, h, lab, t: _core.crown_metrics(x, h, lab[:100], 0),
        lambda x, h, lab, t: _core.crown_metrics_all(x, h[:100], lab),
        lambda x, h, lab, t: _core.crown_metrics_all(x, h, lab[:100]),
    ],
    ids=["make_chm", "stem_root_support", "dbh_profile", "segment_trees", "merge_branches", "tree_heights-labels",
         "tree_heights-heights", "crown_metrics-heights", "crown_metrics-labels", "crown_metrics_all-heights",
         "crown_metrics_all-labels"],
)
def test_a_per_point_array_of_the_wrong_length_is_a_value_error(call, xyz, tree):
    heights = xyz[:, 2].copy()
    labels = np.zeros(len(xyz), dtype=np.int64)
    with pytest.raises(ValueError, match="one value per point"):
        call(xyz, heights, labels, tree)


def test_matching_lengths_still_run(xyz, tree):
    heights = xyz[:, 2].copy()
    labels = np.zeros(len(xyz), dtype=np.int64)
    assert len(_core.tree_heights(heights, labels, tree)) == 1
    assert _core.dbh_profile(xyz, heights, 2.5, 2.5, np.array([1.3])).shape == (1,)
    assert set(_core.crown_metrics_all(xyz, heights, labels)) == {0}


# ------------------------------------------------------------------- sizes


@pytest.mark.parametrize("bad", [0.0, -0.25, float("nan"), float("inf")])
def test_degenerate_sizes_are_value_errors(bad, xyz):
    heights = xyz[:, 2].copy()
    with pytest.raises(ValueError, match="slice_step"):
        _core.detect_stems(xyz, heights, slice_step=bad)
    with pytest.raises(ValueError, match="voxel_size"):
        _core.voxelize(xyz, bad)
    with pytest.raises(ValueError):
        _core.make_chm(xyz, heights, bad)


def test_a_voxel_far_smaller_than_the_cloud_is_refused_not_allocated(xyz):
    with pytest.raises(ValueError, match="larger voxel"):
        _core.voxelize(xyz, 1e-9)
    assert _core.voxelize(xyz, 0.5)["counts"].sum() == len(xyz)


# ------------------------------------------------------------------- shots


def test_shots_come_back_whole_and_owned(xyz, tmp_path):
    amplitude = np.arange(len(xyz), dtype=np.float32)
    shots = _core.shots_from_pointcloud(xyz, {"amplitude": amplitude}, (0.0, 0.0, 10.0))
    n = len(shots["origin"])
    assert n == len(xyz) and shots["origin"].shape == (n, 3) and shots["direction"].shape == (n, 3)
    assert shots["echo_start"].dtype == np.int64 and shots["echo_count"].dtype == np.int64
    assert shots["echo_attrs"]["amplitude"].dtype == np.float32
    for a in (shots["origin"], shots["direction"], shots["echo_range"], shots["echo_attrs"]["amplitude"]):
        assert a.flags.writeable

    same = _core.shots_transform(shots, np.eye(4))
    for k in ("echo_start", "echo_count"):
        np.testing.assert_array_equal(same[k], shots[k])
    for k in ("origin", "direction", "echo_range"):
        np.testing.assert_allclose(same[k], shots[k], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(same["echo_attrs"]["amplitude"], shots["echo_attrs"]["amplitude"])

    keep = np.zeros(n, dtype=bool)
    keep[::2] = True
    half = _core.shots_subset(shots, keep)
    assert len(half["origin"]) == keep.sum()
    np.testing.assert_array_equal(half["direction"], shots["direction"][keep])

    both = _core.shots_concatenate([half, half])
    assert len(both["echo_range"]) == 2 * len(half["echo_range"])
    assert len(_core.shots_shot_of_echo(both)) == len(both["echo_range"])
    assert _core.shots_echo_xyz(shots).shape == (len(shots["echo_range"]), 3)
    zenith, azimuth = _core.shots_zenith_azimuth(shots["direction"])
    assert zenith.shape == azimuth.shape == (n,)

    back, attrs = _core.shots_to_pointcloud(shots)
    assert back.shape == xyz.shape
    np.testing.assert_allclose(np.sort(back[:, 2]), np.sort(xyz[:, 2]), atol=1e-9)
    assert "amplitude" in attrs

    path = tmp_path / "shots.parquet"
    _core.write_shots(shots, str(path), True)
    read = _core.read_shots(str(path))
    np.testing.assert_allclose(read["echo_range"], shots["echo_range"])
    np.testing.assert_array_equal(read["echo_count"], shots["echo_count"])
