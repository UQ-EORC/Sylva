"""sylva.change.occupancy on two simulated scans of one synthetic plot: a block of foliage
removed between the epochs must come out lost, and a crown hidden from the later scan by a
new obstacle must come out unobserved, not lost."""

import numpy as np
import pytest

from sylva import PointCloud, change, synthetic, voxels

BOX = np.array([[4.0, 3.0, 9.0], [7.0, 6.0, 12.0]])  # foliage removed from tree 1
WALL_Y, WALL_X, WALL_Z = 12.0, (6.0, 9.0), (4.0, 9.0)  # opaque screen added in epoch b
BOUNDS = ((-2.0, -2.0, -1.0), (22.0, 22.0, 19.0))
VOXEL = 0.5
ORIGIN_A = np.array([10.0, 10.0, 1.5])
ORIGIN_B = ORIGIN_A + [0.15, -0.1, 0.0]  # the later scan stands slightly elsewhere


def _occupied(xyz):
    shape = np.round((np.array(BOUNDS[1]) - BOUNDS[0]) / VOXEL).astype(int)
    ijk = np.floor((xyz - np.array(BOUNDS[0])) / VOXEL).astype(int)
    ok = np.all((ijk >= 0) & (ijk < shape), axis=1)
    m = np.zeros(shape[::-1], bool)
    m[ijk[ok, 2], ijk[ok, 1], ijk[ok, 0]] = True
    return m


@pytest.fixture(scope="module")
def epochs():
    scene = synthetic.forest(seed=1)
    cls = scene.attrs["classification"]
    removed = np.all((scene.xyz >= BOX[0]) & (scene.xyz <= BOX[1]), axis=1) & (cls == 4)
    kept = scene[~removed]
    gx, gz = np.meshgrid(np.arange(*WALL_X, 0.008), np.arange(*WALL_Z, 0.008))
    wall = np.column_stack([gx.ravel(), np.full(gx.size, WALL_Y), gz.ravel()])
    attrs = {k: np.concatenate([v, np.full(len(wall), 6 if k == "classification" else 0, v.dtype)])
             for k, v in kept.attrs.items()}
    later = PointCloud(np.vstack([kept.xyz, wall]), attrs)
    # One echo per pulse, so that nothing is seen through the screen.
    sa = synthetic.scan(scene, ORIGIN_A, 0.25, max_echoes=1)
    sb = synthetic.scan(later, ORIGIN_B, 0.25, max_echoes=1)
    ga = voxels.ray_voxelize(sa, VOXEL, BOUNDS, ground_class=2, occlusion=True)
    gb = voxels.ray_voxelize(sb, VOXEL, BOUNDS, ground_class=2, occlusion=True)
    veg = later.attrs["classification"] != 2
    emptied = _occupied(scene.xyz[removed]) & ~_occupied(later.xyz[veg])
    return ga, gb, emptied


def _shadow(grid, margin=0.4):
    """Voxels well inside the cone the screen hides from the later scanner."""
    X, Y, Z = grid.centers()
    dy = Y - ORIGIN_B[1]
    t = np.where(dy > 0, (WALL_Y - ORIGIN_B[1]) / np.where(dy > 0, dy, 1.0), np.nan)
    px = ORIGIN_B[0] + t * (X - ORIGIN_B[0])
    pz = ORIGIN_B[2] + t * (Z - ORIGIN_B[2])
    return ((Y > WALL_Y + 1.0) & (px > WALL_X[0] + margin) & (px < WALL_X[1] - margin)
            & (pz > WALL_Z[0] + margin) & (pz < WALL_Z[1] - margin))


def test_removed_foliage_is_lost(epochs):
    ga, gb, emptied = epochs
    occ = change.occupancy(ga, gb)
    filled_a = ga["num_hits"] > 0
    truth = emptied & filled_a
    assert truth.sum() > 50
    cls = occ.classes[truth]
    # Nothing that was removed is called stable or gained.
    assert np.isin(cls, [change.OCCUPANCY_CLASSES["lost"], change.OCCUPANCY_CLASSES["unobserved"]]).all()
    assert (cls == change.OCCUPANCY_CLASSES["lost"]).mean() > 0.7
    # Where the foliage was dense enough to be missed with probability < alpha, all is lost.
    p = np.minimum(ga["num_hits"] / np.maximum(ga["num_beams"], 1), 1.0)
    dense = truth & ((1 - p) ** gb["num_beams"] <= 0.05) & (gb["num_beams"] >= 10)
    assert dense.sum() > 50
    assert (occ.classes[dense] == change.OCCUPANCY_CLASSES["lost"]).all()
    # The layer means drop where the foliage went.
    layers = occ.layers
    in_box = (layers["z"] > BOX[0, 2] + 0.5) & (layers["z"] < BOX[1, 2] - 0.5)
    assert (layers["pad_change"][in_box] < 0).all()
    assert (layers["lost"][in_box] > 0).all()


def test_occluded_crown_is_unobserved_not_lost(epochs):
    ga, gb, emptied = epochs
    occ = change.occupancy(ga, gb)
    hidden = _shadow(ga) & (ga["num_hits"] > 0) & ~emptied
    assert hidden.sum() > 100
    assert not (occ.classes[hidden] == change.OCCUPANCY_CLASSES["lost"]).any()
    assert (occ.classes[hidden] == change.OCCUPANCY_CLASSES["unobserved"]).mean() > 0.95
    assert (gb["state"][hidden] <= voxels.STATES["occluded"]).mean() > 0.95
    assert np.isnan(occ.pad_change[hidden & (gb["num_beams"] == 0)]).all()
    # The screen itself is new.
    X, Y, Z = ga.centers()
    screen = (np.abs(Y - WALL_Y) < VOXEL) & (X > WALL_X[0] + 0.5) & (X < WALL_X[1] - 0.5) \
        & (Z > WALL_Z[0] + 0.5) & (Z < WALL_Z[1] - 0.5) & (gb["num_hits"] > 0)
    assert (occ.classes[screen] == change.OCCUPANCY_CLASSES["gained"]).mean() > 0.8


def test_unchanged_vegetation_is_stable(epochs):
    ga, gb, emptied = epochs
    occ = change.occupancy(ga, gb)
    rest = (ga["num_hits"] > 0) & ~emptied & ~_shadow(ga, margin=-1.0)
    X, Y, Z = ga.centers()
    rest &= ~(np.all([(X > BOX[0, 0] - 1) & (X < BOX[1, 0] + 1), (Y > BOX[0, 1] - 1) & (Y < BOX[1, 1] + 1),
                      (Z > BOX[0, 2] - 1) & (Z < BOX[1, 2] + 1)], axis=0))
    cls = occ.classes[rest]
    assert (cls == change.OCCUPANCY_CLASSES["lost"]).mean() < 0.01
    assert (cls == change.OCCUPANCY_CLASSES["stable_occupied"]).mean() > 0.9
    # Without the miss-probability test sparse, grazed voxels turn into false losses.
    loose = change.occupancy(ga, gb, alpha=1.0)
    assert (loose.classes[rest] == change.OCCUPANCY_CLASSES["lost"]).sum() > (cls == change.OCCUPANCY_CLASSES["lost"]).sum()


def test_result_container_and_criteria(epochs):
    ga, gb, _ = epochs
    occ = change.occupancy(ga, gb, min_pulses=10)
    assert occ.classes.shape == ga["num_hits"].shape and occ.classes.dtype == np.uint8
    counts = occ.counts()
    assert sum(counts.values()) == occ.classes.size
    for name, code in change.OCCUPANCY_CLASSES.items():
        assert occ.layers[name].sum() == counts[name]
        assert occ.volume(name) == pytest.approx(counts[name] * VOXEL ** 3)
    np.testing.assert_array_equal(occ.pad_a, ga["pad_fpl"])
    np.testing.assert_array_equal(occ.pad_b, gb["pad_fpl"])
    both = (ga["num_beams"] >= 10) & (gb["num_beams"] >= 10)
    np.testing.assert_array_equal(np.isfinite(occ.pad_change), both & np.isfinite(ga["pad_fpl"] - gb["pad_fpl"]))
    lost = occ.centers("lost")
    assert lost.shape == (counts["lost"], 3)
    assert np.all((lost >= BOUNDS[0]) & (lost <= BOUNDS[1]))
    assert "lost=" in repr(occ)
    # The same epoch against itself: nothing changes.
    same = change.occupancy(ga, ga)
    assert same.counts()["lost"] == same.counts()["gained"] == 0
    assert np.nanmax(np.abs(same.pad_change)) == 0.0
    # Stricter pulse and hit criteria only move voxels to unobserved.
    strict = change.occupancy(ga, gb, min_pulses=200, min_hits=3)
    moved = strict.classes != occ.classes
    assert (strict.classes[moved] == change.OCCUPANCY_CLASSES["unobserved"]).all()
    assert strict.counts()["unobserved"] > counts["unobserved"]


def test_rejects_bad_input(epochs):
    ga, gb, _ = epochs
    small = voxels.ray_voxelize(synthetic.scan(synthetic.tree(seed=2), (3.0, 0.0, 1.5), 1.0), VOXEL)
    with pytest.raises(ValueError, match="shape|origin"):
        change.occupancy(ga, small)
    with pytest.raises(ValueError, match="RayVoxelGrid"):
        change.occupancy(ga, np.zeros(3))
    for kw, msg in [(dict(min_pulses=0), "min_pulses"), (dict(min_hits=1.5), "min_hits"),
                    (dict(alpha=0), "alpha"), (dict(alpha=2.0), "alpha")]:
        with pytest.raises(ValueError, match=msg):
            change.occupancy(ga, gb, **kw)
    with pytest.raises(ValueError):
        change.occupancy(ga, gb, pad="pad_nonsense")
    with pytest.raises(ValueError, match="unknown class"):
        change.occupancy(ga, gb).mask("vanished")
