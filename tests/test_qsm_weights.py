"""Wood-aware QSM fitting: per-point wood labels and wood weights."""

import warnings

import numpy as np
import pytest

from sylva import PointCloud, leaves, qsm, synthetic, trees


def _sheathed_stem(seed=5):
    """A 0.15 m stem, 12 m tall, sheathed in foliage 0.2-0.55 m out up to 7 m."""
    from conftest import make_stem

    rng = np.random.default_rng(seed)
    stem = make_stem(rng, 0.0, 0.0, 0.15, 12.0, density=3000)
    n = 60000
    r, th, z = rng.uniform(0.2, 0.55, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0.3, 7.0, n)
    pts = np.vstack([stem, np.column_stack([r * np.cos(th), r * np.sin(th), z])])
    is_wood = np.zeros(len(pts), bool)
    is_wood[: len(stem)] = True
    return PointCloud(pts, {"height": pts[:, 2].copy()}), is_wood, np.pi * 0.15**2 * 12.0


def test_unit_weights_give_the_unweighted_model(single_tree):
    plain = qsm.build_qsm(single_tree)
    ones = qsm.build_qsm(single_tree, weights=np.ones(len(single_tree)))
    np.testing.assert_array_equal(plain.cylinders, ones.cylinders)


def test_weights_see_through_foliage_around_the_bole():
    cloud, is_wood, true_volume = _sheathed_stem()
    free = qsm.build_qsm(cloud, base_xy=(0.0, 0.0))
    w = np.where(is_wood, 0.9, 0.1)
    weighted = qsm.build_qsm(cloud, base_xy=(0.0, 0.0), weights=w)
    assert free.total_volume > 1.3 * true_volume
    assert weighted.total_volume == pytest.approx(true_volume, rel=0.1)
    assert weighted.dbh == pytest.approx(0.30, abs=0.03)
    # n_points counts the confident inliers (weight >= 0.5): here bark only.
    n = weighted.column("n_points")
    assert n.max() > 0
    # A floor drops the foliage from the graph altogether.
    floored = qsm.build_qsm(cloud, base_xy=(0.0, 0.0), weights=w, min_weight=0.5)
    assert floored.total_volume == pytest.approx(true_volume, rel=0.1)


def test_foliage_alone_is_never_a_measurement():
    cloud, is_wood, _ = _sheathed_stem()
    leaf = cloud[~is_wood]
    model = qsm.build_qsm(leaf, weights=np.full(len(leaf), 0.1))
    assert (model.column("n_points") == 0).all()


@pytest.mark.parametrize("bad, match", [
    (np.ones(10), "one value per point"),
    ("nan", "finite and in"),
    (1.5, "finite and in"),
    (-0.1, "finite and in"),
])
def test_bad_weights_are_refused(single_tree, bad, match):
    if isinstance(bad, np.ndarray):
        w = bad
    else:
        w = np.ones(len(single_tree))
        w[7] = float(bad)
    with pytest.raises(ValueError, match=match):
        qsm.build_qsm(single_tree, weights=w)
    with pytest.raises(ValueError, match="numbers"):
        qsm.build_qsm(single_tree, weights=np.array(["a"] * len(single_tree)))
    with pytest.raises(ValueError, match="min_weight"):
        qsm.build_qsm(single_tree, weights=np.full(len(single_tree), 0.1), min_weight=0.5)


def test_build_plot_takes_wood_labels():
    cloud, is_wood, true_volume = _sheathed_stem()
    labels = np.ones(len(cloud), np.int64)
    stems = [trees.Tree(1, 0.0, 0.0, np.nan, height=12.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_mask = qsm.build_plot(cloud, labels, stems, wood=is_wood)
        bare = qsm.build_plot(cloud[is_wood], labels[is_wood], stems, wood=False)
        # The same labels as an attribute, as 1 / 0 / -1 codes or booleans.
        codes = np.where(is_wood, 1, -1).astype(np.int8)
        codes[::7] = np.where(is_wood[::7], 1, 0)
        by_attr = qsm.build_plot(cloud.with_attrs(wood=codes), labels, stems, wood="wood")
        by_bool = qsm.build_plot(cloud.with_attrs(w=is_wood), labels, stems, wood="w")
    assert by_mask.volume(1) == pytest.approx(true_volume, rel=0.1)
    assert by_mask.volume(1) == pytest.approx(bare.volume(1), rel=1e-9)
    assert by_attr.volume(1) == by_mask.volume(1) == by_bool.volume(1)


def test_build_plot_takes_wood_weights():
    cloud, is_wood, true_volume = _sheathed_stem()
    labels = np.ones(len(cloud), np.int64)
    stems = [trees.Tree(1, 0.0, 0.0, np.nan, height=12.0)]
    w = np.where(is_wood, 0.9, 0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = qsm.build_plot(cloud, labels, stems, wood=w)
        by_attr = qsm.build_plot(cloud.with_attrs(conf=w.astype(np.float32)), labels, stems, wood="conf")
        ones = qsm.build_plot(cloud, labels, stems, wood=np.ones(len(cloud)))
        plain = qsm.build_plot(cloud, labels, stems, wood=False)
    assert weighted.volume(1) == pytest.approx(true_volume, rel=0.1)
    assert by_attr.volume(1) == pytest.approx(weighted.volume(1), rel=0.05)
    np.testing.assert_array_equal(ones.models[1].cylinders, plain.models[1].cylinders)
    # The weighting settings pass through to each tree.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        floored = qsm.build_plot(cloud, labels, stems, wood=w, min_weight=0.5)
    assert floored.volume(1) == pytest.approx(true_volume, rel=0.1)


def test_build_plot_refuses_bad_wood():
    cloud, is_wood, _ = _sheathed_stem()
    labels = np.ones(len(cloud), np.int64)
    with pytest.raises(ValueError, match="one value per point"):
        qsm.build_plot(cloud, labels, wood=is_wood[1:])
    with pytest.raises(ValueError, match="one value per point"):
        qsm.build_plot(cloud, labels, wood=np.ones(5))
    w = np.ones(len(cloud))
    w[3] = np.nan
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        qsm.build_plot(cloud, labels, wood=w)
    w[3] = 2.0
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        qsm.build_plot(cloud, labels, wood=w)
    with pytest.raises(ValueError, match="no attribute"):
        qsm.build_plot(cloud, labels, wood="wood")
    with pytest.raises(ValueError, match="wood labels"):
        qsm.build_plot(cloud.with_attrs(wood=np.full(len(cloud), 5)), labels, wood="wood")
    with pytest.raises(TypeError, match="wood="):
        qsm.build_plot(cloud, labels, weights=np.ones(len(cloud)))


def test_wood_confidence_scores():
    tree = synthetic.tree(seed=3, leaf_points=4000)
    is_wood = tree.attrs["classification"] == 5
    for method in ("passage", "gbs"):
        mask, conf = leaves.classify_leaf_wood(tree, method=method, return_scores=True)
        np.testing.assert_array_equal(mask, leaves.classify_leaf_wood(tree, method=method))
        assert conf.shape == (len(tree),) and conf.dtype == float
        assert ((conf >= 0) & (conf <= 1)).all()
        # Wood is more confident than leaves, on average.
        assert conf[is_wood].mean() > conf[~is_wood].mean() + 0.2
        # The confidence is a valid weight.
        model = qsm.build_qsm(tree, weights=conf)
        assert len(model) > 0
    empty = PointCloud(np.zeros((0, 3)))
    mask, conf = leaves.classify_leaf_wood(empty, return_scores=True)
    assert len(mask) == len(conf) == 0
