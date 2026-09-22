import numpy as np
import pytest

from sylva import PointCloud, ground, trees


@pytest.fixture(scope="module")
def normalized(forest):
    c = ground.classify_ground_csf(forest)
    dtm = ground.make_dtm(c, resolution=0.5)
    return ground.normalize_height(c, dtm)


def test_circle_fits(rng):
    theta = rng.uniform(0, np.pi, 200)  # half circle (occluded stem)
    xy = np.column_stack([3 + 0.2 * np.cos(theta), -1 + 0.2 * np.sin(theta)])
    xy += rng.normal(0, 0.002, xy.shape)
    cx, cy, r, rmse = trees.fit_circle(xy)
    assert (cx, cy, r) == pytest.approx((3, -1, 0.2), abs=0.005)
    noisy = np.vstack([xy, rng.uniform(2.5, 3.5, (60, 2))])
    cx, cy, r, inl = trees.fit_circle_ransac(noisy, threshold=0.01)
    assert (cx, cy, r) == pytest.approx((3, -1, 0.2), abs=0.01)
    assert inl[:200].mean() > 0.9


def test_detect_stems(normalized, tree_specs):
    found = trees.detect_stems(normalized)
    assert len(found) == len(tree_specs)
    for t in found:
        spec = min(tree_specs, key=lambda s: np.hypot(s[0] - t.x, s[1] - t.y))
        assert np.hypot(spec[0] - t.x, spec[1] - t.y) < 0.03
        assert t.dbh == pytest.approx(spec[2], abs=0.01)


def test_segment_and_height(normalized, tree_specs):
    found = trees.detect_stems(normalized)
    labels = trees.segment_trees(normalized, found)
    assert labels.shape == (len(normalized),)
    trees.tree_heights(normalized, labels, found)
    for t in found:
        spec = min(tree_specs, key=lambda s: np.hypot(s[0] - t.x, s[1] - t.y))
        assert t.height == pytest.approx(spec[3], abs=0.3)
        m = trees.crown_metrics(normalized, labels, t.tree_id)
        assert 0 < m["crown_area"] < 40


def test_prune_and_crowns(normalized, tree_specs):
    found = trees.detect_stems(normalized)
    # Inject a duplicate next to tree 1 and a short bogus candidate.
    fake_dup = trees.Tree(len(found) + 1, found[0].x + 0.2, found[0].y, 0.1)
    fake_short = trees.Tree(len(found) + 2, 2.0, 2.0, 0.08)
    cands = found + [fake_dup, fake_short]
    labels = trees.segment_trees(normalized, cands)
    trees.tree_heights(normalized, labels, cands)
    pruned, new_labels = trees.prune_trees(cands, labels, min_height=3.0, merge_radius=0.5)
    assert len(pruned) == len(tree_specs)
    # A weak, thinly supported candidate is dropped by the quality gate.
    weak = trees.Tree(99, 15.0, 15.0, 0.1, height=10.0, n_points=5, n_slices=3, quality=0.05)
    strong = trees.Tree(98, 16.0, 16.0, 0.1, height=10.0, n_points=5, n_slices=3, quality=0.5)
    lab2 = np.concatenate([labels, [99, 98]])
    kept, _ = trees.prune_trees(cands + [weak, strong], lab2, min_quality_short=0.15)
    assert {(t.x, t.y) for t in kept} >= {(16.0, 16.0)} and (15.0, 15.0) not in {(t.x, t.y) for t in kept}
    assert [t.tree_id for t in pruned] == list(range(1, len(pruned) + 1))
    assert set(np.unique(new_labels)) <= set(range(-1, len(pruned) + 1))
    assert sum(t.n_points for t in pruned) == int((new_labels >= 0).sum())
    allm = trees.crown_metrics_all(normalized, new_labels)
    assert set(allm) == {t.tree_id for t in pruned}
    one = trees.crown_metrics(normalized, new_labels, pruned[0].tree_id)
    assert allm[pruned[0].tree_id]["crown_area"] == pytest.approx(one["crown_area"])


def test_dbh_profile(normalized, tree_specs):
    x, y, dbh, h = tree_specs[0]
    prof = trees.dbh_profile(normalized, (x, y), heights=np.array([1.0, 2.0, 5.0]))
    assert np.allclose(prof[:, 1], dbh, atol=0.01)


def test_segment_leaves_low_vegetation(rng):
    # A litter / understorey layer reachable along the surface stays
    # unassigned beyond low_radius of the base; the stem base keeps its label.
    from conftest import make_crown, make_stem

    import sylva

    stem = make_stem(rng, 0, 0, 0.15, 8.0)
    crown = make_crown(rng, 0, 0, 7.0, 2.0)
    n = 6000
    rad, ang = 3.0 * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
    litter = np.column_stack([rad * np.cos(ang), rad * np.sin(ang), rng.uniform(0.27, 0.45, n)])
    xyz = np.vstack([stem, crown, litter])
    pc = sylva.PointCloud(xyz, {"height": xyz[:, 2].copy()})
    tree = [trees.Tree(1, 0.0, 0.0, 0.30)]
    is_litter = np.arange(len(xyz)) >= len(stem) + len(crown)
    far = is_litter & (np.hypot(xyz[:, 0], xyz[:, 1]) > 1.2)
    labels = trees.segment_trees(pc, tree)
    assert (labels[far] == -1).all()
    assert (labels[: len(stem)][stem[:, 2] > 0.5] == 1).mean() > 0.99
    # Without the rule (and the understorey sources) the layer is swallowed by the tree.
    loose = trees.segment_trees(pc, tree, low_height=0.0, understorey_height=0.0)
    assert (loose[far] == 1).mean() > 0.5


def test_understorey_keeps_its_own_points(rng):
    # A 2.5 m shrub 2 m from the stem, joined to it by a grass layer: without a
    # competing source it goes to the tree, with one it stays unassigned, and
    # the tree keeps its stem and crown.
    from conftest import make_crown, make_stem

    import sylva

    stem = make_stem(rng, 0, 0, 0.15, 8.0)
    crown = make_crown(rng, 0, 0, 7.0, 2.0)
    n = 6000
    rad, ang = 3.0 * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
    grass = np.column_stack([rad * np.cos(ang), rad * np.sin(ang), rng.uniform(0.27, 0.6, n)])
    shrub = make_crown(rng, 2.0, 0.0, 1.5, 0.9, n=3000)
    shrub = shrub[shrub[:, 2] > 0.3]
    xyz = np.vstack([stem, crown, grass, shrub])
    pc = sylva.PointCloud(xyz, {"height": xyz[:, 2].copy()})
    tree = [trees.Tree(1, 0.0, 0.0, 0.30)]
    is_shrub = np.arange(len(xyz)) >= len(xyz) - len(shrub)
    upper = is_shrub & (xyz[:, 2] > 1.0)
    on = trees.segment_trees(pc, tree)
    off = trees.segment_trees(pc, tree, understorey_height=0.0)
    assert (off[upper] == 1).mean() > 0.9
    assert (on[upper] == -1).mean() > 0.9
    tree_pts = np.arange(len(xyz)) < len(stem) + len(crown)
    assert (on[tree_pts & (xyz[:, 2] > 1.0)] == 1).mean() > 0.99


def _base(rng, flanges: bool, clutter: bool):
    """A 0.25 m stem to 6 m, optionally with five flanges fading out by 2 m and a grass clump."""
    t = rng.uniform(0, 2 * np.pi, 400_000)
    h = rng.uniform(0, 6, len(t))
    r = 0.25 * (1 + (3 * np.clip(1 - h / 2, 0, None) * np.cos(2.5 * t) ** 8 if flanges else 0))
    pts = np.column_stack([r * np.cos(t), r * np.sin(t), h])
    pts[:, :2] += rng.normal(0, 0.003, (len(t), 2))
    if clutter:
        g = rng.normal(0, 1, (60_000, 3)) * [0.35, 0.35, 0.25] + [0.7, 0.2, 0.5]
        pts = np.vstack([pts, g[g[:, 2] > 0.05]])
    return PointCloud(pts, {"height": pts[:, 2].copy()})


def test_detect_buttress(rng):
    flanged = trees.detect_buttress(_base(rng, True, False), base_xy=(0, 0))
    assert flanged["buttressed"] and flanged["ridges"] >= 2
    assert 1.0 <= flanged["top"] <= 2.6, flanged["top"]
    round_ = trees.detect_buttress(_base(rng, False, True), base_xy=(0, 0))
    assert not round_["buttressed"], round_
    assert abs(round_["stem_radius"] - 0.25) < 0.03
