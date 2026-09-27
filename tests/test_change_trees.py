"""sylva.change, part A (epochs, trees, plot summary, provenance) and
sylva.synthetic.forest_epochs, validated against the known truth of the
synthetic epochs."""

import importlib
import inspect
import warnings

import numpy as np
import pytest

from sylva import _core, change, ground, synthetic, trees
from sylva.change.epochs import _transform_xy

# ----------------------------------------------------------------- helpers


def inventory(cloud):
    """The operational sequence of the trees guide."""
    cloud = ground.normalize_height(cloud, ground.make_dtm(cloud))
    stems = trees.detect_stems(cloud)
    stems, _ = trees.merge_branches(cloud, stems)
    labels = trees.segment_trees(cloud, stems)
    trees.tree_heights(cloud, labels, stems)
    stems, labels = trees.prune_trees(stems, labels)
    return cloud, stems, labels


def true_ids(cloud, stems, labels):
    """The true tree of each detected tree: the most common ``tree_id`` of its points."""
    tid = np.asarray(cloud.attrs["tree_id"])
    out = []
    for t in stems:
        ids = tid[(labels == t.tree_id) & (tid > 0)]
        out.append(int(np.bincount(ids).argmax()) if len(ids) else -1)
    return np.array(out)


def truth_summary(ep, area, years=1.0):
    """plot_summary's quantities computed from the true tables, without noise."""
    t0, t1 = ep.trees
    growth = {c["tree_id"]: c for c in ep.of_kind("growth")}

    def row(t, i):
        k = np.flatnonzero(t["tree_id"] == i)[0]
        return t["dbh"][k], t["height"][k]

    surv = np.array([[*row(t0, i)[:1], 0, row(t0, i)[1], 0, g["d_dbh"], 0, g["d_height"], 0]
                     for i, g in growth.items()]).reshape(-1, 8)

    def singles(t, ids):
        return np.array([[row(t, i)[0], 0, row(t, i)[1], 0] for i in ids], dtype=float).reshape(-1, 4)

    dead = singles(t0, [c["tree_id"] for c in ep.of_kind("death")])
    rec = singles(t1, [c["tree_id"] for c in ep.of_kind("recruit")])
    rows = _core.change_plot_summary(surv, dead, rec, 0, float(area), float(years), 0, 0.95, 0.5, float("nan"), 0)
    return {name: est for name, _, est, _, _ in rows}


class Run:
    """One synthetic plot taken through the whole part-A workflow."""

    def __init__(self, seed=1, **kw):
        self.ep = synthetic.forest_epochs(seed=seed, **kw)
        (self.ca, self.sa, self.la), (self.cb, self.sb, self.lb) = [inventory(c) for c in self.ep.clouds]
        self.ids_a = true_ids(self.ca, self.sa, self.la)
        self.ids_b = true_ids(self.cb, self.sb, self.lb)
        self.al = change.align_epochs(*self.ep.clouds)
        self.match = change.match_trees(self.sa, self.sb, transform=self.al)
        self.inc = change.tree_increments(self.match, self.ca, self.cb, self.la, self.lb,
                                          noise_a=self.ep.range_noise[0], noise_b=self.ep.range_noise[1])
        self.size = kw.get("size", 30.0)
        self.summary = change.plot_summary(self.inc, area=self.size ** 2, years=1.0)

    def growth(self):
        """(true, measured) rows of every survivor pair."""
        g = {c["tree_id"]: c for c in self.ep.of_kind("growth")}
        return [(g[self.ids_a[i]], k) for k, (i, _) in enumerate(self.match.pairs)]


@pytest.fixture(scope="module")
def run():
    return Run(seed=1)


# ------------------------------------------------------------ forest_epochs


def small_epochs(**kw):
    args = dict(n_trees=5, size=16.0, deaths=1, recruits=1, replaced=1, small_increments=1,
                branch_removals=1, scan_positions=[(0.5, 0.5)], resolution_deg=1.0, ground_density=5.0)
    args.update(kw)
    return synthetic.forest_epochs(**args)


def test_forest_epochs_truth_is_consistent(run):
    ep = run.ep
    t0, t1 = ep.trees
    kinds = {k: ep.of_kind(k) for k in ("growth", "death", "recruit", "branch_removed", "foliage_thinned")}
    assert len(kinds["death"]) == 2 and len(kinds["recruit"]) == 2
    assert len(t0["tree_id"]) == 16 and len(t1["tree_id"]) == 16 - 2 + 2
    assert len(kinds["growth"]) == 14 and len(kinds["branch_removed"]) == 1
    # Growth is the difference of the two tables, and the same tree keeps its id.
    for g in kinds["growth"]:
        a, b = (np.flatnonzero(t["tree_id"] == g["tree_id"])[0] for t in (t0, t1))
        assert t1["dbh"][b] - t0["dbh"][a] == pytest.approx(g["d_dbh"], abs=1e-12)
        assert t1["height"][b] - t0["height"][a] == pytest.approx(g["d_height"], abs=1e-12)
        assert (t0["x"][a], t0["y"][a]) == (t1["x"][b], t1["y"][b])
    # Two survivors grow by only 0.5 mm.
    assert sum(g["d_dbh"] < 0.001 for g in kinds["growth"]) == 2
    # The felled tree's replacement stands 0.3 to 0.6 m from its stump.
    rep = [r for r in kinds["recruit"] if "replaces" in r]
    dead = next(d for d in kinds["death"] if d["tree_id"] == rep[0]["replaces"])
    assert 0.3 <= np.hypot(rep[0]["x"] - dead["x"], rep[0]["y"] - dead["y"]) <= 0.6
    # The box of thinned foliage lost leaf area, split over the trees in it.
    box = next(f for f in kinds["foliage_thinned"] if f["tree_id"] == -1)
    per_tree = sum(f["leaf_area"] for f in kinds["foliage_thinned"] if f["tree_id"] > 0)
    assert box["leaf_area"] == pytest.approx(per_tree) and box["leaf_area"] > 0
    # Epoch 2 is delivered displaced: the transform is the inverse of the default offset.
    assert not np.allclose(ep.transform, np.eye(4))
    assert np.allclose(ep.transform[:3, :3] @ ep.transform[:3, :3].T, np.eye(3))


def test_forest_epochs_clouds_and_shots(run):
    ep = run.ep
    for cloud, shots in zip(ep.clouds, ep.shots, strict=True):
        assert len(cloud) == shots.n_echoes > 100000
        for name in ("classification", "tree_id", "branch_id", "scan_id"):
            assert name in cloud.attrs
        assert set(np.unique(cloud.attrs["scan_id"])) == set(range(5))
        assert shots.n_shots > shots.n_echoes  # misses are kept
    # Dead trees have no points in epoch 2; recruits none in epoch 1.
    tid_a, tid_b = (np.asarray(c.attrs["tree_id"]) for c in ep.clouds)
    for d in ep.of_kind("death"):
        assert np.any(tid_a == d["tree_id"]) and not np.any(tid_b == d["tree_id"])
    for r in ep.of_kind("recruit"):
        assert np.any(tid_b == r["tree_id"]) and not np.any(tid_a == r["tree_id"])
    # The removed limb is gone from its tree in epoch 2.
    br = ep.of_kind("branch_removed")[0]
    on_tree = tid_b == br["tree_id"]
    assert np.any((tid_a == br["tree_id"]) & (ep.clouds[0].attrs["branch_id"] == br["branch_id"]))
    assert not np.any(on_tree & (ep.clouds[1].attrs["branch_id"] == br["branch_id"]))
    # Scanner positions, moved back into the reference frame, stand 1.5 m above the terrain.
    o = np.column_stack([ep.origins[1], np.ones(len(ep.origins[1]))]) @ ep.transform.T
    assert np.allclose(o[:, 2] - synthetic.terrain_height(o[:, 0], o[:, 1]), 1.5)


def test_forest_epochs_is_deterministic_and_checks_input():
    a, b = small_epochs(seed=3), small_epochs(seed=3)
    assert np.array_equal(a.clouds[1].xyz, b.clouds[1].xyz)
    assert a.changes == b.changes
    assert not np.array_equal(a.clouds[1].xyz, small_epochs(seed=4).clouds[1].xyz)
    with pytest.raises(ValueError, match="deaths"):
        small_epochs(deaths=9)
    with pytest.raises(ValueError, match="replaced"):
        small_epochs(replaced=2)
    with pytest.raises(ValueError, match="place"):
        small_epochs(n_trees=200, size=10.0)
    with pytest.raises(ValueError, match="offset"):
        small_epochs(offset=np.eye(3))
    with pytest.raises(ValueError, match=">= 0"):
        small_epochs(recruits=-1)
    # Without an offset and noise the epochs share a frame.
    f = small_epochs(offset=np.eye(4), range_noise=(0, 0))
    assert np.allclose(f.transform, np.eye(4))


# ------------------------------------------------------------ align_epochs


def test_alignment_recovers_the_offset_within_its_sigma(run):
    al, ep = run.al, run.ep
    err = al.transform @ np.linalg.inv(ep.transform)
    # Displacement of the error at the plot centre, per axis, in sigmas.
    c = np.append(al.centre, 1.0)
    z = ((err - np.eye(4)) @ c)[:3] / al.sigma_xyz
    assert np.all(np.abs(z) < 3), z
    # And at every stem.
    t0 = ep.trees[0]
    stems = np.column_stack([t0["x"], t0["y"], t0["z0"] + 1.3, np.ones(len(t0["x"]))])
    d = np.linalg.norm((stems @ (err - np.eye(4)).T)[:, :3], axis=1)
    assert np.sqrt(np.mean(d ** 2)) < 3 * al.registration_sigma
    assert 0 < al.registration_sigma < 0.005
    assert al.n_stems == 14  # the survivors; dead trees and recruits are not stable
    assert len(al.ground_residuals) > 500
    assert "registration sigma" in al.report()
    moved = al.apply(ep.clouds[1])
    assert np.allclose(moved.xyz[:3], ep.clouds[1].transform(al.transform).xyz[:3])


def test_alignment_modes_and_errors(run):
    ep = run.ep
    # Stems alone leave z and the tilts at the coarse transform; their sigma is unknown.
    st = change.align_epochs(run.al.reference, run.al.new, stable="stems", initial=run.al.transform)
    assert np.isnan(st.sigma_vertical) and np.isfinite(st.sigma_horizontal)
    assert st.coarse is None
    gr = change.align_epochs(run.al.reference, run.al.new, stable="ground", initial=run.al.transform)
    assert np.isnan(gr.sigma_horizontal) and np.isfinite(gr.sigma_vertical) and gr.n_stems == 0
    with pytest.raises(ValueError, match="stable"):
        change.align_epochs(*ep.clouds, stable="leaves")
    with pytest.raises(ValueError, match="initial"):
        change.align_epochs(run.al.reference, run.al.new, initial=np.eye(3))


# ------------------------------------------------------------- match_trees


def test_deaths_and_recruits_recovered_exactly(run):
    m, ep = run.match, run.ep
    assert len(run.sa) == len(ep.trees[0]["x"]) and len(run.sb) == len(ep.trees[1]["x"])
    dead = sorted(run.ids_a[m.status_a == "dead"])
    rec = sorted(run.ids_b[m.status_b == "recruit"])
    assert dead == sorted(c["tree_id"] for c in ep.of_kind("death"))
    assert rec == sorted(c["tree_id"] for c in ep.of_kind("recruit"))
    # Every survivor pair is one tree, the felled-and-replaced one included.
    assert all(run.ids_a[i] == run.ids_b[j] for i, j in m.pairs)
    assert m.counts() == {"survivors": 14, "deaths": 2, "recruits": 2, "merged": 0, "split": 0}
    assert len(m.survivors) == 14 and len(m.deaths) == 2 and len(m.recruits) == 2


def rows(*xyd):
    return [trees.Tree(k + 1, x, y, dbh=d) for k, (x, y, d) in enumerate(xyd)]


def test_trees_that_moved_slightly_still_match():
    # Truth tables of a plot whose survivors each moved 0.3 m.
    ep = small_epochs(n_trees=8, size=20.0, tree_shift=0.3, seed=2)
    m = change.match_trees(ep.trees[0], ep.trees[1])
    assert m.counts()["survivors"] == 7 and m.counts()["deaths"] == 1 and m.counts()["recruits"] == 1
    ids_a = np.asarray(ep.trees[0]["tree_id"])[m.pairs[:, 0]]
    ids_b = np.asarray(ep.trees[1]["tree_id"])[m.pairs[:, 1]]
    assert np.array_equal(ids_a, ids_b)
    assert np.allclose(m.distance, 0.3)
    # With a search radius below the shift they are deaths and recruits instead.
    tight = change.match_trees(ep.trees[0], ep.trees[1], max_distance=0.2)
    assert tight.counts()["survivors"] == 0


def test_assignment_beats_nearest_neighbour():
    # b1 is nearest to a0, but taking it would leave a1 without a partner.
    a = rows((0.0, 0.0, 0.3), (0.6, 0.0, 0.2))
    b = rows((0.35, 0.0, 0.2), (0.25, 0.0, 0.31))
    m = change.match_trees(a, b)
    assert m.pairs.tolist() == [[0, 1], [1, 0]]


def test_felled_tree_and_new_one_nearby():
    a = rows((0.0, 0.0, 0.40), (10.0, 0.0, 0.3))
    b = rows((0.4, 0.1, 0.09), (10.0, 0.0, 0.31))
    m = change.match_trees(a, b)
    assert m.pairs.tolist() == [[1, 1]]
    assert list(m.status_a) == ["dead", "survivor"] and list(m.status_b) == ["recruit", "survivor"]
    # Nearest-neighbour matching within 1 m would have called it one tree that shrank.
    loose = change.match_trees(a, b, max_shrink=1.0)
    assert [0, 0] in loose.pairs.tolist()


def test_two_trees_merging_and_splitting():
    a = rows((0.0, 0.0, 0.20), (0.4, 0.0, 0.18), (8.0, 0.0, 0.3))
    b = rows((0.2, 0.0, 0.30), (8.0, 0.0, 0.31))
    m = change.match_trees(a, b)
    assert sorted(m.status_a) == ["merged", "survivor", "survivor"]
    assert m.counts()["deaths"] == 0
    (ta, tb), = m.merged
    assert tb is b[0]
    # The pair the merged stem belongs to is flagged: its increment is not growth.
    amb = m.ambiguous_pairs()
    assert amb.tolist() == [bool(j == 0) for _, j in m.pairs]
    # The reverse: one stem found as two.
    s = change.match_trees(b, a)
    assert sorted(s.status_b) == ["split", "split", "survivor"] and s.counts()["recruits"] == 0
    assert list(s.status_a) == ["split", "survivor"] and s.counts()["deaths"] == 0
    assert len(s.split) == 2


def test_match_edge_cases():
    m = change.match_trees([], [])
    assert m.pairs.shape == (0, 2) and m.counts()["survivors"] == 0
    m = change.match_trees(rows((0, 0, 0.3)), [])
    assert list(m.status_a) == ["dead"]
    # NaN DBHs are matched on position alone.
    m = change.match_trees(rows((0, 0, np.nan)), rows((0.1, 0, 0.3)))
    assert m.pairs.tolist() == [[0, 0]]
    with pytest.raises(ValueError, match="finite"):
        change.match_trees(rows((np.nan, 0, 0.3)), rows((0, 0, 0.3)))
    with pytest.raises(ValueError, match="positive"):
        change.match_trees(rows((0, 0, 0.3)), rows((0, 0, 0.3)), max_distance=0)
    with pytest.raises(ValueError, match="transform"):
        change.match_trees(rows((0, 0, 0.3)), rows((0, 0, 0.3)), transform=np.eye(3))
    with pytest.raises(ValueError, match="x"):
        change.match_trees([object()], [])
    # A transform moves the second epoch's positions.
    shift = np.eye(4)
    shift[0, 3] = 5.0
    m = change.match_trees(rows((5, 0, 0.3)), rows((0, 0, 0.3)), transform=shift)
    assert m.pairs.tolist() == [[0, 0]] and m.distance[0] == pytest.approx(0)


# --------------------------------------------------------- tree_increments


def test_increments_within_their_mdi(run):
    inc = run.inc
    rows_ = run.growth()
    within = [abs(inc["d_dbh"][k] - g["d_dbh"]) <= inc["d_dbh_mdi"][k] for g, k in rows_]
    assert np.mean(within) >= 0.9
    assert np.all(inc["n_slices"] >= 6)
    assert np.median(inc["d_dbh_mdi"]) < 0.003
    # Height: within the MDI where the crown top was not changed on purpose.
    changed = {c["tree_id"] for c in run.ep.changes if c["kind"] in ("branch_removed", "foliage_thinned")}
    h = [abs(inc["d_height"][k] - g["d_height"]) <= inc["d_height_mdi"][k]
         for g, k in rows_ if g["tree_id"] not in changed]
    assert np.mean(h) >= 0.8
    # Single-epoch DBHs agree with the truth.
    t0 = run.ep.trees[0]
    for (i, _), k in zip(run.match.pairs, range(len(inc)), strict=True):
        true = t0["dbh"][t0["tree_id"] == run.ids_a[i]][0]
        assert abs(inc["dbh_a"][k] - true) < 4 * inc["dbh_a_se"][k]


def test_small_increments_are_below_detection(run):
    inc = run.inc
    for g, k in run.growth():
        if g["d_dbh"] < 0.001:
            assert inc["dbh_change"][k] == "below_detection"
            assert inc["d_height_mdi"][k] > abs(inc["d_height"][k]) or inc["height_change"][k] != "growth"
        else:
            assert inc["dbh_change"][k] == "growth"
    assert not inc["implausible"].any()


def test_increments_table(run):
    inc = run.inc
    assert len(inc) == 14
    for name in ("tree_id_a", "tree_id_b", "x", "y", "dbh_a", "dbh_b", "d_dbh", "d_dbh_mdi", "dbh_change",
                 "d_height", "height_change", "d_crown_area", "d_crown_volume", "flags", "ambiguous"):
        assert len(inc[name]) == 14, name
    assert inc.as_dict()["d_dbh"] is not inc.columns["d_dbh"]
    assert inc.measures[0]["diameter"].shape == (len(run.sa), len(inc.slice_heights))
    assert np.isfinite(inc["d_crown_area"]).all()
    pd = pytest.importorskip("pandas")
    assert isinstance(inc.to_pandas(), pd.DataFrame)


def test_unmeasured_and_implausible(run):
    # No labelled points in the second epoch: nothing measured, nothing claimed.
    none = np.full(len(run.cb), -1)
    inc = change.tree_increments(run.match, run.ca, run.cb, run.la, none, top_radius=0)
    assert set(inc["dbh_change"]) == {"unmeasured"}
    assert all("unmeasured" in f for f in inc["flags"])
    # A growth limit flags the fast growers.
    fast = change.tree_increments(run.match, run.ca, run.cb, run.la, run.lb, max_dbh_increment=0.01)
    assert fast["implausible"].sum() == np.sum(fast["d_dbh"] > 0.01) > 0
    with pytest.raises(ValueError, match="labels_b"):
        change.tree_increments(run.match, run.ca, run.cb, run.la, run.lb[:-1])
    with pytest.raises(ValueError, match="confidence"):
        change.tree_increments(run.match, run.ca, run.cb, run.la, run.lb, confidence=1.5)
    with pytest.raises(ValueError, match="noise_a"):
        change.tree_increments(run.match, run.ca, run.cb, run.la, run.lb, noise_a=-1)
    with pytest.raises(ValueError, match="TreeMatch"):
        change.tree_increments(None, run.ca, run.cb, run.la, run.lb)


def test_registration_uncertainty_widens_the_mdi(run):
    base = run.inc["d_dbh_mdi"]
    wide = change.tree_increments(run.match, run.ca, run.cb, run.la, run.lb, noise_a=0.003, noise_b=0.005,
                                  registration_sigma=0.2)["d_dbh_mdi"]
    assert np.all(wide > base)


# ------------------------------------------------------------ plot_summary


def test_plot_summary_covers_the_truth(run):
    s = run.summary
    truth = truth_summary(run.ep, run.size ** 2)
    for name in ("basal_area_growth", "basal_area_mortality", "basal_area_recruitment", "basal_area_net",
                 "volume_growth", "volume_net", "dbh_increment"):
        assert s[name].low <= truth[name] <= s[name].high, name
    assert (s["n_survivors"].estimate, s["n_deaths"].estimate, s["n_recruits"].estimate) == (14, 2, 2)
    assert s["mortality_rate"].estimate == pytest.approx(2 / 16)
    assert "biomass_net" not in s
    assert "basal_area_net" in s.report()
    bio = change.plot_summary(run.inc, area=900, wood_density=0.6)
    assert bio["biomass_net"].estimate == pytest.approx(0.6 * bio["volume_net"].estimate)
    with pytest.raises(ValueError, match="area"):
        change.plot_summary(run.inc, area=0)
    with pytest.raises(ValueError, match="TreeIncrements"):
        change.plot_summary(run.match, area=900)


def test_repeated_plots():
    """Across plots with different seeds the stated uncertainties hold."""
    names = ("basal_area_growth", "basal_area_mortality", "basal_area_net", "dbh_increment")
    cover = dict.fromkeys(names, 0)
    within, n, zs = 0, 0, []
    seeds = range(20, 26)
    for seed in seeds:
        r = Run(seed=seed, n_trees=10, size=24.0)
        # Matching is exact.
        assert sorted(r.ids_a[r.match.status_a == "dead"]) == sorted(c["tree_id"] for c in r.ep.of_kind("death"))
        assert sorted(r.ids_b[r.match.status_b == "recruit"]) == sorted(c["tree_id"] for c in r.ep.of_kind("recruit"))
        for g, k in r.growth():
            within += abs(r.inc["d_dbh"][k] - g["d_dbh"]) <= r.inc["d_dbh_mdi"][k]
            n += 1
        err = r.al.transform @ np.linalg.inv(r.ep.transform)
        zs.append(((err - np.eye(4)) @ np.append(r.al.centre, 1.0))[:3] / r.al.sigma_xyz)
        truth = truth_summary(r.ep, r.size ** 2)
        for name in names:
            cover[name] += r.summary[name].low <= truth[name] <= r.summary[name].high
    assert within / n >= 0.95
    assert np.all(np.abs(zs) < 3)
    assert all(c >= len(seeds) - 1 for c in cover.values()), cover


# --------------------------------------------------------------- provenance


def test_provenance_warns_when_epochs_differ(run):
    rec_a = change.provenance({"detect_stems": {"min_slices": 3}, "voxel": 0.01}).records[0]
    rec_b = change.provenance({"detect_stems": {"min_slices": 4}, "voxel": 0.01}).records[0]
    import sylva

    assert rec_a["sylva_version"] == sylva.__version__ and "created" in rec_a
    with pytest.warns(change.ProvenanceWarning, match="min_slices"):
        p = change.provenance(rec_a, rec_b)
    assert p.differences == [("settings.detect_stems.min_slices", "3", "4")]
    assert not p.consistent
    # Same settings, different dates: consistent and silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same = change.provenance(rec_a, change.provenance({"voxel": 0.01, "detect_stems": {"min_slices": 3}}).records[0])
    assert same.consistent
    # A different Sylva version is a difference too.
    old = dict(rec_b, sylva_version="0.0.1", settings=rec_a["settings"])
    with pytest.warns(change.ProvenanceWarning):
        assert change.provenance(rec_a, old).differences[0][0] == "sylva_version"
    # Results carry their settings, dataclasses included.
    with pytest.warns(change.ProvenanceWarning):
        p = change.provenance({"align": run.al.settings, "increments": run.inc.settings},
                              {"align": run.al.settings, "increments": dict(run.inc.settings, slice_thickness=0.2)})
    assert [d[0] for d in p.differences] == ["settings.increments.slice_thickness"]
    three = change.provenance({"a": 1}, {"a": 1}, {"a": 2}, warn=False)
    assert three.differences == [("epoch 2: settings.a", "1", "2")]
    assert '"sylva_version"' in three.to_json()
    with pytest.raises(ValueError, match="at least one"):
        change.provenance()


# ---------------------------------------------------------------- the docs


@pytest.mark.parametrize("module", ["sylva.change.epochs", "sylva.change.trees", "sylva.change.summary"])
def test_public_api_is_documented(module):
    mod = importlib.import_module(module)
    for name in mod.__all__:
        obj = getattr(mod, name)
        doc = inspect.getdoc(obj)
        assert doc, name
        if inspect.isfunction(obj) and len(inspect.signature(obj).parameters):
            assert "Parameters" in doc, name
    for name in change.__all__:
        assert hasattr(change, name)


def test_transform_xy_matches_points():
    t = np.eye(4)
    t[:2, 3] = [1.0, -2.0]
    assert np.allclose(_transform_xy(t, [[0.0, 0.0], [1.0, 1.0]]), [[1.0, -2.0], [2.0, -1.0]])
    assert _transform_xy(t, np.zeros((0, 2))).shape == (0, 2)
