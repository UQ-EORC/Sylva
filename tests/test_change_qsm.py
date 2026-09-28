"""sylva.change.qsm: comparing the QSMs of two epochs.

The synthetic scenario scans one tree in two epochs from three positions:
between them the stem thickens by 10 mm along its length, one limb is cut,
one grows 1 m longer and one is wrapped in foliage that hides it from every
later scan. The answers are known analytically."""

import csv

import numpy as np
import pytest

from sylva import PointCloud, Shots, change, qsm, synthetic, voxels
from sylva.change import QSMChange

# ------------------------------------------------------------------ models by hand


def _cylinder_model(r0=0.2, dr=0.0, limbs=(), stem_points=lambda z: 80):
    """A 10 m stem of 0.1 m cylinders (radius r0 - 0.01 z + dr) and horizontal
    limbs ``(height, azimuth deg, length)`` of 0.1 m cylinders."""
    rows = []
    for k in range(100):
        z = k * 0.1
        rows.append([0, 0, z, 0, 0, 1, 0.1, r0 - 0.01 * (z + 0.05) + dr, k - 1, 0, 0, stem_points(z)])
    for j, (h, az, length) in enumerate(limbs):
        c, s = np.cos(np.radians(az)), np.sin(np.radians(az))
        for k in range(round(length / 0.1)):
            parent = round(h / 0.1) - 1 if k == 0 else len(rows) - 1
            rows.append([c * 0.1 * k, s * 0.1 * k, h, c, s, 0, 0.1, 0.03 - 0.001 * k, parent, 1, j + 1, 60])
    return qsm.QSM(np.array(rows, dtype=float))


class _Grid:
    """Stands in for a RayVoxelGrid: one state everywhere."""

    def __init__(self, state, origin=(-2.0, -2.0, -1.0), shape=(120, 40, 40)):
        self.origin = np.array(origin)
        self.voxel_size = 0.1
        self._state = np.full(shape, state, dtype=np.uint8)

    def __getitem__(self, name):
        return self._state


def test_thickening_is_the_taper_increment():
    a = _cylinder_model(limbs=[(5.0, 0.0, 2.0)])
    b = _cylinder_model(dr=0.01, limbs=[(5.0, 0.0, 2.0)])
    c = change.compare_qsms(a, b)
    assert isinstance(c, QSMChange)
    assert c.taper["trusted"].all() and c.taper["fitted"].all()
    np.testing.assert_allclose(c.taper["increment"], 0.01, atol=1e-12)
    assert c.taper_increment == pytest.approx(0.01, abs=1e-12)
    assert c.n_taper_bins == 10
    # Every cylinder is measured, so the whole change is trusted.
    assert c.trusted_change == pytest.approx(c.change, abs=1e-12)
    assert c.untrusted_change == pytest.approx(0, abs=1e-12)
    assert c.orders["trusted_change"][0] == pytest.approx(b.stem_volume - a.stem_volume, abs=1e-12)
    assert c.dbh_b - c.dbh_a == pytest.approx(0.02, abs=1e-12)
    assert c.dbh_trusted and c.height_trusted
    s = c.summary()
    assert (s["n_matched"], s["n_lost"], s["n_new"]) == (2, 0, 0)
    assert s["change_m3"] == pytest.approx(b.total_volume - a.total_volume)


def test_prior_filled_parts_are_not_trusted():
    a = _cylinder_model(limbs=[(5.0, 0.0, 2.0)])
    # The later model fitted nothing between 4 and 6 m, nor on the limb.
    b = _cylinder_model(dr=0.01, limbs=[(5.0, 0.0, 2.0)], stem_points=lambda z: 0 if 4.0 <= z < 6.0 else 80)
    b.cylinders[b.column("branch_order") == 1, 11] = 0
    c = change.compare_qsms(a, b)
    z0 = c.taper["z0"]
    np.testing.assert_array_equal(c.taper["trusted"], (z0 < 4.0) | (z0 >= 6.0))
    np.testing.assert_array_equal(c.taper["measured_b"] == 0, (z0 >= 4.0) & (z0 < 6.0))
    gap = ~c.taper["trusted"]
    assert c.orders["untrusted_change"][0] == pytest.approx(
        (c.taper["volume_b"][gap] - c.taper["volume_a"][gap]).sum(), abs=1e-12)
    limb = c.matched["order_a"] == 1
    assert not c.matched["trusted"][limb].any()
    assert c.orders["trusted_change"][1] == 0
    assert c.change == pytest.approx(c.trusted_change + c.untrusted_change, abs=1e-15)


def test_branches_matched_lost_and_new():
    a = _cylinder_model(limbs=[(4.0, 0.0, 1.5), (6.0, 120.0, 1.5)])
    b = _cylinder_model(limbs=[(4.0, 0.0, 2.0), (8.0, 240.0, 1.0)])
    c = change.compare_qsms(a, b)
    limb = np.flatnonzero(c.matched["order_a"] == 1)
    assert len(limb) == 1
    k = limb[0]
    assert c.matched["length_b"][k] - c.matched["length_a"][k] == pytest.approx(0.5, abs=1e-9)
    assert c.matched["tip_shift"][k] == pytest.approx(0.5, abs=1e-9)
    assert c.matched["parent_consistent"][k] and c.matched["trusted"][k]
    lost, new = c.branches("lost"), c.branches("new")
    assert list(lost["id"]) == [2] and list(new["base_z"]) == [8.0]
    assert np.isnan(lost["observed_share"]).all()
    with pytest.raises(ValueError, match="unknown status"):
        c.branches("gone")


@pytest.mark.parametrize("state, status", [(2, "lost"), (1, "unobserved"), (0, "unobserved"), (3, "present")])
def test_grid_checks_lost_branches(state, status):
    a = _cylinder_model(limbs=[(6.0, 0.0, 1.5)])
    b = _cylinder_model()
    c = change.compare_qsms(a, b, grid_b=_Grid(state))
    assert list(c.lost["status"]) == [status]
    assert c.lost["trusted"][0] == (status == "lost")
    # The branch's volume is trusted change only when it is confirmed lost.
    expect = -c.lost["volume"][0] if status == "lost" else 0.0
    assert c.orders["trusted_change"][1] == pytest.approx(expect, abs=1e-15)
    # A new branch is checked against the earlier grid the same way.
    back = change.compare_qsms(b, a, grid_a=_Grid(state))
    assert list(back.new["status"]) == ["new" if status == "lost" else status]


def test_empty_models():
    e = qsm.QSM(np.zeros((0, 12)))
    c = change.compare_qsms(e, e)
    assert c.change == 0 and len(c.taper["z0"]) == 0 and np.isnan(c.taper_increment)
    assert np.isnan(c.height_a) and len(c.matched["id_a"]) == 0
    a = _cylinder_model(limbs=[(5.0, 0.0, 1.0)])
    gone = change.compare_qsms(a, e)
    assert gone.change == pytest.approx(-a.total_volume)
    assert set(gone.lost["status"]) == {"lost"}


def test_invalid_input():
    a = _cylinder_model()
    with pytest.raises(ValueError, match="cylinder array"):
        change.compare_qsms(np.zeros((3, 5)), a)
    bad = a.cylinders.copy()
    bad[3, 7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        change.compare_qsms(a, bad)
    with pytest.raises(ValueError, match="height_step"):
        change.compare_qsms(a, a, height_step=0)
    with pytest.raises(ValueError, match="min_measured"):
        change.compare_qsms(a, a, min_measured=2)
    with pytest.raises(ValueError, match="min_fits"):
        change.compare_qsms(a, a, min_fits=2.5)
    with pytest.raises(ValueError, match="RayVoxelGrid"):
        change.compare_qsms(a, a, grid_b=np.zeros((2, 2, 2)))
    # Plain cylinder arrays work as well as QSM objects.
    assert change.compare_qsms(a.cylinders, a.cylinders).change == 0


# ------------------------------------------------------------------ plot level


def test_plot_table(tmp_path):
    a = _cylinder_model(limbs=[(5.0, 0.0, 1.0)])
    b = _cylinder_model(dr=0.01, limbs=[(5.0, 0.0, 1.0)])
    qa = {1: a, 2: a, 3: a}
    qb = {11: b, 19: b}
    plot = change.compare_plot_qsms(qa, qb, pairs=[(1, 11), (3, 12)], deaths=[2], recruits=[19])
    assert len(plot) == 4
    assert [r["fate"] for r in plot.rows] == ["survivor", "survivor", "death", "recruit"]
    assert plot.rows[1]["note"] == "no model in epoch b" and np.isnan(plot.rows[1]["change_m3"])
    assert set(plot.changes) == {(1, 11)}
    single = change.compare_qsms(a, b)
    assert plot.rows[0]["change_m3"] == pytest.approx(single.change)
    assert plot.rows[0]["taper_increment_m"] == pytest.approx(0.01)
    t = plot.totals
    assert (t["n_survivors"], t["n_deaths"], t["n_recruits"], t["n_unmodelled"]) == (2, 1, 1, 1)
    assert t["mortality_m3"] == pytest.approx(a.total_volume)
    assert t["recruitment_m3"] == pytest.approx(b.total_volume)
    assert t["net_m3"] == pytest.approx(single.change - a.total_volume + b.total_volume)
    assert plot.rows[2]["tree_id_b"] is None and plot.rows[3]["tree_id_a"] is None

    path = tmp_path / "change.csv"
    plot.to_csv(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["fate"] for r in rows] == ["survivor", "survivor", "death", "recruit"]
    assert float(rows[0]["change_m3"]) == pytest.approx(single.change)
    assert rows[1]["change_m3"] == "" and rows[2]["tree_id_b"] == ""

    # A PlotQSMs is accepted as well as a dict.
    plots = qsm.PlotQSMs(models=qa, buttresses={}, skipped={})
    plots._points, plots._heights = {}, {}
    again = change.compare_plot_qsms(plots, qb, pairs=[(1, 11)])
    assert again.rows[0]["change_m3"] == pytest.approx(single.change)


def test_plot_invalid_input():
    a = _cylinder_model()
    with pytest.raises(ValueError, match="more than once"):
        change.compare_plot_qsms({1: a}, {2: a}, pairs=[(1, 2)], deaths=[1])
    with pytest.raises(ValueError, match="pair"):
        change.compare_plot_qsms({1: a}, {2: a}, pairs=[(1, 2, 3)])
    with pytest.raises(ValueError, match="unknown settings"):
        change.compare_plot_qsms({1: a}, {2: a}, pairs=[(1, 2)], height=1.0)
    with pytest.raises(ValueError, match="PlotQSMs or a mapping"):
        change.compare_plot_qsms([a], {2: a}, pairs=[(0, 2)])


# ------------------------------------------------------------------ synthetic scenario

RHO = 40000.0  # scene points per m2 of bark
HEIGHT = 10.0
DR = 0.01  # stem radius increment (m)
EXTENSION = 1.0  # limb extension (m)
#: name: (height, azimuth deg, zenith deg, length, base radius)
LIMBS = {
    "hidden": (4.0, 0.0, 60.0, 2.5, 0.05),
    "cut": (5.0, 120.0, 55.0, 2.5, 0.05),
    "grown": (6.0, 240.0, 55.0, 2.0, 0.045),
    "kept1": (7.0, 60.0, 50.0, 2.0, 0.04),
    "kept2": (8.0, 180.0, 50.0, 1.8, 0.035),
}
SCANNERS = [(7 * np.cos(np.radians(a)), 7 * np.sin(np.radians(a)), 1.5) for a in (30, 150, 270)]


def _stem_r(h, dr=0.0):
    return 0.16 - 0.011 * np.asarray(h) + dr


def _limb_r(rb, length):
    return lambda s: rb * np.maximum(1 - 0.5 * np.asarray(s) / length, 0.2)


def _limb_axis(az, zen):
    az, zen = np.radians(az), np.radians(zen)
    return np.array([np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)])


def _limb_start(h, zen, dr):
    """Where the limb axis leaves the stem surface, along the limb."""
    return (_stem_r(h, dr) + 0.005) / np.sin(np.radians(zen))


def _frame(axis):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    return a, u, np.cross(a, u)


def _tube(rng, start, axis, s0, s1, radius, noise=0.002):
    """Points on a tube of radius ``radius(s)`` from ``s0`` to ``s1`` along the axis."""
    a, u, v = _frame(axis)
    grid = np.linspace(s0, s1, 200)
    n = int(RHO * np.trapezoid(2 * np.pi * radius(grid), grid))
    s = rng.uniform(s0, s1, n)
    rs = radius(s)
    keep = rng.uniform(0, rs.max(), n) < rs  # uniform over the surface
    s, rs = s[keep], rs[keep]
    t = rng.uniform(0, 2 * np.pi, len(s))
    r = rs + rng.normal(0, noise, len(s))
    return np.asarray(start) + s[:, None] * a + (r * np.cos(t))[:, None] * u + (r * np.sin(t))[:, None] * v


def _tree(seed, dr=0.0, cut=(), extend=()):
    rng = np.random.default_rng(seed)
    parts = [_tube(rng, (0, 0, 0), (0, 0, 1), 0.0, HEIGHT, lambda h: _stem_r(h, dr))]
    for name, (h, az, zen, length, rb) in LIMBS.items():
        if name in cut:
            continue
        end = length + (EXTENSION if name in extend else 0.0)
        parts.append(_tube(rng, (0, 0, h), _limb_axis(az, zen), _limb_start(h, zen, dr), end, _limb_r(rb, length)))
    return np.vstack(parts)


def _foliage(name, dr, radius=0.3, seed=1):
    """A closed sleeve of leaves around a limb, hiding it from every direction."""
    rng = np.random.default_rng(seed)
    h, az, zen, length, _ = LIMBS[name]
    a, u, v = _frame(_limb_axis(az, zen))
    base = np.array([0.0, 0.0, h])
    s0 = (_stem_r(h, dr) + 0.02) / np.sin(np.radians(zen))
    pts = [_tube(rng, base, a, s0, length + 0.3, lambda s: np.full_like(np.asarray(s, float), radius), noise=0.0)]
    for s in (s0, length + 0.3):
        n = int(RHO * np.pi * radius**2)
        rr, t = radius * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
        pts.append(base + s * a + (rr * np.cos(t))[:, None] * u + (rr * np.sin(t))[:, None] * v)
    p = np.vstack(pts)
    return p[np.hypot(p[:, 0], p[:, 1]) > _stem_r(p[:, 2], dr) + 0.01]


def _scan(wood, leaves=None):
    """Three single-return scans (so the foliage is opaque), as one set of pulses."""
    leaves = np.zeros((0, 3)) if leaves is None else leaves
    cls = np.r_[np.full(len(wood), 5, np.uint8), np.full(len(leaves), 4, np.uint8)]
    cloud = PointCloud(np.vstack([wood, leaves]), {"classification": cls})
    return Shots.concatenate([synthetic.scan(cloud, origin=o, resolution_deg=0.12, max_zenith_deg=100.0,
                                             max_echoes=1) for o in SCANNERS])


def _model(shots):
    pc = shots.to_pointcloud()
    return qsm.build_qsm(PointCloud(pc.xyz[pc.attrs["classification"] == 5]), base_xy=(0, 0))


def _limb_volume(name, s0, s1):
    """True volume of a limb between two distances along it (m3)."""
    h, az, zen, length, rb = LIMBS[name]
    s = np.linspace(s0, s1, 2001)
    return np.trapezoid(np.pi * _limb_r(rb, length)(s) ** 2, s)


def _stem_change(z0, z1):
    """True stem volume added between two heights (m3)."""
    z = np.linspace(z0, z1, 2001)
    return np.trapezoid(np.pi * (_stem_r(z, DR) ** 2 - _stem_r(z) ** 2), z)


@pytest.fixture(scope="module")
def scenario():
    shots_a = _scan(_tree(0))
    shots_b = _scan(_tree(100, dr=DR, cut=("cut",), extend=("grown",)), _foliage("hidden", DR))
    grid_b = voxels.ray_voxelize(shots_b, 0.1, bounds=((-3.5, -3.5, -0.5), (3.5, 3.5, 11.0)), occlusion=True,
                                 tree_attr=None, average_leaf_area=0.0)
    qa, qb = _model(shots_a), _model(shots_b)
    return qa, qb, grid_b, change.compare_qsms(qa, qb, grid_b=grid_b)


def _matched_limb(c, qa, name):
    """Index in ``c.matched`` of the pair whose earlier branch is the named limb."""
    br = qa.branches()
    h, az, *_ = LIMBS[name]
    ids = br["id"][(np.abs(br["base_height"] - h) < 0.3) & (br["order"] == 1)]
    assert len(ids) == 1, name
    k = np.flatnonzero(c.matched["id_a"] == ids[0])
    return k[0] if len(k) else None


def test_scenario_taper_increment(scenario):
    _, _, _, c = scenario
    # The known 10 mm increment lies within two of the stated uncertainties.
    assert abs(c.taper_increment - DR) < 2 * c.taper_sigma
    assert c.taper_sigma < 0.002
    assert c.n_taper_bins >= 7
    assert c.dbh_b - c.dbh_a == pytest.approx(2 * DR, abs=0.004)


def test_scenario_branches(scenario):
    qa, _, _, c = scenario
    # The cut limb is lost (its space was seen empty); the hidden one is
    # unobserved, not lost; the grown limb is matched with its extension.
    for name, status in (("cut", "lost"), ("hidden", "unobserved")):
        h = LIMBS[name][0]
        k = np.flatnonzero((np.abs(c.lost["base_z"] - h) < 0.3) & (c.lost["order"] == 1))
        assert len(k) == 1, name
        assert c.lost["status"][k[0]] == status
        assert c.lost["trusted"][k[0]] == (status == "lost")
    cut = np.flatnonzero(c.lost["status"] == "lost")[0]
    h, az, zen, length, rb = LIMBS["cut"]
    assert c.lost["volume"][cut] == pytest.approx(_limb_volume("cut", _limb_start(h, zen, 0.0), length), rel=0.15)
    assert c.lost["observed_share"][cut] > 0.9
    hidden = np.flatnonzero(c.lost["status"] == "unobserved")[0]
    assert c.lost["observed_share"][hidden] < 0.5
    k = _matched_limb(c, qa, "grown")
    assert c.matched["length_b"][k] - c.matched["length_a"][k] == pytest.approx(EXTENSION, abs=0.15)
    assert c.matched["tip_shift"][k] == pytest.approx(EXTENSION, abs=0.15)
    for name in ("kept1", "kept2"):
        k = _matched_limb(c, qa, name)
        assert abs(c.matched["length_b"][k] - c.matched["length_a"][k]) < 0.15
    assert len(c.new["id"]) == 0
    # A limb the later scans could not see leaves the crown untrusted.
    assert not c.crown_trusted


def test_scenario_volume_totals(scenario):
    _, _, _, c = scenario
    t = c.taper
    # Stem: trusted change against the known increment over the same bins.
    truth_stem = sum(_stem_change(z0, min(z1, HEIGHT)) for z0, z1 in zip(t["z0"][t["trusted"]], t["z1"][t["trusted"]]))
    assert c.orders["trusted_change"][0] == pytest.approx(truth_stem, rel=0.08)
    # Order 1, trusted, against the cut limb and the extension; the hidden
    # limb's volume is left out of it, as untrusted change.
    h, az, zen, length, rb = LIMBS["cut"]
    grown = _limb_volume("grown", LIMBS["grown"][3], LIMBS["grown"][3] + EXTENSION)
    truth_branches = grown - _limb_volume("cut", _limb_start(h, zen, 0.0), length)
    assert c.orders["trusted_change"][1] == pytest.approx(truth_branches, rel=0.2)
    hidden = c.lost["status"] == "unobserved"
    assert c.orders["untrusted_change"][1] < 0 and not c.lost["trusted"][hidden].any()
    # Everything trusted against the known edits: stem increment, cut limb, extension.
    truth = truth_stem - _limb_volume("cut", _limb_start(h, zen, 0.0), length) + grown
    assert abs(c.trusted_change - truth) < 2 * c.trusted_sigma
    assert c.change == pytest.approx(c.trusted_change + c.untrusted_change, abs=1e-12)


def test_scenario_prior_filled_is_untrusted(scenario):
    qa, qb, grid_b, _ = scenario
    # Take the fits away from the later stem between 3 and 5 m and from the grown limb.
    cyl = qb.cylinders.copy()
    z = (cyl[:, 2] + qb.end[:, 2]) / 2 - cyl[:, 2][cyl[:, 8] < 0].min()
    cyl[(cyl[:, 9] == 0) & (z >= 3.0) & (z < 5.0), 11] = 0
    c0 = change.compare_qsms(qa, qb, grid_b=grid_b)
    k = _matched_limb(c0, qa, "grown")
    cyl[cyl[:, 10] == c0.matched["id_b"][k], 11] = 0
    c = change.compare_qsms(qa, qsm.QSM(cyl), grid_b=grid_b)
    gap = (c.taper["z0"] >= 3.0) & (c.taper["z1"] <= 5.0)
    assert not c.taper["trusted"][gap].any() and not c.taper["fitted"][gap].any()
    assert not c.matched["trusted"][k]
    assert c.untrusted_change > c0.untrusted_change
    assert c.n_taper_bins == c0.n_taper_bins - 2


def test_scenario_plot_wrapper(scenario):
    qa, qb, grid_b, c = scenario
    plot = change.compare_plot_qsms({7: qa}, {8: qb}, pairs=[(7, 8)], grid_b=grid_b)
    row = plot.rows[0]
    assert row["change_m3"] == pytest.approx(c.change)
    assert row["trusted_change_m3"] == pytest.approx(c.trusted_change)
    assert (row["n_lost"], row["n_unobserved"], row["n_new"]) == (1, 1, 0)
    assert plot.totals["growth_trusted_m3"] == pytest.approx(c.trusted_change)


def test_public_api_is_documented():
    import sylva.change.qsm as mod
    from test_docs import _public, test_public_api_is_documented as check

    names = [name for name, _ in _public(mod)]
    assert {"compare_qsms", "compare_plot_qsms", "QSMChange", "PlotQSMChange"} <= set(names)
    for _, obj in _public(mod):
        check(obj)
