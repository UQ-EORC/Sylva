# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Realistic synthetic trees, plots and the beam scanner, checked against
their own truth and analytic cases."""

import numpy as np
import pytest
from scipy import integrate, stats

from sylva import PointCloud, synthetic, trees
from sylva.leaves import LeafAngleDistribution


@pytest.fixture(scope="module")
def broadleaf():
    return synthetic.tree_model("broadleaf", 1.0, 2.0, 0.5, dbh=0.35, height=16.0, point_density=400, seed=3)


# ----------------------------------------------------------------- trees


@pytest.mark.parametrize("arch", ["broadleaf", "conifer", "eucalypt", "savanna"])
def test_archetypes_honour_dbh_and_height(arch):
    t = synthetic.tree_model(arch, dbh=0.4, height=20.0, point_density=200, seed=1)
    assert t.dbh == pytest.approx(0.4, abs=1e-9)
    assert 0.8 * 20 < t.height < 1.2 * 20
    assert t.qsm.column("branch_order").max() == 3
    assert 0 < t.crown_base < t.height and t.crown_area > 0
    lab = t.points.attrs["label"]
    assert set(np.unique(lab)) == {2, 3, 4}
    # Stem points are the order-0 wood; leaves are class 4.
    wood = t.points.attrs["classification"] == 5
    assert np.array_equal(lab[wood] == 2, t.points.attrs["branch_order"][wood] == 0)
    assert np.array_equal(lab == 4, ~wood)


def test_truth_tables_are_consistent(broadleaf):
    t = broadleaf
    assert t.wood_volume == pytest.approx(np.sum(np.pi * t.qsm.column("radius") ** 2 * t.qsm.column("length")))
    assert sum(t.volume_by_order().values()) == pytest.approx(t.wood_volume)
    assert sum(t.length_by_order().values()) == pytest.approx(t.qsm.column("length").sum())
    lv = t.leaves
    assert t.leaf_area == pytest.approx(np.sum(np.pi / 4 * lv["length"] * lv["width"]))
    assert t.leaf_area == pytest.approx(lv["area"].sum())
    # Every wood point is counted on its cylinder.
    cyl = t.points.attrs["cylinder"]
    counts = np.bincount(cyl[cyl >= 0], minlength=len(t.qsm))
    assert np.array_equal(counts, t.qsm.column("n_points").astype(int))
    # Leaves hang from terminal twigs of order >= 1.
    assert (t.qsm.column("branch_order")[lv["cylinder"]] >= 1).all()


def test_points_lie_on_their_cylinders_and_blades(broadleaf):
    t = broadleaf
    c = t.qsm.cylinders
    cyl = t.points.attrs["cylinder"]
    br = (cyl >= 0) & (t.points.attrs["branch_order"] > 0)
    p = t.points.xyz[br]
    row = c[cyl[br]]
    v = p - row[:, 0:3]
    along = np.einsum("ij,ij->i", v, row[:, 3:6])
    perp = np.linalg.norm(v - along[:, None] * row[:, 3:6], axis=1)
    np.testing.assert_allclose(perp, row[:, 7], rtol=1e-9, atol=1e-12)
    assert (along >= -1e-9).all() and (along <= row[:, 6] + 1e-9).all()
    lf = t.points.attrs["leaf"]
    m = lf >= 0
    q = t.points.xyz[m] - t.leaves["centre"][lf[m]]
    off_plane = np.einsum("ij,ij->i", q, t.leaves["normal"][lf[m]])
    assert np.abs(off_plane).max() < 1e-12
    a = np.einsum("ij,ij->i", q, t.leaves["axis"][lf[m]]) / (0.5 * t.leaves["length"][lf[m]])
    b2 = (np.linalg.norm(q, axis=1) ** 2 - (a * 0.5 * t.leaves["length"][lf[m]]) ** 2) / (0.5 * t.leaves["width"][lf[m]]) ** 2
    assert (a ** 2 + b2 <= 1 + 1e-9).all()


def test_bare_stem_is_an_analytic_cone():
    # Crown above the top, no lean, sweep or swell: a straight stem whose taper
    # (exponent 1 for the conifer) is a cone of radius dbh/2 at 1.3 m.
    t = synthetic.tree_model("conifer", dbh=0.4, height=10.0, crown_base=10.0, lean=0, sweep=0,
                             butt_swell=0, point_density=50, seed=1)
    assert (t.qsm.column("branch_order") == 0).all()
    big_s = t.qsm.column("length").sum()
    r0 = 0.2 * big_s / (big_s - 1.3)
    assert t.wood_volume == pytest.approx(np.pi * r0 ** 2 * big_s / 3, rel=0.01)
    np.testing.assert_allclose(t.qsm.axis, np.tile([0, 0, 1.0], (len(t.qsm), 1)), atol=1e-12)
    assert not t.leaves["area"].size


def test_stem_shape_options_keep_or_add_area():
    kw = dict(dbh=0.6, height=20.0, point_density=20, seed=2)
    round_ = synthetic.tree_model("broadleaf", **kw)
    shaped = synthetic.tree_model("broadleaf", ellipticity=0.15, bark_depth=0.01, **kw)
    # Ellipse and fissures are rescaled to the round section's area.
    assert shaped.dbh == pytest.approx(0.6, rel=1e-9)
    assert shaped.qsm.column("radius")[0] == pytest.approx(round_.qsm.column("radius")[0], rel=1e-9)
    butt = synthetic.tree_model("broadleaf", buttresses=4, buttress_height=1.0, buttress_extent=0.8, **kw)
    assert butt.dbh == pytest.approx(0.6, rel=1e-9)  # flanges end below breast height
    assert butt.qsm.column("radius")[0] > 1.1 * round_.qsm.column("radius")[0]


def test_pipe_model_conserves_area():
    t = synthetic.tree_model("savanna", dbh=0.3, height=8.0, point_density=10, seed=4, min_radius=1e-5)
    c = t.qsm.cylinders
    parent = c[:, 8].astype(int)
    r2 = c[:, 7] ** 2
    child = np.zeros(len(c))
    np.add.at(child, parent[parent >= 0], r2[parent >= 0])
    br = (c[:, 9] > 0) & (child > 0)
    assert (child[br] <= r2[br] * (1 + 1e-6)).all()  # up to the clamped radius of leafless tips
    # Murray-like exponent 3 gives thicker twigs relative to the stem.
    t3 = synthetic.tree_model("savanna", dbh=0.3, height=8.0, point_density=10, seed=4, min_radius=1e-5,
                              pipe_exponent=3.0)
    tw = lambda q: np.median(q.qsm.column("radius")[q.qsm.column("branch_order") == 3])  # noqa: E731
    assert tw(t3) > tw(t)


def _mean_inclination(name, params=()):
    if name == "beta":
        mu, nu = params
        return np.pi / 2 * nu / (mu + nu)
    pdf = {"spherical": np.sin, "planophile": lambda x: 2 / np.pi * (1 + np.cos(2 * x)),
           "erectophile": lambda x: 2 / np.pi * (1 - np.cos(2 * x))}[name]
    return integrate.quad(lambda x: x * pdf(x), 0, np.pi / 2)[0]


@pytest.mark.parametrize("lad", ["spherical", "planophile", "erectophile", ("beta", 2.0, 5.0)])
def test_leaf_angles_follow_the_requested_distribution(lad):
    t = synthetic.tree_model("broadleaf", dbh=0.3, height=15.0, lad=lad, point_density=1, seed=5)
    incl = np.arccos(np.clip(t.leaves["normal"][:, 2], -1, 1))
    assert (t.leaves["normal"][:, 2] >= 0).all()
    name, params = (lad, ()) if isinstance(lad, str) else (lad[0], lad[1:])
    assert len(incl) > 10000
    assert incl.mean() == pytest.approx(_mean_inclination(name, params), abs=np.radians(0.6))
    if name == "beta":
        ks = stats.kstest(incl / (np.pi / 2), stats.beta(params[1], params[0]).cdf)
        assert ks.pvalue > 1e-3
    else:
        # G function of the generated leaves against the textbook distribution's.
        z = np.radians([0, 30, 57.5, 80])
        np.testing.assert_allclose(t.leaf_angles().g(z), LeafAngleDistribution.from_type(name).g(z), atol=0.01)
    if name == "spherical":
        np.testing.assert_allclose(t.leaf_angles().g(np.radians([10, 45, 80])), 0.5, atol=0.01)
    # Azimuths are uniform.
    az = np.arctan2(t.leaves["normal"][:, 1], t.leaves["normal"][:, 0])
    assert stats.kstest(az, stats.uniform(-np.pi, 2 * np.pi).cdf).pvalue > 1e-3


def test_leaf_size_and_area_are_honoured():
    t = synthetic.tree_model("broadleaf", dbh=0.3, height=15.0, leaf_area=40.0, leaf_size=(0.08, 0.04),
                             point_density=1, seed=6)
    assert t.leaf_area == pytest.approx(40.0, rel=0.02)
    assert np.allclose(t.leaves["length"], 0.08) and np.allclose(t.leaves["width"], 0.04)


def test_epicormic_shoots_sit_on_the_bole():
    t = synthetic.tree_model("eucalypt", dbh=0.5, height=30.0, epicormic=5.0, point_density=100, seed=7)
    epi = t.leaves["epicormic"]
    assert epi.any() and t.epicormic_leaf_area == pytest.approx(t.leaves["area"][epi].sum())
    z = t.leaves["centre"][epi, 2] - t.base[2]
    assert z.max() < t.crown_base + 1.0 and z.min() > 0.3
    pe = t.points.attrs["epicormic"] == 1
    assert pe.any() and (t.points.xyz[pe, 2] < t.crown_base + 1.0).all()
    assert t.cylinder_epicormic.any()
    assert synthetic.tree_model("eucalypt", dbh=0.5, height=30.0, point_density=10, seed=7).epicormic_leaf_area == 0


def test_same_seed_same_tree():
    a = synthetic.tree_model("conifer", dbh=0.3, height=12.0, point_density=100, seed=9)
    b = synthetic.tree_model("conifer", dbh=0.3, height=12.0, point_density=100, seed=9)
    c = synthetic.tree_model("conifer", dbh=0.3, height=12.0, point_density=100, seed=10)
    assert np.array_equal(a.points.xyz, b.points.xyz) and np.array_equal(a.qsm.cylinders, b.qsm.cylinders)
    assert len(c.points) != len(a.points) or not np.array_equal(c.points.xyz, a.points.xyz)


@pytest.mark.parametrize("kw, msg", [
    (dict(archetype="palm"), "archetype"),
    (dict(lad="flat"), "leaf angle"),
    (dict(lad=("beta", -1, 2)), "positive"),
    (dict(dbh=-0.1), "dbh"),
    (dict(height=float("nan")), "height"),
    (dict(max_order=5), "max_order"),
    (dict(ellipticity=0.95), "ellipticity"),
    (dict(leaf_size=(0.1, 0)), "leaf_size"),
])
def test_tree_model_rejects_bad_input(kw, msg):
    with pytest.raises(ValueError, match=msg):
        synthetic.tree_model(**{"point_density": 10, **kw})


def test_archetype_table():
    assert set(synthetic.ARCHETYPES) >= {"broadleaf", "conifer", "eucalypt", "savanna"}
    a = synthetic.archetype("eucalypt")
    assert a["lad"] == "erectophile" and len(a["allometry"]) == 3
    with pytest.raises(ValueError):
        synthetic.archetype("palm")


def test_stems_are_found_with_their_dbh():
    # Sylva's own stem detection on a generated tree finds the stem and a DBH near the truth.
    t = synthetic.tree_model("eucalypt", dbh=0.45, height=25.0, point_density=1500, seed=8)
    cloud = t.points.with_attrs(height=t.points.xyz[:, 2] - t.base[2])
    found = trees.detect_stems(cloud)
    best = min(found, key=lambda s: np.hypot(s.x - t.stem_bh[0], s.y - t.stem_bh[1]))
    assert np.hypot(best.x - t.stem_bh[0], best.y - t.stem_bh[1]) < 0.05
    assert best.dbh == pytest.approx(t.dbh, rel=0.08)


# ----------------------------------------------------------------- plots


@pytest.fixture(scope="module")
def small_plot():
    return synthetic.plot(size=20.0, density=500, point_density=80, ground_density=30, max_order=2,
                          archetypes={"broadleaf": 1, "conifer": 1}, seed=2)


def test_plot_truth(small_plot):
    p = small_plot
    tr = p.trees
    assert len(tr["dbh"]) == 20
    assert p.stem_density == pytest.approx(500)
    assert (tr["dbh"] >= 0.07).all() and np.all(np.diff(tr["dbh"]) <= 0)
    assert set(np.unique(tr["archetype"])) <= {"broadleaf", "conifer"}
    lab = p.points.attrs["label"]
    assert set(np.unique(lab)) == {1, 2, 3, 4, 5, 6}
    # Bases do not overlap.
    d = np.hypot(tr["x"][:, None] - tr["x"], tr["y"][:, None] - tr["y"])
    np.fill_diagonal(d, np.inf)
    assert (d >= 0.75 * (tr["dbh"][:, None] + tr["dbh"]) + 0.2 - 1e-9).all()
    # The per-tree table agrees with the cylinders and leaves.
    for i in (1, 7, 20):
        q = p.qsm(i)
        assert q.total_volume == pytest.approx(tr["wood_volume"][i - 1])
        assert q.stem_volume == pytest.approx(tr["stem_volume"][i - 1])
        assert p.leaves["area"][p.leaves["tree"] == i].sum() == pytest.approx(tr["leaf_area"][i - 1])
    with pytest.raises(KeyError):
        p.qsm(99)
    # Ground points are on the terrain, nothing below it, bases on it.
    g = lab == 1
    np.testing.assert_allclose(p.points.xyz[g, 2], p.ground_height(p.points.xyz[g, 0], p.points.xyz[g, 1]), atol=1e-12)
    assert (p.points.xyz[:, 2] >= p.ground_height(p.points.xyz[:, 0], p.points.xyz[:, 1]) - 1e-9).all()
    np.testing.assert_allclose(tr["z"], p.ground_height(tr["x"], tr["y"]), atol=1e-12)
    tid = p.points.attrs["tree_id"]
    assert set(np.unique(tid)) == set(range(21))
    assert (tid[np.isin(lab, [1, 5, 6])] == 0).all()
    assert len(p.dead_wood["kind"]) == round(60 * 0.04) + round(40 * 0.04)


def test_plot_heights_follow_the_allometry():
    p = synthetic.plot(size=40.0, density=400, point_density=5, ground_density=1, max_order=1, shrubs=0,
                       grass_cover=0, logs=0, stumps=0, height_noise=0.0, seed=3)
    a, b, c = synthetic.archetype("broadleaf")["allometry"]
    expected = 1.3 + a * (1 - np.exp(-b * p.trees["dbh"] * 100)) ** c
    assert np.median(p.trees["height"] / expected) == pytest.approx(1.0, abs=0.12)


def test_diameter_distributions():
    kw = dict(size=100.0, point_density=1, ground_density=0.01, max_order=1, shrubs=0, grass_cover=0,
              logs=0, stumps=0, seed=4)
    shape, scale, dmin = 2.2, 0.25, 0.07
    p = synthetic.plot(density=300, dbh=("weibull", shape, scale), min_dbh=dmin, **kw)
    d = p.trees["dbh"]
    assert len(d) == 300
    w = stats.weibull_min(shape, scale=scale)
    trunc = lambda x: (w.cdf(x) - w.cdf(dmin)) / w.sf(dmin)  # noqa: E731
    assert stats.kstest(d, trunc).pvalue > 1e-3
    p = synthetic.plot(density=300, dbh=("reverse_j", 0.08), min_dbh=0.05, **kw)
    ex = p.trees["dbh"] - 0.05
    assert stats.kstest(ex, stats.expon(scale=0.08).cdf).pvalue > 1e-3
    p = synthetic.plot(dbh=[0.5, 0.3, 0.2], **kw)
    np.testing.assert_allclose(p.trees["dbh"], [0.5, 0.3, 0.2])


def test_plot_terrain_roughness():
    p = synthetic.plot(size=30.0, density=0.0, dbh=("weibull", 2, 0.2), slope=0.2, aspect=0.0, roughness=0.1,
                       roughness_length=1.5, shrubs=0, grass_cover=0, logs=0, stumps=0, ground_density=50, seed=5)
    g = p.points.xyz
    detrended = g[:, 2] - 0.2 * g[:, 1]
    assert np.std(detrended) == pytest.approx(0.1, rel=0.25)
    # Fitted plane slope is the requested one.
    A = np.c_[g[:, 0], g[:, 1], np.ones(len(g))]
    coef = np.linalg.lstsq(A, g[:, 2], rcond=None)[0]
    assert coef[1] == pytest.approx(0.2, abs=0.02) and abs(coef[0]) < 0.02


def test_plot_is_deterministic_and_checked(small_plot):
    q = synthetic.plot(size=20.0, density=500, point_density=80, ground_density=30, max_order=2,
                       archetypes={"broadleaf": 1, "conifer": 1}, seed=2)
    assert np.array_equal(q.points.xyz, small_plot.points.xyz)
    with pytest.raises(ValueError, match="overlap"):
        synthetic.plot(size=5.0, density=200000, point_density=1)
    with pytest.raises(ValueError, match="grass_cover"):
        synthetic.plot(grass_cover=1.5, point_density=1)
    with pytest.raises(ValueError, match="diameter distribution"):
        synthetic.plot(dbh=("lognormal", 1, 2))
    with pytest.raises(ValueError, match="archetype"):
        synthetic.plot(archetypes={"palm": 1})


# ----------------------------------------------------------------- scanner


def _wall(y, x0, x1, z0, z1, step, cls=5):
    x, z = np.meshgrid(np.arange(x0, x1, step), np.arange(z0, z1, step))
    xyz = np.c_[x.ravel(), np.full(x.size, y), z.ravel()]
    return xyz, np.full(len(xyz), cls, dtype=np.uint8)


def _cloud(*walls):
    return PointCloud(np.vstack([w[0] for w in walls]), {"classification": np.concatenate([w[1] for w in walls])})


def _depth(s):
    """Distance along y of every first echo (the walls face the scanner at the origin)."""
    has = s.echo_count > 0
    return s.echo_range[s.echo_start[has]] * s.direction[has, 1]


def test_default_scan_is_unchanged():
    f = synthetic.forest(trees=[(5, 5, 0.3, 4.0)], size=10, ground_points=500)
    a = synthetic.scan(f, origin=(2, 2, 1.5), resolution_deg=1.0)
    assert "reflectance" not in a.echo_attrs
    b = synthetic.scan(f, origin=(2, 2, 1.5), resolution_deg=1.0, range_noise=0.0)
    assert "range_spread" in b.echo_attrs and b.n_shots == a.n_shots


def test_range_noise_statistics():
    cloud = _cloud(_wall(10.0, -3, 3, -2, 2, 0.01))
    s = synthetic.scan(cloud, origin=(0, 0, 0), resolution_deg=0.15, min_zenith_deg=80, max_zenith_deg=100,
                       range_noise=0.008, target_radius=0.008, seed=1)
    d = _depth(s) - 10.0
    assert len(d) > 5000
    assert d.std() == pytest.approx(0.008, rel=0.05) and abs(d.mean()) < 5e-4
    assert stats.normaltest(d).pvalue > 1e-4
    # Range-dependent noise a + b R: at 10 m and 20 m.
    far = _cloud(_wall(20.0, -6, 6, -4, 4, 0.02))
    for cl, r in ((cloud, 10.0), (far, 20.0)):
        s = synthetic.scan(cl, origin=(0, 0, 0), resolution_deg=0.15, min_zenith_deg=80, max_zenith_deg=100,
                           range_noise=(0.002, 0.0005), target_radius=0.016, seed=2)
        assert (_depth(s) - r).std() == pytest.approx(0.002 + 0.0005 * r, rel=0.08)
    # The same seed gives the same scan.
    a = synthetic.scan(cloud, origin=(0, 0, 0), resolution_deg=0.5, min_zenith_deg=80, max_zenith_deg=100, range_noise=0.01, seed=3)
    b = synthetic.scan(cloud, origin=(0, 0, 0), resolution_deg=0.5, min_zenith_deg=80, max_zenith_deg=100, range_noise=0.01, seed=3)
    assert np.array_equal(a.echo_range, b.echo_range)


def _edge_scene():
    # A foreground half-wall (x < 0) at 10 m in front of a background wall at 10.4 m.
    return _cloud(_wall(10.0, -3, 0.0, -1, 1, 0.004), _wall(10.4, -3, 3, -1, 1, 0.004, cls=2))


def test_mixed_pixel_rate_follows_the_footprint():
    cloud = _edge_scene()
    res, div = 0.1, 4.0
    kw = dict(origin=(0, 0, 0), resolution_deg=res, min_zenith_deg=87, max_zenith_deg=93, beam_divergence=div,
              exit_diameter=0.0, footprint_samples=37, target_radius=0.003, echo_separation=1.0)
    s = synthetic.scan(cloud, **kw)
    d = _depth(s)
    mixed = np.sum((d > 10.01) & (d < 10.39))
    # Footprint diameter at 10 m is 4 cm; pulses 1.75 cm apart along the edge.
    rows = round(6 / res)
    spacing = 10 * np.radians(res)
    expected = rows * (10 * div * 1e-3) / spacing
    assert mixed == pytest.approx(expected, rel=0.2)
    rs = s.echo_attrs["range_spread"][s.echo_start[s.echo_count > 0]]
    assert np.sum(rs > 0.3) == pytest.approx(mixed, abs=rows)
    # Without mixing, no echo falls in the gap; with a fine range resolution
    # the edge gives two echoes instead.
    s = synthetic.scan(cloud, mixed_pixels=False, **kw)
    d = _depth(s)
    assert np.sum((d > 10.01) & (d < 10.39)) == 0
    s = synthetic.scan(cloud, **{**kw, "echo_separation": 0.2})
    d = _depth(s)
    assert np.sum((d > 10.01) & (d < 10.39)) == 0
    assert np.sum(s.echo_count == 2) == pytest.approx(expected, rel=0.25)
    # A point beam (no divergence) never mixes.
    s = synthetic.scan(cloud, **{**kw, "beam_divergence": 0.0, "footprint_samples": None})
    d = _depth(s)
    assert np.sum((d > 10.01) & (d < 10.39)) == 0


def test_reflectance_and_detection():
    cloud = _cloud(_wall(10.0, -1, 1, -1, 1, 0.005))
    kw = dict(origin=(0, 0, 0), resolution_deg=0.5, min_zenith_deg=85, max_zenith_deg=95, target_radius=0.005)
    s = synthetic.scan(cloud, **kw)
    np.testing.assert_allclose(s.echo_attrs["reflectance"], 10 * np.log10(0.5), atol=1e-5)  # wood 0.5
    s = synthetic.scan(cloud, reflectance={5: 0.2}, **kw)
    np.testing.assert_allclose(s.echo_attrs["reflectance"], 10 * np.log10(0.2), atol=1e-5)
    s = synthetic.scan(cloud, reflectance=0.8, **kw)
    np.testing.assert_allclose(s.echo_attrs["reflectance"], 10 * np.log10(0.8), atol=1e-5)
    s = synthetic.scan(cloud.with_attrs(rho=np.full(len(cloud), 0.1)), reflectance="rho", **kw)
    np.testing.assert_allclose(s.echo_attrs["reflectance"], -10, atol=1e-5)
    assert synthetic.scan(cloud, reflectance=0.1, detection_threshold=0.2, **kw).n_echoes == 0


def test_scanner_presets_and_misses():
    cloud = _cloud(_wall(5.0, -1, 1, -1, 1, 0.01))
    s = synthetic.scan(cloud, origin=(0, 0, 0), resolution_deg=1.0, scanner="vz400")
    zen, _ = s.zenith_azimuth()
    assert zen.min() > 30 and zen.max() < 130  # degrees
    assert s.n_shots == 100 * 360 and s.n_echoes > 0 and (s.echo_count == 0).sum() > 0.9 * s.n_shots
    assert synthetic.scanner_preset("VZ-2000i")["beam_divergence"] == 0.27
    with pytest.raises(ValueError, match="scanner"):
        synthetic.scan(cloud, scanner="p50")
    with pytest.raises(ValueError, match="range_noise"):
        synthetic.scan(cloud, range_noise=-1)
    with pytest.raises(ValueError, match="footprint_samples"):
        synthetic.scan(cloud, beam_divergence=1.0, footprint_samples=0)


def test_scan_of_a_plot_keeps_labels(small_plot):
    s = synthetic.scan(small_plot.points, origin=(10, 10, 1.5), resolution_deg=0.5, scanner="vz400",
                       range_noise=0.005, seed=1)
    lab = s.echo_attrs["label"]
    assert {1, 2, 4} <= set(np.unique(lab))
    assert len(s.echo_range) == s.echo_count.sum() and (s.echo_count == 0).any()


def test_scan_of_empty_and_nan_clouds():
    empty = PointCloud(np.zeros((0, 3)))
    s = synthetic.scan(empty, origin=(0, 0, 0), resolution_deg=5.0, range_noise=0.01)
    assert s.n_shots == 26 * 72 and s.n_echoes == 0
    cloud = _cloud(_wall(5.0, -1, 1, -1, 1, 0.01))
    xyz = cloud.xyz.copy()
    xyz[::7] = np.nan
    s = synthetic.scan(PointCloud(xyz, cloud.attrs), origin=(0, 0, 0), resolution_deg=1.0, range_noise=0.001)
    assert s.n_echoes > 0 and np.isfinite(s.echo_range).all()
