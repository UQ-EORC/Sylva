# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Change between two airborne surveys (sylva.change.als)."""

import json

import numpy as np
import pytest

from sylva import PointCloud, Raster, als, change, cli, synthetic
from sylva.als_canopy import ALSProfile

OFFSET = (0.3, -0.2, 0.15)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """A 50 m forest with two buildings, flown twice: trees felled, a gap
    opened, the west half grown by 1 m, a sparser sensor, and a known
    misalignment."""
    d = tmp_path_factory.mktemp("als_change")
    ep = synthetic.als_epochs(n_trees=40, size=50.0, removed=2, gap=(35.0, 35.0, 3), offset=OFFSET,
                              seed=2)
    a, b = ep.write_tiles(d / "a", d / "b", size=35.0, epsg=32755)
    return ep, a, b, d


@pytest.fixture(scope="module")
def aligned(pair):
    ep, a, b, _ = pair
    return change.align_surveys(a, b, block_size=35.0, stable_classes=(2, 6), model="constant")


@pytest.fixture(scope="module")
def chm(pair, aligned):
    """At 2 m, a cell holds about 20 pulses of the sparser survey."""
    ep, a, b, _ = pair
    return change.chm_change(a, b, resolution=2.0, alignment=aligned)


def true_chm(scene, r):
    """Highest scene point above the terrain in each cell of ``r``'s grid."""
    keep = np.asarray(scene.attrs["classification"]) != 2
    p = scene.xyz[keep]
    h = p[:, 2] - synthetic.terrain_height(p[:, 0], p[:, 1])
    rows = np.floor((p[:, 1] - r.ymin) / r.resolution).astype(int)
    cols = np.floor((p[:, 0] - r.xmin) / r.resolution).astype(int)
    ok = (rows >= 0) & (cols >= 0) & (rows < r.data.shape[0]) & (cols < r.data.shape[1])
    out = np.zeros(r.data.shape)
    np.maximum.at(out, (rows[ok], cols[ok]), h[ok])
    return out


def zones(ep, r):
    """Cells well away from every changed tree, and those whose true canopy
    fell by more than 3 m."""
    t = ep.trees
    x, y = r.cell_centers()
    changed = np.zeros(x.shape, bool)
    for k in range(len(t["x"])):
        if t["fate"][k] != "unchanged":
            reach = 0.25 * (t["height_a"][k] + 1.0) + 1.5
            changed |= np.hypot(x - t["x"][k], y - t["y"][k]) <= reach
    inside = (x > 3) & (x < 67) & (y > 3) & (y < 67)
    drop = true_chm(ep.scenes[1], r) - true_chm(ep.scenes[0], r)
    return (~changed) & inside, (drop < -3.0) & inside


# ---------------------------------------------------------------- alignment

def test_alignment_recovers_a_known_offset(pair, aligned):
    ep, a, b, _ = pair
    o, s = aligned.offset, aligned.sigma
    assert np.all(np.abs(o - np.array(OFFSET)) < 4 * s + 0.01), (o, s)
    assert s[2] < 0.01 and s[0] < 0.1 and s[1] < 0.1
    assert aligned.birge[2] < 3
    t = aligned.table()
    assert len(t["x"]) >= 1 and {"dx", "sd_dz", "model_dz", "n_used"} <= set(t)
    assert "constant offset" in aligned.report()
    # Ground alone: the slope in x is uniform, so dx is left to the prior,
    # and the vertical offset absorbs the shift along it, within its sigma.
    g = change.align_surveys(a, b, block_size=35.0, model="field")
    assert g.sigma[0] > 0.5 * 0.3 and g.sigma[1] < 0.1
    assert abs(g.offset[1] - OFFSET[1]) < 4 * g.sigma[1] + 0.01
    assert abs(g.offset[2] - OFFSET[2]) < 4 * g.sigma[2] + 0.01
    assert g.values.shape[2] == 3 and g.residuals.shape == g.values.shape
    # The block estimates are the same whatever the chunks and workers.
    g2 = change.align_surveys(a, b, block_size=35.0, model="field", chunk_size=70.0, workers=3)
    assert np.array_equal(g.values, g2.values)
    assert np.array_equal(g.blocks["offset"], g2.blocks["offset"], equal_nan=True)


def test_constant_alignment_and_apply(tmp_path):
    al = change.ALSAlignment.constant((1.0, -2.0, 0.5), (0.1, 0.1, 0.01))
    assert np.allclose(al.offset_at([0.0, 100.0], [5.0, 7.0]), [[1.0, -2.0, 0.5]] * 2)
    assert np.allclose(al.sigma_at(3.0, 4.0), [[0.1, 0.1, 0.01]])
    c = PointCloud(np.array([[10.0, 10.0, 5.0]]), {"intensity": np.array([7], dtype=np.uint16)})
    m = al.apply(c)
    assert np.allclose(m.xyz, [[9.0, 12.0, 4.5]]) and np.allclose(c.xyz, [[10.0, 10.0, 5.0]])
    assert m.attrs["intensity"][0] == 7
    with pytest.raises(ValueError):
        change.ALSAlignment.constant((np.nan, 0, 0))
    with pytest.raises(ValueError):
        al.apply(np.zeros((3, 3)))


# ---------------------------------------------------------------- surfaces

def test_chm_change_finds_felled_crowns_and_little_else(pair, chm):
    ep = pair[0]
    unchanged, felled = zones(ep, chm.a)
    c = chm.classes
    assert set(np.unique(c)) <= {0, 1, 2, 3}
    assessed = unchanged & (c > 0)
    false = ((c == 2) | (c == 3))[assessed].mean()
    assert false < 0.05, false
    assert felled.sum() > 20
    assert (c[felled] == 3).mean() > 0.8, (c[felled] == 3).mean()
    # The interval brackets its bias, and the classes follow from it.
    ok = c > 0
    d = chm.difference.data
    assert np.all(chm.lower.data[ok] <= chm.bias.data[ok] + 1e-9)
    assert np.all(chm.upper.data[ok] >= chm.bias.data[ok] - 1e-9)
    assert np.all((d[c == 2] > chm.upper.data[c == 2]) & (d[c == 3] < chm.lower.data[c == 3]).all())
    assert np.allclose(chm.lod.data[ok], (chm.upper.data[ok] - chm.lower.data[ok]) / 2)
    assert np.all(np.isnan(d[c == 0]))
    assert chm.dod.area_changed == ((c == 2) | (c == 3)).sum() * 4.0
    assert chm.dod.volume_lost > 0 and chm.areas()["loss"] > 0
    # The second survey is sparser: its highest returns fall shorter of the
    # canopy tops, and the notes say so.
    assert chm.median_bias < 0
    assert chm.sensors["a"]["pulse_density"] > 1.5 * chm.sensors["b"]["pulse_density"]
    assert any("pulse density differs" in n for n in chm.notes)
    assert "CHM change" in chm.report()


def test_harmonisation_in_the_pass_equals_harmonised_catalogues(pair, tmp_path):
    ep, a, b, _ = pair
    ha, hb = change.harmonise(a, b, tmp_path / "ha", tmp_path / "hb", density_cell=10.0, seed=3)
    assert len(ha) and len(hb)
    in_pass = change.chm_change(a, b, resolution=2.0, harmonise=True, density_cell=10.0, seed=3,
                                buffer=25.0)
    from_tiles = change.chm_change(ha, hb, resolution=2.0, buffer=25.0)
    # The same pulses are kept; writing the tiles requantises coordinates by
    # up to a millimetre, which can move a return across a cell edge.
    assert from_tiles.sensors["a"]["pulses"] == in_pass.sensors["a_compared"]["pulses"]
    pulses_b = from_tiles.sensors["b"]["pulses"]
    assert abs(pulses_b - in_pass.sensors["b_compared"]["pulses"]) < 0.005 * pulses_b
    x, y = in_pass.a.cell_centers()
    inner = (x > 12) & (x < 58) & (y > 12) & (y < 58)
    for k in ("a", "b"):
        r = getattr(from_tiles, k)
        rows = np.floor((y[inner] - r.ymin) / r.resolution).astype(int)
        cols = np.floor((x[inner] - r.xmin) / r.resolution).astype(int)
        close = np.isclose(getattr(in_pass, k).data[inner], r.data[rows, cols], rtol=0, atol=2e-3)
        assert close.mean() > 0.97, (k, close.mean())
    s = in_pass.sensors
    ratio = s["a_compared"]["pulse_density"] / s["b_compared"]["pulse_density"]
    assert 0.8 < ratio < 1.25, ratio
    assert s["a"]["pulse_density"] > 1.5 * s["b"]["pulse_density"]
    assert "after harmonisation" in in_pass.report()


def test_surface_change_does_not_depend_on_the_chunks(pair, aligned):
    ep, a, b, _ = pair
    c1 = change.chm_change(a, b, 1.0, alignment=aligned, workers=1)
    c2 = change.chm_change(a, b, 1.0, alignment=aligned, chunk_size=23.0, workers=3)
    for k in ("a", "b", "difference", "bias"):
        u, v = getattr(c1, k).data, getattr(c2, k).data
        assert np.allclose(u, v, rtol=0, atol=1e-12, equal_nan=True), k
    assert np.array_equal(c1.classes, c2.classes)
    c3 = change.chm_change(a, b, 1.0, alignment=aligned, workers=4)
    for k in ("a", "b", "difference", "lod"):
        assert np.array_equal(getattr(c1, k).data, getattr(c3, k).data, equal_nan=True), k


def test_dtm_and_dsm_change(pair, aligned):
    ep, a, b, _ = pair
    # Terrain did not change: aligned, nearly every cell is below detection.
    d = change.dtm_change(a, b, 2.0, alignment=aligned)
    ok = d.classes > 0
    assert ok.mean() > 0.8
    assert ((d.classes == 2) | (d.classes == 3))[ok].mean() < 0.05
    assert np.nanmedian(np.abs(d.difference.data)) < 0.05
    # Without the alignment the 15 cm offset is a rise of the whole terrain.
    raw = change.dtm_change(a, b, 2.0)
    assert (raw.classes == 2)[raw.classes > 0].mean() > 0.9
    s = change.surface_change(a, b, "dsm", 2.0, alignment=aligned)
    assert s.surface == "dsm" and np.nanmax(s.a.data) > 20


# ---------------------------------------------------------------- gaps

def test_gaps_are_regions_with_outlines(tmp_path):
    data = np.full((12, 16), 20.0)
    data[2:6, 2:7] = 0.5      # 20 m², rows 2-5 (south up), columns 2-6
    data[8:10, 10:13] = 1.0   # 6 m²
    data[3, 4] = 20.0         # a tree standing in the first gap
    data[0, 15] = np.nan
    r = Raster(data, 100.0, 200.0, 1.0)
    g = change.canopy_gaps(r, height=2.0, min_area=5.0)
    assert len(g) == 2 and sorted(g.area) == [6.0, 19.0]
    k = int(np.argmax(g.area))
    (ext, holes), = g.polygons[k]
    assert len(holes) == 1

    def ring_area(p):
        return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])

    assert ring_area(ext) == 20.0 and ring_area(holes[0]) == -1.0
    assert ext[:, 0].min() == 102.0 and ext[:, 1].min() == 202.0
    assert g.gap_fraction == pytest.approx(25.0 / (12 * 16 - 1))
    assert len(change.canopy_gaps(r, min_area=10.0)) == 1
    edges, counts = g.size_distribution()
    assert counts.sum() == 2
    alpha, se, n = g.size_exponent()
    assert n == 2 and alpha > 1 and se > 0
    g.to_geojson(tmp_path / "g.geojson")
    fc = json.loads((tmp_path / "g.geojson").read_text())
    assert len(fc["features"]) == 2 and fc["features"][0]["geometry"]["type"] == "Polygon"
    with pytest.raises(ValueError):
        change.canopy_gaps(r, connectivity=6)
    with pytest.raises(ValueError):
        change.canopy_gaps(r, min_area=10.0, max_area=5.0)


def test_gap_formation_needs_significant_loss(pair, chm, tmp_path):
    ep = pair[0]
    g = change.gap_change(chm, height=2.0, min_area=5.0, years=5)
    s = g.summary()
    assert s["area_formed"] > 20
    # The felled trees' crowns lie in gaps that are new or expanded.
    gx, gy, _ = ep.gap
    lab = g.b.labels
    x, y = chm.a.cell_centers()
    at = lab.ravel()[np.argmin(np.hypot(x - gx, y - gy))]
    assert at > 0 and g.status_b[at - 1] in ("new", "expanded")
    assert g.formed_area[at - 1] > 10
    assert "formation_rate" in s and s["formation_rate"] > 0
    assert "formed" in g.report()
    g.to_geojson(tmp_path / "b.geojson")
    assert json.loads((tmp_path / "b.geojson").read_text())["features"][0]["properties"]["status"]
    # Without significance every crossing counts, so at least as much forms.
    raw = change.gap_change(chm.a, chm.b, height=2.0, min_area=5.0)
    assert raw.areas["formed"] >= g.areas["formed"] and raw.areas["uncertain"] == 0
    with pytest.raises(ValueError):
        change.gap_change(chm.a)


# ---------------------------------------------------------------- trees

def test_tree_change(pair, aligned, chm, tmp_path):
    ep, a, b, _ = pair
    kw = dict(resolution=1.0, window=als.LinearWindow(0.0, 0.2, 3.0, 20.0), max_cr=20, hmin=5.0)
    ta, tb = als.find_trees(a, **kw), als.find_trees(b, **kw)
    chm = change.chm_change(a, b, resolution=1.0, alignment=aligned)
    tc = change.tree_change(ta, tb, chm, alignment=aligned)
    t = tc.table
    assert len(tc) >= len(ta)
    counts = tc.counts()
    assert counts.get("survivor", 0) > 5 and counts.get("dead", 0) >= 1
    ok_a = {"survivor", "damaged", "dead", "undetected", "unobserved"}
    ok_b = {"recruit", "released", "undetected", "unobserved"}
    for s, ia in zip(t["status"], t["id_a"], strict=True):
        assert s in (ok_a if ia else ok_b)
    surv = t["status"] == "survivor"
    assert np.all(np.isfinite(t["sigma"][surv])) and np.all(t["lod"][surv] > 0)
    assert np.allclose(t["dh"][surv], (t["height_b"] - t["height_a"])[surv])
    # Dead trees are the felled ones.
    truth = ep.trees
    for k in np.flatnonzero(t["status"] == "dead"):
        d = np.hypot(truth["top_x"] - t["x"][k], truth["top_y"] - t["y"][k])
        d[np.isnan(d)] = np.inf
        assert truth["fate"][np.argmin(d)] in ("removed", "gap")
    s = tc.summary(area=70 * 70, years=5)
    assert s["n_growth"] > 0 and np.isfinite(s["mean_growth"]) and s["mortality_rate"] > 0
    assert s["dead_per_ha"] == pytest.approx(s["dead"] / 0.49)
    grid = tc.grid(35.0, bounds=(0, 0, 70, 70))
    assert grid["dead"].data.sum() == s["dead"] and grid["survivors"].data.sum() == s["survivors"]
    tc.to_csv(tmp_path / "t.csv")
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert lines[0].startswith("id_a,id_b,status") and len(lines) == len(tc) + 1
    assert "survivors" in tc.report(years=5)
    # A dict of columns works as well, and one tree matched with itself grew by 0.
    one = {"x": np.array([ta.x[0]]), "y": np.array([ta.y[0]]), "height": np.array([ta.height[0]])}
    same = change.tree_change(one, one, chm)
    assert same.table["status"][0] in ("survivor", "damaged") and same.table["dh"][0] == 0.0
    with pytest.raises(ValueError):
        change.tree_change({"x": [np.nan], "y": [0.0], "height": [5.0]}, one, chm)
    with pytest.raises(ValueError):
        change.tree_change(one, one, change.dtm_change(a, b, 2.0))
    with pytest.raises(ValueError):
        change.tree_change(one, one, chm, max_drop=1.0)


# ---------------------------------------------------------------- metrics and PAI

def test_metric_change(pair, aligned):
    ep, a, b, _ = pair
    m = change.metric_change(a, b, 35.0, metrics=["zq95", "cover", "n"], alignment=aligned,
                             permutations=40)
    assert m.names == ["zq95", "cover", "n"]
    sd = m.sigma["zq95"].data
    assert np.all(sd[np.isfinite(sd)] >= 0) and (sd > 0.01).sum() >= 4
    ok = m.classes["zq95"] > 0
    assert ok.any()
    d = m.difference["zq95"].data
    assert np.allclose(d[ok], (m.b["zq95"].data - m.a["zq95"].data)[ok])
    lo, hi = m.lower["zq95"].data, m.upper["zq95"].data
    assert np.allclose(m.lod["zq95"].data[ok], (hi - lo)[ok] / 2)
    c = m.classes["zq95"]
    assert np.all(d[c == 2] > hi[c == 2]) and np.all(d[c == 3] < lo[c == 3])
    assert set(m["cover"]) == {"a", "b", "difference", "bias", "lower", "upper", "lod", "sigma",
                               "classes"}
    assert "zq95" in m.report()
    # Values do not depend on the chunks; the bootstrap is seeded per cell.
    m2 = change.metric_change(a, b, 35.0, metrics=["zq95", "cover", "n"], alignment=aligned,
                              permutations=40, chunk_size=35.0, workers=2)
    for n in m.names:
        assert np.allclose(m.a[n].data, m2.a[n].data, rtol=0, atol=1e-9, equal_nan=True)
        assert np.allclose(m.lower[n].data, m2.lower[n].data, rtol=0, atol=1e-9, equal_nan=True)
    none = change.metric_change(a, b, 35.0, metrics="zq95", permutations=0)
    assert np.all(none.classes["zq95"] == 0) and np.isfinite(none.difference["zq95"].data).any()
    with pytest.raises(ValueError, match="unknown metric"):
        change.metric_change(a, b, 35.0, metrics=["nope"])


def profile(w0, w_layer, xmin=0.0):
    w = np.array([[[w0]], [[w_layer]], [[0.0]]], dtype=float)
    return ALSProfile(xmin, 0.0, 20.0, 1.0, 30.0, w, 0.5 * w)


def test_pai_change_from_counts():
    a, b = profile(400.0, 600.0), profile(200.0, 800.0)
    c = change.pai_change(a, b)

    def pai(p):
        return -np.log(p) / 0.5

    def sd(p):
        return np.sqrt((1 - p) / (1000 * p)) / 0.5

    assert c.pai_a.data[0, 0] == pytest.approx(pai(0.4))
    assert c.pai_b.data[0, 0] == pytest.approx(pai(0.2))
    assert c.sigma.data[0, 0] == pytest.approx(np.hypot(sd(0.4), sd(0.2)))
    assert c.mask("gain")[0, 0]
    assert change.pai_change(a, profile(0.0, 1000.0)).mask("saturated")[0, 0]
    assert c.profile["height"][0] == pytest.approx(16.0)
    p = change.profile_change(a, b, mask=np.ones((1, 1), bool))
    assert p["difference"][0] == pytest.approx(p["pad_b"][0] - p["pad_a"][0])
    with pytest.raises(ValueError, match="share their grid"):
        change.pai_change(a, profile(400.0, 600.0, xmin=5.0))
    with pytest.raises(ValueError):
        change.pai_change(a, "not a profile")
    with pytest.raises(ValueError):
        change.profile_change(a, b, mask=np.ones((2, 2), bool))


def test_pai_change_from_flights(tmp_path):
    ep = synthetic.als_epochs(n_trees=12, size=30.0, removed=0, gap=(15.0, 15.0, 4), growth=0.0,
                              offset=(0.0, 0.0, 0.0), buildings=False, seed=5)
    a, b = ep.write_tiles(tmp_path / "a", tmp_path / "b", size=25.0)
    bounds = (0.0, 0.0, 50.0, 50.0)
    kw = dict(resolution=10.0, bounds=bounds, max_height=30.0)
    pa = als.gap_profile(a, ep.flights[0].trajectory, **kw)
    pb = als.gap_profile(b, ep.flights[1].trajectory, **kw)
    c = change.pai_change(pa, pb)
    x, y = c.pai_a.cell_centers()
    gap = np.hypot(x - 15, y - 15) < 6
    assert np.nanmean(c.difference.data[gap]) < 0
    assert (c.classes[gap] == 3).any()


# ---------------------------------------------------------------- checks and command line

def test_bad_arguments(pair):
    ep, a, b, _ = pair
    with pytest.raises(ValueError, match="surface"):
        change.surface_change(a, b, "tree")
    with pytest.raises(ValueError):
        change.chm_change(a, b, resolution=0.0)
    with pytest.raises(ValueError):
        change.chm_change(a, b, confidence=1.0)
    with pytest.raises(ValueError, match="ALSAlignment"):
        change.chm_change(a, b, alignment="none")
    with pytest.raises(ValueError, match="dtm_method"):
        change.chm_change(a, b, dtm_method="spline")
    with pytest.raises(ValueError):
        change.align_surveys(a, b, model="spline")
    with pytest.raises(ValueError):
        change.align_surveys(a, b, stable_classes=())
    with pytest.raises(ValueError, match="no block"):
        change.align_surveys(a, b, stable_classes=(17,))
    with pytest.raises(ValueError):
        change.harmonise(a, b, "x", "y", density_cell=-1.0)


def test_command_line(pair, tmp_path, capsys):
    ep, a, b, d = pair
    out = tmp_path / "out"
    cli.main(["--no-progress", "als-change", str(d / "a"), str(d / "b"), str(out), "--align",
              "--stable-classes", "2,6", "--block-size", "35", "--gaps", "--gap-min-area", "5",
              "--resolution", "2"])
    for f in ("chm_a.asc", "chm_b.asc", "chm_difference.asc", "chm_lod.asc", "chm_classes.asc",
              "alignment.csv", "gaps_a.geojson", "gaps_b.geojson"):
        assert (out / f).exists(), f
    assert "CHM change" in capsys.readouterr().out
    csv = tmp_path / "trees.csv"
    cli.main(["--no-progress", "als-tree-change", str(d / "a"), str(d / "b"), str(csv),
              "--resolution", "1", "--window-linear", "0", "0.2", "3", "20", "--hmin", "5",
              "--grid", str(tmp_path / "grid"), "--grid-resolution", "35", "--years", "5"])
    assert csv.read_text().startswith("id_a,id_b,status")
    assert (tmp_path / "grid" / "dead.asc").exists()
    assert "Trees:" in capsys.readouterr().out
