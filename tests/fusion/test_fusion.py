# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""TLS and ALS together: registration against a known pose, tree links against
known trees, fused profiles against a known plant area, and regressions against
known relations and brute-force refits."""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from sylva import PointCloud, Raster, als, fusion, ground, registration, synthetic, voxels

R = 0.03


def pose(heading, t):
    m = registration.rotation_z(heading)
    m[:3, 3] = t
    return m


def pose_error(est, truth, at=(30.0, 30.0)):
    e = est @ np.linalg.inv(truth)
    p = np.array([at[0], at[1], 0.0, 1.0])
    d = (e @ p)[:3] - p[:3]
    return float(np.hypot(d[0], d[1])), float(d[2]), float(np.degrees(np.arctan2(e[1, 0], e[0, 0])))


@pytest.fixture(scope="module")
def stand():
    """A 60 m stand of solid crowns, flown and scanned from two positions."""
    trees = synthetic.stand(30, size=60.0, min_spacing=3.0, heights=(8, 20), seed=5)
    scene = synthetic.crown_forest(trees, size=60.0, ground_points=100, margin=0.0, seed=5)
    flight = synthetic.als_flight(scene, pulse_rate=8000, line_spacing=25.0, bounds=(0, 0, 60, 60),
                                  seed=5, target_radius=R)
    origins = [(x, y, float(synthetic.terrain_height(x, y)) + 1.5) for x, y in ((30, 30), (22, 36))]
    shots = fusion.synthetic_scan(scene, origins, resolution_deg=0.4, target_radius=R)
    tls = PointCloud.concatenate([s.to_pointcloud() for s in shots])
    tls = tls[np.hypot(tls.x - 30, tls.y - 30) < 18]
    return {"trees": np.array(trees), "scene": scene, "flight": flight, "shots": shots, "tls": tls}


# ------------------------------------------------------------ synthetic scanner


def test_synthetic_scan_is_a_turbid_medium():
    """Spheres in a slab stop a thin ray as Beer's law says, at the analytic terrain otherwise."""
    rng = np.random.default_rng(3)
    n_per, lo, hi = 25.0, 5.0, 9.0
    n = int(n_per * 50 * 50 * (hi - lo))
    xyz = np.column_stack([rng.uniform(-25, 25, n), rng.uniform(-25, 25, n), rng.uniform(lo, hi, n)])
    scene = PointCloud(xyz, {"classification": np.full(n, 4, np.uint8), "tree_id": np.ones(n, np.int32)})
    shots = fusion.synthetic_scan(scene, (0.0, 0.0, 1.5), resolution_deg=0.5, max_zenith_deg=20.0,
                                  target_radius=0.05, terrain_slope=0.0)
    assert shots.n_shots == 40 * 720
    cosz = shots.direction[:, 2]
    expected = np.mean(np.exp(-n_per * np.pi * 0.05 ** 2 * (hi - lo) / cosz))
    gap = np.mean(shots.echo_count == 0)
    assert gap == pytest.approx(expected, abs=0.015)
    assert set(np.unique(shots.echo_attrs["tree_id"])) == {1}
    # Downward pulses end on the terrain, as class 2.
    down = fusion.synthetic_scan(PointCloud(np.zeros((0, 3))), [(0.0, 0.0, 1.5)], resolution_deg=5.0,
                                 max_zenith_deg=180.0, terrain_slope=0.0)[0]
    below = down.direction[:, 2] < -0.2
    assert np.all(down.echo_count[below] == 1)
    assert np.all(down.echo_attrs["classification"] == 2)
    with pytest.raises(ValueError, match="above the terrain"):
        fusion.synthetic_scan(scene, (0.0, 0.0, -3.0))
    with pytest.raises(ValueError, match="resolution_deg"):
        fusion.synthetic_scan(scene, (0.0, 0.0, 1.5), resolution_deg=0.0)


# ------------------------------------------------------------------ register


@pytest.mark.parametrize("case", ["local", "gnss"])
def test_register_recovers_a_known_pose(stand, case):
    if case == "local":
        truth = pose(-128.0, [31.2, 28.1, 1.3])          # a scanner frame at the plot, turned
        initial = pose(0.0, [30.0, 30.0, 0.0])            # position known roughly, heading not at all
        kw = {"search_radius": 5.0}
    else:
        truth = pose(1.2, [1.6, -2.2, 0.35])             # georeferenced with a GNSS error
        initial = np.eye(4)
        kw = {"search_radius": 4.0, "heading_range": 4.0, "heading_step": 1.0}
    tls = stand["tls"].transform(np.linalg.inv(truth))
    reg = fusion.register(tls, stand["flight"].points, initial, **kw)
    h, v, a = pose_error(reg.transform, truth)
    assert h < 0.05 and abs(v) < 0.03 and abs(a) < 0.2, (h, v, a)
    # From two scan positions the TLS canopy model is read from below and to
    # the side, so the search alone can be a few metres and degrees off; the
    # ICP from the peaks corrects it.
    hs, vs, as_ = pose_error(reg.search_transform, truth)
    assert hs < 3.0 and abs(as_) < 8.0
    assert reg.icp["accepted"]
    assert reg.residuals["ground_nmad"] < 0.05
    assert reg.covariance.shape == (6, 6) and len(reg.jackknife) == 4
    s = reg.std
    assert all(np.isfinite(v) for v in s.values()) and s["x"] < 0.1
    assert 0.0 <= reg.ambiguity <= 1.0
    assert "TLS to ALS registration" in reg.report()
    moved = reg.apply(tls)
    np.testing.assert_allclose(moved.xyz, stand["tls"].xyz, atol=0.1)


def test_register_from_a_catalogue_and_json(stand, tmp_path):
    truth = pose(0.8, [1.0, 1.5, -0.2])
    tls = stand["tls"].transform(np.linalg.inv(truth))
    cat = stand["flight"].write_tiles(tmp_path / "tiles", size=30.0)
    a = fusion.register(tls, cat, search_radius=3.0, heading_range=2.0, heading_step=1.0, jackknife=False)
    b = fusion.register(tls, stand["flight"].points, search_radius=3.0, heading_range=2.0, heading_step=1.0,
                        jackknife=False)
    # Tiles quantise to 1 mm; both answers are right to a centimetre or two.
    for r in (a, b):
        h, v, _ = pose_error(r.transform, truth)
        assert h < 0.03 and abs(v) < 0.03
    assert a.jackknife is None
    a.save(tmp_path / "reg.json")
    back = fusion.Registration.load(tmp_path / "reg.json")
    np.testing.assert_allclose(back.transform, a.transform)
    assert back.icp == a.icp and back.settings["search_radius"] == 3.0
    assert back.resolution == a.resolution


def test_register_without_refinement_and_ground_only(stand):
    truth = pose(0.5, [0.8, -0.6, 0.2])
    tls = stand["tls"].transform(np.linalg.inv(truth))
    n = fusion.register(tls, stand["flight"].points, search_radius=2.0, heading_range=2.0, heading_step=1.0,
                        refine="none")
    assert n.icp is None and n.covariance is None and n.jackknife is None
    np.testing.assert_array_equal(n.transform, n.search_transform)
    assert pose_error(n.transform, truth)[0] < 0.5
    g = fusion.register(tls, stand["flight"].points, search_radius=2.0, heading_range=2.0, heading_step=1.0,
                        refine="ground", jackknife=False)
    assert abs(pose_error(g.transform, truth)[1]) < 0.03


def test_register_does_not_depend_on_threads(stand, tmp_path):
    truth = pose(0.7, [0.9, -1.1, 0.1])
    tls = stand["tls"].transform(np.linalg.inv(truth))
    np.save(tmp_path / "tls.npy", tls.xyz)
    np.save(tmp_path / "tls_g.npy", tls.attrs["classification"] == 2)
    p = stand["flight"].points
    np.save(tmp_path / "als.npy", p.xyz)
    np.save(tmp_path / "als_g.npy", p.attrs["classification"] == 2)
    code = (
        "import numpy as np, sys\n"
        "from sylva import PointCloud, fusion\n"
        "d = sys.argv[1]\n"
        "t = PointCloud(np.load(d + '/tls.npy'), {'classification': np.where(np.load(d + '/tls_g.npy'), 2, 4)})\n"
        "a = PointCloud(np.load(d + '/als.npy'), {'classification': np.where(np.load(d + '/als_g.npy'), 2, 4)})\n"
        "r = fusion.register(t, a, search_radius=3.0, heading_range=3.0, heading_step=1.0)\n"
        "np.save(d + '/out_' + sys.argv[2] + '.npy', np.concatenate([r.transform.ravel(), r.jackknife]))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    for threads in ("1", "4"):
        subprocess.run([sys.executable, "-c", code, str(tmp_path), threads], check=True,
                       env={**env, "RAYON_NUM_THREADS": threads})
    np.testing.assert_array_equal(np.load(tmp_path / "out_1.npy"), np.load(tmp_path / "out_4.npy"))


def test_register_refuses_bad_input(stand):
    tls, als_pts = stand["tls"], stand["flight"].points
    with pytest.raises(ValueError, match="classification"):
        fusion.register(PointCloud(tls.xyz), als_pts)
    with pytest.raises(ValueError, match="empty"):
        fusion.register(tls[np.zeros(len(tls), bool)], als_pts)
    with pytest.raises(ValueError, match="4, 4"):
        fusion.register(tls, als_pts, np.eye(3))
    with pytest.raises(ValueError, match="no ALS points|within reach"):
        fusion.register(tls, als_pts, pose(0.0, [5000.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="refine"):
        fusion.register(tls, als_pts, refine="icp")
    with pytest.raises(ValueError, match="tls_ground has"):
        fusion.register(tls, als_pts, tls_ground=np.ones(3, bool))
    with pytest.raises(ValueError, match="resolution"):
        fusion.register(tls, als_pts, resolution=-1.0)
    with pytest.raises(ValueError, match="ground points"):
        fusion.register(tls, als_pts, tls_ground=np.zeros(len(tls), bool))


# ---------------------------------------------------------------------- trees


def _square(cx, cy, r):
    return np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]])


def test_link_trees_reports_the_trees_under_a_crown(tmp_path):
    als_t = {"id": np.array([11, 12, 13]), "x": np.array([0.0, 10.0, 30.0]), "y": np.zeros(3),
             "height": np.array([20.0, 15.0, 18.0]),
             "crowns": [_square(0, 0, 3), _square(10, 0, 2.5), _square(30, 0, 3)]}
    # The TLS trees are in their own frame, turned by 90 degrees and shifted.
    m2 = pose(90.0, [5.0, -3.0, 0.0])
    xy = np.array([[1.5, 0.5], [-1.8, -1.0], [10.3, 0.2], [11.5, 1.0], [20.0, 0.0]])
    local = (np.linalg.inv(m2) @ np.column_stack([xy, np.zeros(5), np.ones(5)]).T).T[:, :2]
    tls_t = {"tree_id": np.array([1, 2, 3, 4, 5]), "x": local[:, 0], "y": local[:, 1],
             "dbh": np.array([0.15, 0.45, 0.30, 0.31, 0.2]), "height": np.array([8.0, 19.6, 14.0, 14.8, 9.0])}
    links = fusion.link_trees(tls_t, als_t, m2, volumes={2: 1.5, 3: 0.4})
    assert list(links.status) == ["suppressed", "matched", "matched", "codominant", "unlinked"]
    t = links.table()
    assert list(t["als_id"]) == [11, 11, 12, 12, -1]
    np.testing.assert_allclose(t["x"], xy[:, 0], atol=1e-9)
    assert links.one_to_many() == {11: [2, 1], 12: [3, 4]}
    c = links.counts()
    assert c["matched"] == 2 and c["one_to_many"] == 2 and c["als_unmatched"] == 1
    assert t["height"][1] == 20.0 and t["height_source"][0] == "tls"
    assert np.isnan(t["volume"][0]) and t["volume"][1] == 1.5
    a = links.als_table()
    assert list(a["tls_id"]) == [2, 3, -1] and a["stems"][0] == [1, 2]
    links.to_csv(tmp_path / "links.csv")
    rows = list(csv.DictReader(open(tmp_path / "links.csv")))
    assert rows[0]["status"] == "suppressed" and rows[4]["als_height"] == ""
    # A sighting given by tree_sampling decides the flags.
    s = {"tree_id": np.array([1, 2]), "above_observed_fraction": np.array([0.1, 0.9])}
    l2 = fusion.link_trees(tls_t, als_t, m2, sampling=s)
    assert list(l2.flag[:2]) == ["top_not_seen", "top_seen"] and list(l2.top_seen[:2]) == [0.0, 1.0]
    with pytest.raises(ValueError, match="not both"):
        fusion.link_trees(tls_t, als_t, top_seen=[1] * 5, sampling=s)
    with pytest.raises(ValueError, match="finite"):
        fusion.link_trees({"x": [np.nan], "y": [0.0]}, als_t)
    with pytest.raises(ValueError, match="'x' and 'y'"):
        fusion.link_trees({"x": [1.0]}, als_t)
    with pytest.raises(ValueError, match="top_seen has"):
        fusion.link_trees(tls_t, als_t, top_seen=[1, 0])
    empty = fusion.link_trees({"x": [], "y": []}, als_t)
    assert len(empty.status) == 0 and all(v < 0 for v in empty.als_tls)


def test_link_trees_on_a_scanned_stand(stand):
    """Stems from the TLS pipeline linked to ALS crowns: every stem whose tree the
    ALS found is linked to it, and the combined heights are those of the trees."""
    from sylva import filters, trees

    tls = filters.voxel_downsample(stand["tls"], 0.02)
    tls = ground.normalize_height(tls, ground.make_dtm(tls, 0.5))
    st = trees.detect_stems(tls)
    st, _ = trees.merge_branches(tls, st)
    lab = trees.segment_trees(tls, st)
    trees.tree_heights(tls, lab, st)
    st, lab = trees.prune_trees(st, lab, min_height=3.0)
    pts = stand["flight"].points
    h = ground.normalize_height(pts, ground.make_dtm(pts, 1.0)).attrs["height"]
    at = als.segment_trees(pts, heights=h, window=als.LinearWindow(0.0, 0.2, 2.0, 20.0), max_cr=20)
    links = fusion.link_trees(st, at)
    t = links.table()
    truth = stand["trees"]
    near = [np.argmin(np.hypot(truth[:, 0] - x, truth[:, 1] - y)) for x, y in zip(t["x"], t["y"], strict=True)]
    tid = pts.attrs["tree_id"]
    als_tree = [np.bincount(tid[(at.tree_id == i) & (h > 0.5 * at.height[k]) & (tid > 0)]).argmax()
                for k, i in enumerate(at.id)]
    matched = links.status == "matched"
    assert matched.sum() >= 2
    right = [als_tree[links.als_index[i]] == near[i] + 1 for i in np.where(matched)[0]]
    assert all(right)
    err = t["height"][matched] - truth[np.array(near)[matched], 3]
    assert np.sqrt(np.mean(err ** 2)) < 0.5


# ---------------------------------------------------------- clouds, profiles


def test_merge_clouds_keeps_tls_below_and_als_above(stand):
    pts = stand["flight"].points
    dtm = ground.make_dtm(pts, 1.0)
    tls = stand["tls"]
    m = fusion.merge_clouds(tls, pts, dtm, split=6.0)
    src, h = m.attrs["source"], m.attrs["height"]
    assert set(np.unique(src)) == {1, 2}
    assert np.all(h[src == 1] < 6.0) and np.all(h[src == 2] >= 6.0)
    assert "classification" in m.attrs and "tree_id" in m.attrs
    n_tls = int(np.sum(ground.normalize_height(tls, dtm).attrs["height"] < 6.0))
    assert int(np.sum(src == 1)) == n_tls
    # A raster split: 2 m in the west half, 10 m in the east.
    r = Raster(np.array([[2.0, 10.0]]), 0.0, 0.0, 30.0)
    r.data = np.repeat(r.data, 2, axis=0)
    m2 = fusion.merge_clouds(tls, pts, dtm, split=r)
    west = m2.x < 30
    assert np.all(m2.attrs["height"][west & (m2.attrs["source"] == 1)] < 2.0)
    assert np.all(m2.attrs["height"][~west & (m2.attrs["source"] == 1)] < 10.0)
    # A transform is applied to the TLS first.
    shift = pose(0.0, [100.0, 0.0, 0.0])
    m3 = fusion.merge_clouds(tls.transform(np.linalg.inv(shift)), pts, dtm, split=6.0, transform=shift)
    assert len(m3) == len(m)


def _sphere_slab(pad, lo, hi, half, seed):
    rng = np.random.default_rng(seed)
    n = int(pad / (2 * np.pi * R * R) * (2 * half) ** 2 * (hi - lo))
    xyz = np.column_stack([rng.uniform(-half, half, n) + 20, rng.uniform(-half, half, n) + 20,
                           rng.uniform(lo, hi, n)])
    return PointCloud(xyz, {"classification": np.full(n, 4, np.uint8)})


def test_fused_profile_recovers_known_layers():
    """An understorey layer (1-4 m) and a canopy layer (10-16 m) of known plant
    area density, scanned from below and flown over."""
    scene = PointCloud.concatenate([_sphere_slab(0.4, 1.0, 4.0, 30, 1), _sphere_slab(0.3, 10.0, 16.0, 30, 2)])
    flight = synthetic.als_flight(scene, altitude=60.0, line_spacing=20.0, pulse_rate=30_000,
                                  footprint_samples=1, target_radius=R, terrain_slope=0.0,
                                  bounds=(0, 0, 40, 40), range_noise=0.0, clip=False, seed=2)
    shots = fusion.synthetic_scan(scene, [(20.0, 20.0, 1.5)], resolution_deg=0.5, target_radius=R,
                                  terrain_slope=0.0)[0]
    b = (5.0, 5.0, -2.0, 35.0, 35.0, 20.0)
    g_tls = voxels.ray_voxelize(shots, 1.0, ((b[0], b[1], b[2]), (b[3], b[4], b[5])), ground_class=2,
                                unbounded_range=40.0)
    g_als = als.ray_voxelize(flight.points, flight.trajectory, voxel_size=1.0, bounds=b)
    dtm = Raster(np.zeros((40, 40)), 0.0, 0.0, 1.0)
    f = fusion.fuse_profiles(g_tls, g_als, dtm, (20.0, 20.0, 8.0), bin_size=2.0)
    seen = np.isfinite(f.weight_tls)
    np.testing.assert_allclose((f.weight_tls + f.weight_als)[seen], 1.0)
    under = (f.height >= 1) & (f.height + 2 <= 4)
    canopy = (f.height >= 10) & (f.height + 2 <= 16)
    assert np.mean(f.pad[under]) == pytest.approx(0.4, rel=0.15)
    assert np.mean(f.pad[canopy]) == pytest.approx(0.3, rel=0.15)
    # The TLS under the dense understorey outweighs the ALS, which sees it only
    # through the canopy.
    assert np.all(f.weight_tls[under] > 0.5)
    assert f.pai() == pytest.approx(0.4 * 3 + 0.3 * 6, rel=0.15)
    assert f.pai("tls") > 0 and f.pai("als") > 0
    best = fusion.fuse_profiles(g_tls, g_als, dtm, (20.0, 20.0, 8.0), bin_size=2.0, mode="best")
    w = best.weight_tls[np.isfinite(best.weight_tls)]
    assert set(np.unique(w)) <= {0.0, 1.0}
    obs = fusion.fuse_profiles(g_tls, g_als, dtm, (5.0, 5.0, 35.0, 35.0), mode="observed")
    assert obs.bin_size == 1.0 and len(obs.table()["pad"]) == len(obs.height)
    square = np.array([[12.0, 12.0], [28.0, 12.0], [28.0, 28.0], [12.0, 28.0]])
    poly = fusion.fuse_profiles(g_tls, g_als, dtm, square, estimator="pooled")
    assert np.nanmax(poly.pad) > 0
    with pytest.raises(ValueError, match="mode"):
        fusion.fuse_profiles(g_tls, g_als, dtm, mode="mean")
    with pytest.raises(ValueError, match="estimator"):
        fusion.fuse_profiles(g_tls, g_als, dtm, estimator="median")
    with pytest.raises(ValueError, match="no 'pad_ppl' field"):
        fusion.fuse_profiles(g_tls, g_als, dtm, pad_field="pad_ppl")
    with pytest.raises(ValueError, match="area"):
        fusion.fuse_profiles(g_tls, g_als, dtm, (1.0, 2.0))
    with pytest.raises(ValueError, match="no column"):
        fusion.fuse_profiles(g_tls, g_als, dtm, (500.0, 500.0, 1.0))
    with pytest.raises(ValueError, match="RayVoxelGrid"):
        fusion.fuse_profiles(np.zeros(3), g_als, dtm)
    # The fused profile's split height steers a merged cloud.
    m = fusion.merge_clouds(shots.to_pointcloud(), flight.points, dtm, split=f)
    assert np.all(m.attrs["height"][m.attrs["source"] == 1] < f.split_height)


# ---------------------------------------------------------------- upscaling


def test_plot_values():
    t = {"dbh": np.array([0.2, 0.4, 0.05]), "volume": np.array([0.5, 2.0, 0.01]), "tree_id": np.array([1, 2, 3])}
    v = fusion.plot_values(t, 1000.0, 500.0, min_dbh=0.1)
    assert v["n_trees"] == 2 and v["agb"] == pytest.approx(2.5 * 0.5 / 0.1)
    assert v["basal_area"] == pytest.approx(np.pi / 4 * (0.04 + 0.16) / 0.1)
    assert v["stems"] == pytest.approx(20.0) and v["volume"] == pytest.approx(25.0)
    v2 = fusion.plot_values({"dbh": t["dbh"], "tree_id": t["tree_id"]}, 1000.0, 500.0, volumes={1: 0.5, 2: 2.0})
    assert v2["missing_volume"] == 1 and v2["agb"] == pytest.approx(v["agb"])
    with pytest.raises(ValueError, match="positive"):
        fusion.plot_values(t, 0.0, 500.0)


def _power_law(n=12, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    h = rng.uniform(8, 30, n)
    c = rng.uniform(30, 95, n)
    agb = 0.8 * h ** 1.6 * c ** 0.5 * np.exp(rng.normal(0, noise, n))
    return h, c, agb


def test_fit_model_recovers_a_known_relation():
    h, c, agb = _power_law(noise=0.0)
    m = fusion.fit_model(agb, {"zq95": h, "cover": c})
    np.testing.assert_allclose(m.coef, [np.log(0.8), 1.6, 0.5], atol=1e-9)
    assert m.r2 == pytest.approx(1.0) and m.correction == pytest.approx(1.0)
    h, c, agb = _power_law(n=40, seed=1)
    m = fusion.fit_model(agb, {"zq95": h, "cover": c})
    x = np.column_stack([np.ones(40), np.log(h), np.log(c)])
    resid = np.log(agb) - x @ m.coef
    s2 = resid @ resid / 37
    np.testing.assert_allclose(m.se, np.sqrt(s2 * np.diag(np.linalg.inv(x.T @ x))), rtol=1e-9)
    for b, se, true in zip(m.coef[1:], m.se[1:], [1.6, 0.5], strict=True):
        assert abs(b - true) < 4 * se
    assert "loglog model, 40 plots" in m.summary() and "ln(zq95)" in m.equation()
    # Against numpy's least squares and a brute-force leave-one-out.
    x = np.column_stack([np.ones(40), np.log(h), np.log(c)])
    ref, *_ = np.linalg.lstsq(x, np.log(agb), rcond=None)
    np.testing.assert_allclose(m.coef, ref, rtol=1e-10)
    for i in (0, 17, 39):
        keep = np.arange(40) != i
        mi = fusion.fit_model(agb[keep], {"zq95": h[keep], "cover": c[keep]})
        p = mi.predict({"zq95": h[i:i + 1], "cover": c[i:i + 1]})["mean"][0]
        assert m.loo[i] == pytest.approx(p, rel=1e-10)
    lin = fusion.fit_model(agb, np.column_stack([h, c]), "linear", names=["h", "c"])
    ref, *_ = np.linalg.lstsq(np.column_stack([np.ones(40), h, c]), agb, rcond=None)
    np.testing.assert_allclose(lin.coef, ref, rtol=1e-10)
    assert lin.names == ["h", "c"] and lin.correction == 1.0


def test_prediction_intervals_match_student_t():
    h, c, agb = _power_law(n=15, seed=2)
    m = fusion.fit_model(agb, {"zq95": h}, "linear")
    p = m.predict({"zq95": np.array([10.0, np.nan, 100.0])}, level=0.9)
    x0 = np.array([1.0, 10.0])
    se = m.sigma * np.sqrt(1 + x0 @ m.xtx_inv @ x0)
    assert p["se"][0] == pytest.approx(se)
    stats = pytest.importorskip("scipy.stats")
    assert p["upper"][0] - p["mean"][0] == pytest.approx(stats.t.ppf(0.95, 13) * se, rel=1e-9)
    assert np.isnan(p["mean"][1]) and p["extrapolated"][2] and not p["extrapolated"][0]
    ll = fusion.fit_model(agb, {"zq95": h})
    q = ll.predict({"zq95": np.array([15.0, -1.0])})
    s = ll.sigma * np.sqrt(1 + np.array([1, np.log(15)]) @ ll.xtx_inv @ np.array([1, np.log(15)]))
    m15 = ll.coef[0] + ll.coef[1] * np.log(15)
    assert q["lower"][0] == pytest.approx(np.exp(m15 - stats.t.ppf(0.975, 13) * s))
    assert q["mean"][0] == pytest.approx(np.exp(m15 + ll.sigma ** 2 / 2))
    assert np.isnan(q["mean"][1])


def test_fit_model_refuses_bad_input():
    h, c, agb = _power_law(n=6)
    with pytest.raises(ValueError, match="too few"):
        fusion.fit_model(agb[:3], {"zq95": h[:3]})
    with pytest.raises(ValueError, match="positive"):
        fusion.fit_model(np.r_[agb[:-1], 0.0], {"zq95": h})
    with pytest.raises(ValueError, match="collinear"):
        fusion.fit_model(agb, {"a": h, "b": 2 * h}, "linear")
    with pytest.raises(ValueError, match="finite"):
        fusion.fit_model(np.r_[agb[:-1], np.nan], {"zq95": h}, "linear")
    with pytest.raises(ValueError, match="no predictor"):
        fusion.fit_model(agb, {"zq95": h}, names=["cover"])
    with pytest.raises(ValueError, match="model"):
        fusion.fit_model(agb, {"zq95": h}, "quadratic")
    with pytest.raises(ValueError, match="rows"):
        fusion.fit_model(agb, {"zq95": h[:-1]})


def test_upscale_predicts_rasters(tmp_path):
    h, c, agb = _power_law(n=20, seed=3)
    rng = np.random.default_rng(4)
    gh = Raster(rng.uniform(8, 30, (5, 6)), 100.0, 200.0, 25.0, "EPSG:32755")
    gc = Raster(rng.uniform(30, 95, (5, 6)), 100.0, 200.0, 25.0, "EPSG:32755")
    gh.data[0, 0] = np.nan
    up = fusion.upscale(agb, {"zq95": h, "cover": c}, {"zq95": gh, "cover": gc}, ["zq95", "cover"])
    assert up.mean.shape == (5, 6) and up.mean.crs == "EPSG:32755" and up.mean.xmin == 100.0
    assert np.isnan(up.mean.data[0, 0]) and np.isnan(up.extrapolated.data[0, 0])
    truth = 0.8 * gh.data ** 1.6 * gc.data ** 0.5
    ok = np.isfinite(truth)
    assert np.median(np.abs(up.mean.data[ok] / truth[ok] - 1)) < 0.1
    assert np.all(up.lower.data[ok] < up.mean.data[ok]) and np.all(up.upper.data[ok] > up.mean.data[ok])
    files = up.to_ascii_grids(tmp_path / "out", prefix="agb_")
    assert len(files) == 5 and Raster.from_ascii_grid(files[0]).shape == (5, 6)
    with pytest.raises(ValueError, match="no 'zmean'"):
        fusion.upscale(agb, {"zq95": h}, {"zq95": gh}, ["zmean"])
    other = Raster(np.ones((5, 6)), 0.0, 0.0, 25.0)
    with pytest.raises(ValueError, match="one grid"):
        fusion.upscale(agb, {"zq95": h, "cover": c}, {"zq95": gh, "cover": other}, ["zq95", "cover"])


def test_upscale_on_als_plot_and_grid_metrics(stand, tmp_path):
    cat = stand["flight"].write_tiles(tmp_path / "tiles", size=30.0)
    grid = als.grid_metrics(cat, 15.0, ["zq95", "cover"], min_height=0.0)
    centres = np.array([[7.5 + 15 * i, 7.5 + 15 * j] for i in range(4) for j in range(4)])
    pm = als.plot_metrics(cat, centres, radius=np.sqrt(225 / np.pi), metrics=["zq95", "cover"], min_height=0.0)
    tr = stand["trees"]
    agb = []
    for x, y in centres:
        m = (np.abs(tr[:, 0] - x) < 7.5) & (np.abs(tr[:, 1] - y) < 7.5)
        vol = 0.5 * np.pi / 4 * tr[m, 2] ** 2 * tr[m, 3]
        agb.append(fusion.plot_values({"dbh": tr[m, 2], "volume": vol}, 225.0, 600.0)["agb"] + 1.0)
    up = fusion.upscale(np.array(agb), pm, grid, ["zq95"], "linear")
    assert up.mean.shape == grid["zq95"].shape
    assert up.model.n == 16 and np.isfinite(up.model.loo_rmse)


# -------------------------------------------------------------- command line


def test_command_line(stand, tmp_path):
    from sylva import io
    from sylva.cli import main

    truth = pose(0.6, [0.7, -0.9, 0.2])
    io.write(stand["tls"].transform(np.linalg.inv(truth)), tmp_path / "tls.laz")
    stand["flight"].write_tiles(tmp_path / "tiles", size=30.0)
    main(["--no-progress", "fusion-register", str(tmp_path / "tls.laz"), str(tmp_path / "tiles"),
          str(tmp_path / "reg.json"), "--search-radius", "2", "--heading-range", "2", "--heading-step", "1",
          "--no-jackknife", "--transformed", str(tmp_path / "tls_map.laz")])
    reg = fusion.Registration.load(tmp_path / "reg.json")
    assert pose_error(reg.transform, truth)[0] < 0.05
    assert (tmp_path / "tls_map.laz").exists()
    with open(tmp_path / "tls_trees.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree_id", "x", "y", "dbh", "height"])
        w.writerow([1, 1.5, 0.5, 0.15, 8.0])
        w.writerow([2, -1.8, -1.0, 0.45, 19.6])
    with open(tmp_path / "als_trees.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "height", "crown_area", "n_points"])
        w.writerow([7, 0.0, 0.0, 20.0, 36.0, 100])
    ring = _square(0, 0, 3).tolist()
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"id": 7},
                                                     "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}}]}
    (tmp_path / "crowns.geojson").write_text(json.dumps(gj))
    main(["--no-progress", "fusion-trees", str(tmp_path / "tls_trees.csv"), str(tmp_path / "als_trees.csv"),
          str(tmp_path / "links.csv"), "--crowns", str(tmp_path / "crowns.geojson"),
          "--als-output", str(tmp_path / "als_links.csv")])
    rows = list(csv.DictReader(open(tmp_path / "links.csv")))
    assert [r["status"] for r in rows] == ["suppressed", "matched"]
    a = list(csv.DictReader(open(tmp_path / "als_links.csv")))
    assert a[0]["stems"] == "1 2" and a[0]["tls_id"] == "2"
    h, c, agb = _power_law(n=12, seed=5)
    with open(tmp_path / "plots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plot", "zq95", "cover", "agb"])
        for k in range(12):
            w.writerow([k, h[k], c[k], agb[k]])
    (tmp_path / "metrics").mkdir()
    Raster(np.full((3, 4), 20.0), 0.0, 0.0, 25.0).to_ascii_grid(tmp_path / "metrics" / "zq95.asc")
    Raster(np.full((3, 4), 60.0), 0.0, 0.0, 25.0).to_ascii_grid(tmp_path / "metrics" / "cover.asc")
    main(["--no-progress", "fusion-upscale", str(tmp_path / "plots.csv"), str(tmp_path / "metrics"),
          str(tmp_path / "up"), "--response", "agb", "--predictors", "zq95,cover"])
    mean = Raster.from_ascii_grid(tmp_path / "up" / "mean.asc")
    assert mean.data[0, 0] == pytest.approx(0.8 * 20 ** 1.6 * 60 ** 0.5, rel=0.15)
    with pytest.raises(SystemExit):
        main(["--no-progress", "fusion-upscale", str(tmp_path / "plots.csv"), str(tmp_path / "metrics"),
              str(tmp_path / "up"), "--response", "biomass", "--predictors", "zq95"])
