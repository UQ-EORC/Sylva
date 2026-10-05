"""Parity cases for sylva.coreg.pipeline: scans, pairs and whole surveys.

Surveys come from :mod:`parity.coreg_scenes` (seeded NumPy only). Timings
are left out of every recorded message, report and saved file.
"""

import dataclasses
import functools
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from parity import coreg_scenes as cs
from sylva import PointCloud, io
from sylva.coreg import pipeline as pl
from sylva.coreg import transforms as tf
from sylva.coreg import OptimisationResult, PoseGraph, ReflectorMatch, fit_ground
from sylva.coreg.icp import ICPResult, ICPTarget
from sylva.coreg.matching import match_stem_maps
from sylva.coreg.reflectors import Reflector
from sylva.coreg.stems import StemMap

TIMING = re.compile(r"\(\d+\.\d s\)|in \d+\.\d s")
# With workers=0 the count is the machine's CPUs; results do not depend on it.
WORKERS = re.compile(r"on \d+ workers")


def _untimed(text):
    """A log line without what depends on the machine: timings, worker counts."""
    return WORKERS.sub("on N workers", TIMING.sub("", text))


def _thin(a, step=7):
    """Every ``step``-th row, the count and the column sums: large arrays recorded compactly."""
    a = np.asarray(a)
    return {"rows": a[::step], "n": len(a), "sum": a.astype(float).sum(axis=0)}


def _put(out, key, a, step=7):
    for k, v in _thin(a, step).items():
        out[f"{key}_{k}"] = v


def _cfg(**kw):
    return dataclasses.replace(pl.CoregConfig(verbose=False, workers=3), **kw)


def _strings(values):
    return np.array([str(v) for v in values], dtype="U400").reshape(-1)


@functools.lru_cache(maxsize=None)
def _scene():
    return cs.survey(1, [(-6, -4), (6, -3), (0, 7), (-8, 6)],
                     targets=[(-3, -1, 1.2), (4, 1, 0.8), (1, 4, 1.5), (-5, 3, 1.0), (7, 6, 1.1), (-1, -7, 0.9)])


@functools.lru_cache(maxsize=None)
def _prepared():
    sc = _scene()
    cfg = _cfg()
    return tuple(pl.prepare_scan(c, cfg, name=f"scan_{k:02d}",
                                 reflectors=[Reflector(*p) for p in sc.targets[k]])
                 for k, c in enumerate(sc.clouds))


@functools.lru_cache(maxsize=None)
def _stranger():
    sc = cs.survey(77, [(0, 0)])
    return pl.prepare_scan(sc.clouds[0], _cfg(), name="stranger")


def _relative(k, ref=0):
    sc = _scene()
    return tf.invert(sc.truth[ref]) @ sc.truth[k]


def _scan_out(f, pre):
    out = {
        f"{pre}_name": f.name,
        f"{pre}_n_points": f.n_points,
        f"{pre}_error": f.error,
        f"{pre}_usable": f.usable,
        f"{pre}_repr": repr(f),
        f"{pre}_levelling": f.levelling,
        f"{pre}_origin": f.origin,
        f"{pre}_source": str(f.source),
        f"{pre}_has_ground": f.ground is not None,
        f"{pre}_stems": np.array([[s.x, s.y, s.z, s.dbh, *s.axis, s.reference_height, s.n_slices, s.n_points,
                                   s.rmse, s.coverage, s.lean_deg, s.quality] for s in f.stem_map]).reshape(-1, 14),
        f"{pre}_stem_map_name": f.stem_map.name,
        f"{pre}_reflectors": np.array([[r.x, r.y, r.z, r.reflectance, r.diameter, r.n_points]
                                       for r in f.reflectors]).reshape(-1, 6),
    }
    _put(out, f"{pre}_icp_points", f.icp_points)
    _put(out, f"{pre}_icp_heights", f.icp_heights)
    if f.ground is not None:
        g = f.ground
        out.update({f"{pre}_elevation": g.elevation, f"{pre}_ground_origin": g.origin,
                    f"{pre}_cell_size": g.cell_size, f"{pre}_observed": g.observed})
    return out


def _match_out(m, pre):
    return {
        f"{pre}_transform": m.transform, f"{pre}_n_inliers": m.n_inliers, f"{pre}_inlier_rmse": m.inlier_rmse,
        f"{pre}_score": m.score, f"{pre}_correspondences": np.asarray(m.correspondences).reshape(-1, 2),
        f"{pre}_success": m.success, f"{pre}_ambiguity": m.ambiguity, f"{pre}_has_rival": m.rival is not None,
    }


def _pair_out(p, pre):
    out = {
        f"{pre}_ij": np.array([p.i, p.j]),
        f"{pre}_names": _strings([p.name_i, p.name_j]),
        f"{pre}_success": p.success,
        f"{pre}_reason": p.reason,
        f"{pre}_transform": p.transform,
        f"{pre}_coarse": p.coarse_transform,
        f"{pre}_matched_source": np.asarray(p.matched_source).reshape(-1, 3),
        f"{pre}_matched_target": np.asarray(p.matched_target).reshape(-1, 3),
        f"{pre}_stem_rmse": np.array([p.coarse_stem_rmse, p.fine_stem_rmse, p.stem_rmse]),
        f"{pre}_fitness_above": p.fitness_above,
        f"{pre}_used_icp": p.used_icp,
        f"{pre}_trusted": p.trusted,
        f"{pre}_ground_offset": p.ground_offset,
        f"{pre}_quality": np.array([p.fitness, p.rmse, p.n_stem_matches]),
        f"{pre}_summary": p.summary(),
        f"{pre}_has": np.array([p.match is not None, p.reflector_match is not None, p.icp is not None,
                                p.rival is not None]),
    }
    if p.match is not None:
        out.update(_match_out(p.match, f"{pre}_match"))
    if p.rival is not None:
        out.update(_match_out(p.rival, f"{pre}_rival"))
    if p.reflector_match is not None:
        r = p.reflector_match
        out.update({f"{pre}_refl_transform": r.transform, f"{pre}_refl_n": r.n_inliers, f"{pre}_refl_rmse": r.rmse,
                    f"{pre}_refl_corr": np.asarray(r.correspondences).reshape(-1, 2),
                    f"{pre}_refl_success": r.success})
    if p.icp is not None:
        c = p.icp
        out.update({f"{pre}_icp_transform": c.transform, f"{pre}_icp_fitness": c.fitness,
                    f"{pre}_icp_rmse": c.inlier_rmse, f"{pre}_icp_n": c.n_correspondences,
                    f"{pre}_icp_iterations": c.iterations, f"{pre}_icp_converged": c.converged,
                    f"{pre}_icp_history": np.asarray(c.history, float), f"{pre}_icp_repr": repr(c),
                    f"{pre}_icp_has_info": c.information is not None})
        if c.information is not None:
            out.update({f"{pre}_info_hessian": c.information.hessian, f"{pre}_info_sigma": c.information.sigma,
                        f"{pre}_info_n": c.information.n})
    return out


def _survey_out(r, pre, messages=None, strip=None):
    out = {
        f"{pre}_poses": np.array(r.poses),
        f"{pre}_registered": np.array(r.registered),
        f"{pre}_reference": r.reference,
        f"{pre}_edge_to_pair": np.array(r.edge_to_pair, int),
        f"{pre}_names": _strings(r.names),
        f"{pre}_n_pairs": len(r.pairs),
        f"{pre}_report": _untimed(r.report()),
        f"{pre}_rejected": _strings(f"{p.i}-{p.j}" for p in r.rejected_pairs()),
        f"{pre}_transform_for": np.array([r.transform_for(k) for k in range(len(r.scans))]),
    }
    for robust in (True, False):
        c = r.consistency(robust=robust)
        out[f"{pre}_consistency_{robust}"] = np.array([[i, j, v] for (i, j), v in c.items()]).reshape(-1, 3)
    if r.optimisation is not None:
        o = r.optimisation
        out.update({f"{pre}_opt": np.array([o.iterations, o.converged, o.initial_error, o.final_error]),
                    f"{pre}_opt_rejected": np.array(o.rejected_edges, int),
                    f"{pre}_opt_edge_errors": np.asarray(o.edge_errors), f"{pre}_opt_repr": repr(o)})
    for k, p in enumerate(r.pairs):
        out.update(_pair_out(p, f"{pre}_pair{k}"))
    if messages is not None:
        out[f"{pre}_log"] = _strings(_untimed(m) for m in messages)
    with tempfile.TemporaryDirectory() as d:
        path = r.save(Path(d) / "sub" / "transforms.json")
        text = path.read_text()
        payload = json.loads(text)
        out[f"{pre}_saved_seconds_is_float"] = isinstance(payload.pop("seconds"), float)
        out[f"{pre}_saved"] = json.dumps(payload, indent=2)
        out[f"{pre}_saved_layout"] = re.sub(r'"seconds": [^,]*,', '"seconds": ?,', text)
        if strip:
            for key in ("saved", "saved_layout"):
                out[f"{pre}_{key}"] = out[f"{pre}_{key}"].replace(strip, "<dir>")
        loaded = pl.load_transforms(path)
        out[f"{pre}_loaded_names"] = _strings(loaded)
        out[f"{pre}_loaded"] = np.array(list(loaded.values())).reshape(-1, 4, 4)
    return out


# --------------------------------------------------------------------------- per scan


def prepare():
    sc = _scene()
    out = {}
    for k, f in enumerate(_prepared()):
        out.update(_scan_out(f, f"scan{k}"))
    rng = np.random.default_rng(5)
    level = tf.se3_exp(np.array([0.02, -0.03, 0.4, 0.0, 0.0, 0.0]))
    raw = tf.transform_points(tf.invert(level), sc.clouds[1])
    targets = [Reflector(1.0, 2.0, 1.5, 3.0, 0.1, 12, "T1"), Reflector(-4.0, 0.5, 0.8)]
    f = pl.prepare_scan(raw, _cfg(), name="levelled", reflectors=targets, levelling=level, origin=[0.0, 0.0, 0.0])
    out.update(_scan_out(f, "levelled"))
    f = pl.prepare_scan(PointCloud(sc.clouds[2]), _cfg(icp_min_planarity=0.0, ground_min_coverage=None,
                                                         icp_voxel=0.08, icp_max_height=6.0), origin=(0, 0, 0))
    out.update(_scan_out(f, "voxelled"))
    f = pl.prepare_scan(sc.clouds[3].ravel().tolist(), _cfg(ground_cell_size=0.7, ground_min_coverage=0.95),
                        name="list", origin=np.zeros(3))
    out.update(_scan_out(f, "list"))
    out.update(_scan_out(pl.prepare_scan(rng.normal(0, 1, (500, 3)), _cfg()), "few"))
    out.update(_scan_out(pl.prepare_scan(rng.normal(0, 1, (1500, 3)), _cfg(min_points_per_scan=2000,
                                                                          riegl_options={"x": 1})), "filtered"))
    out.update(_scan_out(pl.prepare_scan(np.zeros((0, 3)), _cfg()), "empty"))
    # A scanner tilted on its side: blind azimuths along the tilt axis.
    xy = rng.uniform(-30, 30, size=(150_000, 2))
    ground = np.column_stack([xy, np.zeros(len(xy))])
    crowns = np.column_stack([rng.uniform(-30, 30, size=(60_000, 2)), rng.uniform(5, 8, 60_000)])
    rel = np.vstack([ground, crowns]) - [0.0, 0.0, 1.6]
    azimuth = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    elevation = np.degrees(np.arctan2(rel[:, 2], np.hypot(rel[:, 0], rel[:, 1])))
    along_axis = (np.abs(azimuth) < 30) | (np.abs(azimuth) > 150)
    points = rel[~(along_axis & (elevation > -40) & (elevation < 10))]
    for pre, origin in (("tilted_blind", None), ("tilted_seen", [0.0, 0.0, 0.0])):
        f = pl.prepare_scan(points, _cfg(), name="tilted", origin=origin)
        out[f"{pre}_elevation"] = f.ground.elevation
        out[f"{pre}_observed"] = f.ground.observed
        _put(out, f"{pre}_icp_points", f.icp_points)
    g0 = fit_ground(points, 0.5)
    for cov in (0.8, 0.3, 0.0, 1.1):
        g = pl.scan._refit_visible_ground(points, np.array([0.5, -0.5, 0.2]), g0, _cfg(ground_min_coverage=cov))
        out[f"refit_{cov}_elevation"] = g.elevation
        out[f"refit_{cov}_same"] = g is g0
    return out


def _attributed_cloud(path, rng, n=6000):
    xyz = np.column_stack([rng.uniform(-20, 20, (n, 2)), rng.uniform(-2, 10, n)])
    xyz[:5] = [[0.1, 0.1, 0.1], [0.3, 0.0, 0.0], [0.0, -0.2, 0.1], [1.0, 1.0, 0.0], [0.0, 0.0, 0.2]]
    deviation = rng.integers(0, 30, n).astype(float)
    deviation[::7] = 65535.0
    attrs = {"deviation": deviation, "reflectance": rng.normal(-5, 4, n),
             "amplitude": rng.uniform(0, 40, n)}
    io.write(PointCloud(xyz, attrs), path)
    io.write(PointCloud(xyz, {"reflectance": attrs["reflectance"]}), path.with_name("bare.ply"))
    return xyz


def reading():
    rng = np.random.default_rng(9)
    out = {}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "scan.ply"
        _attributed_cloud(path, rng)
        settings = Path(d) / "export.txt"
        settings.write_text("# export filter\nriegl.deviation, 0, 20\nrange; 0.2; 25\nreflectance, -15, 5\n")
        opts = pl.reading_options(settings)
        out["options_file"] = json.dumps(opts, sort_keys=True)
        out["options_override"] = json.dumps(pl.reading_options(settings, max_range=18.0, min_range=None,
                                                                min_amplitude=3.0), sort_keys=True)
        out["options_none"] = json.dumps(pl.reading_options(None, max_deviation=4), sort_keys=True)
        cases = {
            "plain": _cfg(),
            "gates": _cfg(riegl_options=opts),
            "amplitude": _cfg(riegl_options={"min_amplitude": 10.0, "max_amplitude": 30.0, "library": "x"}),
            "deviation": _cfg(riegl_options={"max_deviation": 12}),
            "range_only_min": _cfg(riegl_options={"min_range": 0.25}),
            "current": _cfg(riscan_filter="current"),
            "legacy": _cfg(riscan_filter="legacy", riegl_options={"max_range": 30.0}),
            "capped": _cfg(max_points_per_scan=1000, riegl_options={"min_reflectance": -8.0}),
        }
        for name, cfg in cases.items():
            xyz, extra = pl.scan._read_scan(path, cfg)
            out[f"read_{name}"] = xyz
            out[f"read_{name}_extra"] = len(extra)
        for name, (file, cfg) in {
            "no_amplitude": ("bare.ply", _cfg(riscan_filter="current")),
            "no_deviation": ("bare.ply", _cfg(riegl_options={"min_deviation": 1.0})),
            "bad_mode": ("scan.ply", _cfg(riscan_filter="bogus")),
            "missing": ("nothing.ply", _cfg()),
        }.items():
            try:
                pl.scan._read_scan(Path(d) / file, cfg)
                out[f"error_{name}"] = ""
            except Exception as exc:  # the type and message are the behaviour
                out[f"error_{name}"] = f"{type(exc).__name__}: {exc}".replace(d, "<dir>")
        f = pl.prepare_scan(path, _cfg(riegl_options={"max_range": 15.0}, min_points_per_scan=10))
        out.update(_scan_out(f, "from_path"))
        out["from_path_source"] = str(f.source).replace(d, "<dir>")
        f = pl.prepare_scan(str(path), _cfg(min_points_per_scan=100_000, riegl_options={"max_range": 15.0}),
                            name="named")
        out.update(_scan_out(f, "too_few"))
        out["too_few_source"] = str(f.source).replace(d, "<dir>")
    return out


# --------------------------------------------------------------------------- helpers


def helpers():
    scans = _prepared()
    cfg = _cfg()
    out = {}
    for k in (0, 3):
        out[f"terrain_samples_{k}"] = pl.pair._terrain_samples(scans[k], 12.0)
    out["terrain_samples_none"] = pl.pair._terrain_samples(dataclasses.replace(scans[0], ground=None), 30.0)
    rel = _relative(1)
    lifted = rel.copy()
    lifted[2, 3] += 0.7
    far = rel.copy()
    far[0, 3] += 500.0
    for name, T, targets in (("pair", rel, [(scans[0], np.eye(4))]), ("lifted", lifted, [(scans[0], np.eye(4))]),
                             ("far", far, [(scans[0], np.eye(4))]),
                             ("world", _relative(1), [(scans[0], np.eye(4)), (scans[2], _relative(2)),
                                                      (dataclasses.replace(scans[3], ground=None), _relative(3))]),
                             ("none", rel, [])):
        for c in (cfg, _cfg(ground_radius=10.0, min_ground_cells=0), _cfg(min_ground_cells=100_000)):
            key = f"height_{name}_{c.ground_radius}_{c.min_ground_cells}"
            out[key] = pl.pair._height_offset(scans[1], T, targets, c)
            out[f"on_ground_{key}"] = pl.pair._on_ground(lifted, scans[1], targets, c)
    out["on_ground_off"] = pl.pair._on_ground(lifted, scans[1], [(scans[0], np.eye(4))], _cfg(height_from_ground=False))
    out["height_no_ground"] = pl.pair._height_offset(dataclasses.replace(scans[1], ground=None), rel,
                                                [(scans[0], np.eye(4))], cfg)
    origin = np.array([0.3, -0.2, 1.6])
    for name, pose in (("same", rel), ("near", tf.se3_exp(np.r_[0.0, 0.0, 0.3, 1.0, 1.0, 0.0]) @ rel),
                       ("far", tf.se3_exp(np.r_[0.0, 0.0, 0.0, 6.0, 0.0, 0.0]) @ rel)):
        for c in (cfg, _cfg(max_prior_rotation=5.0), _cfg(max_prior_shift=0.5, max_prior_rotation=0.1)):
            ok, why = pl.survey._prior_ok(pose, rel, origin, c)
            out[f"prior_{name}_{c.max_prior_shift}_{c.max_prior_rotation}"] = f"{ok}|{why}"
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (2, 3), (1, 3)]
    messages = []
    positions = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [np.nan, 1.0, 0.0], [45.0, 0.0, 5.0]])
    for name, pos, limit in (("gnss", positions, 40.0), ("tight", positions, 10.0), ("none", None, 40.0),
                             ("inf", positions, np.inf), ("flat", np.zeros(4), 1.0), ("empty", np.zeros((0, 3)), 1.0)):
        out[f"reach_{name}"] = np.array(pl.survey._within_reach(pairs, pos, limit, messages.append)).reshape(-1, 2)
    out["reach_log"] = _strings(messages)
    matches = [((i, j), match_stem_maps(scans[i].stem_map, scans[j].stem_map, cfg.matching)) for i, j in pairs]
    for limit in (1, 2, 5):
        out[f"limit_{limit}"] = np.array([ij for ij, _ in pl.survey._limit_per_scan(matches, limit, 4)]).reshape(-1, 2)
    graph = PoseGraph(4)
    graph.poses = [_relative(k) for k in range(4)]
    for mask in ([True, True, True, True], [True, False, True, False], [False] * 4):
        combined = pl.survey._combined_stem_map(list(scans), mask, graph)
        out[f"combined_{''.join('1' if m else '0' for m in mask)}"] = np.array(
            [[s.x, s.y, s.z, s.dbh, s.quality, *s.axis] for s in combined]).reshape(-1, 8)
        out[f"combined_name_{''.join('1' if m else '0' for m in mask)}"] = combined.name
    for k, (T, p) in enumerate(((rel, pl.PairResult(0, 1, matched_source=scans[1].stem_map.positions[:5],
                                                     matched_target=scans[1].stem_map.positions[:5] + 0.1)),
                                (rel, pl.PairResult(0, 1)),
                                (np.eye(4), pl.PairResult(0, 1, matched_source=np.ones((4, 3)),
                                                          matched_target=np.zeros((4, 3)))))):
        out[f"stem_residual_{k}"] = pl.pair._stem_median_residual(T, p)
    return out


# --------------------------------------------------------------------------- pairs


def _fake(positions):
    return [Reflector(float(p[0]), float(p[1]), float(p[2])) for p in positions]


def pairs():
    scans = _prepared()
    sc = _scene()
    cfg = _cfg(use_reflectors=False)
    a, b = scans[0], scans[1]
    out = {}
    runs = {
        "default": (a, b, cfg, {}),
        "reverse": (b, a, cfg, {"i": 1, "j": 0}),
        "initial": (a, b, cfg, {"initial": _relative(0, 1)}),
        "initial_off": (a, b, cfg, {"initial": tf.se3_exp(np.r_[0, 0, 0.05, 0.3, -0.2, 0.1]) @ _relative(0, 1)}),
        "match": (a, scans[2], cfg, {"match": match_stem_maps(a.stem_map, scans[2].stem_map, cfg.matching), "i": 0,
                                     "j": 2}),
        "prepared_target": (scans[3], b, cfg, {"target_icp": ICPTarget(b.icp_points, cfg.icp), "i": 3, "j": 1}),
        "targets": (a, b, _cfg(), {}),
        "targets_12": (scans[1], scans[2], _cfg(), {"i": 1, "j": 2}),
        "stranger": (a, _stranger(), cfg, {}),
        "no_stems": (dataclasses.replace(a, stem_map=StemMap([], name="x")), b, cfg, {}),
        "diverged": (a, b, _cfg(max_coarse_to_fine_shift=0.0, use_reflectors=False), {}),
        "low_fitness": (a, b, _cfg(min_icp_fitness=0.95, use_reflectors=False), {}),
        "low_above": (a, b, _cfg(min_icp_fitness_above_ground=0.95, use_reflectors=False), {}),
        "high_rmse": (a, b, _cfg(max_icp_rmse=0.001, use_reflectors=False), {}),
        "terrain": (a, b, _cfg(max_ground_disagreement=0.0001, use_reflectors=False), {}),
        "terrain_off": (a, b, _cfg(max_ground_disagreement=None, use_reflectors=False), {}),
        "loose": (a, b, _cfg(max_match_rmse=0.001, use_reflectors=False), {}),
        "few_inliers": (a, b, _cfg(min_match_inliers=500, use_reflectors=False), {}),
        "coarse_stems": (a, b, _cfg(max_coarse_stem_rmse=0.0, use_reflectors=False), {}),
        "kept_coarse": (a, b, _cfg(stem_agreement_tolerance=-1.0, use_reflectors=False), {}),
        "kept_coarse_targets": (a, b, _cfg(stem_agreement_tolerance=-1.0), {}),
        "no_above": (a, b, _cfg(fitness_min_height=0.0, use_reflectors=False), {}),
        "high_above": (a, b, _cfg(fitness_min_height=40.0, use_reflectors=False), {}),
        "stem_height": (a, b, _cfg(height_from_ground=False, use_reflectors=False), {}),
        "rival": (a, b, _cfg(max_match_ambiguity=-1.0, use_reflectors=False), {}),
        "rival_margin": (a, b, _cfg(max_match_ambiguity=-1.0, ambiguity_margin=1e9, use_reflectors=False), {}),
        "rival_close": (a, scans[3], _cfg(max_match_ambiguity=-1.0, ambiguity_margin=0.0, use_reflectors=False,
                                          matching=dataclasses.replace(cfg.matching, distinct_translation=0.05)), {"j": 3}),
        "rival_prepared": (a, b, _cfg(max_match_ambiguity=-1.0, use_reflectors=False),
                           {"target_icp": ICPTarget(b.icp_points, cfg.icp)}),
        "ambiguous": (a, b, _cfg(max_match_ambiguity=-1.0, use_reflectors=False,
                                 matching=dataclasses.replace(cfg.matching, distinct_translation=1e6,
                                                              distinct_yaw_deg=1e6)), {}),
    }
    # Targets: right, misleading, trusted when ICP fails, and a trusted pose kept from the targets.
    truth = _relative(0, 1)
    corners = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 1.0], [0.0, 12.0, 0.5], [-9.0, 4.0, 1.2], [5.0, -8.0, 0.3]])
    good_a = dataclasses.replace(a, reflectors=_fake(corners))
    good_b = dataclasses.replace(b, reflectors=_fake(tf.transform_points(truth, corners)))
    bogus = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    runs.update({
        "reflectors": (good_a, good_b, _cfg(), {}),
        "reflectors_bogus": (dataclasses.replace(a, reflectors=_fake(bogus)),
                             dataclasses.replace(b, reflectors=_fake(bogus + [40.0, 40.0, 0.0])), _cfg(), {}),
        "trusted_icp": (good_a, good_b, _cfg(min_icp_fitness=0.99), {}),
        "trusted_targets": (good_a, good_b, _cfg(min_icp_fitness=0.99, reflector_tolerance=1e-6,
                                                 min_reflector_matches=5), {}),
        "untrusted": (good_a, good_b, _cfg(min_icp_fitness=0.99, trusted_reflector_matches=0), {}),
        "trust_rmse": (good_a, good_b, _cfg(min_icp_fitness=0.99, trusted_reflector_rmse=-1.0), {}),
        "reflectors_kept_coarse": (good_a, good_b, _cfg(max_coarse_to_fine_shift=-1.0), {}),
        "reflectors_initial": (good_a, good_b, _cfg(min_icp_fitness=0.99), {"initial": truth}),
    })
    for name, (s, t, c, kw) in runs.items():
        p = pl.register_pair(s, t, c, **kw)
        out.update(_pair_out(p, name))
    return out


def place():
    scans = _prepared()
    out = {}
    poses = [_relative(k) for k in range(4)]
    truth = poses[2]
    prior = tf.se3_exp(np.array([0.0, 0.0, np.radians(1.0), 0.25, -0.20, 0.0])) @ truth
    prior[2, 3] += 2.0
    for name, (scan, survey, ps, pr, cfg, n) in {
        "default": (scans[2], [scans[0], scans[1], scans[3]], [poses[0], poses[1], poses[3]], prior, _cfg(), None),
        "one": (scans[2], [scans[0], scans[1], scans[3]], [poses[0], poses[1], poses[3]], prior, _cfg(), 1),
        "far": (scans[2], scans[:2], poses[:2], tf.se3_exp(np.r_[0, 0, 0, 80.0, 0, 0]) @ prior, _cfg(), None),
        "empty": (scans[2], [], [], prior, _cfg(), None),
        "no_points": (scans[2], [dataclasses.replace(scans[0], icp_points=np.zeros((0, 3), np.float32))],
                      poses[:1], prior, _cfg(), None),
        "shift": (scans[2], scans[:2], poses[:2], prior, _cfg(max_prior_shift=0.01), None),
        "fitness": (scans[2], scans[:2], poses[:2], prior, _cfg(min_icp_fitness=0.99), None),
        "above": (scans[2], scans[:2], poses[:2], prior, _cfg(min_icp_fitness_above_ground=0.99), None),
        "rmse": (scans[2], scans[:2], poses[:2], prior, _cfg(max_icp_rmse=0.001), None),
        "terrain": (scans[2], scans[:2], poses[:2], prior, _cfg(max_ground_disagreement=1e-5), None),
    }.items():
        r, used = pl.place_from_prior(scan, survey, ps, pr, cfg, neighbours=n)
        out.update(_pair_out(r, name))
        out[f"{name}_used"] = np.array(used, int)
    return out


# --------------------------------------------------------------------------- surveys


def _run(scans, cfg, **kw):
    messages = []
    return pl.coregister_prepared(list(scans), cfg, progress=messages.append, **kw), messages


def survey():
    scans = _prepared()
    out = {}
    for name, cfg in {
        "default": _cfg(workers=0),
        "stems_only": _cfg(use_reflectors=False, workers=1),
        "unscreened": _cfg(screen_pairs=False, workers=3, use_reflectors=False),
        "unoptimised": _cfg(optimise_globally=False),
        "limited": _cfg(max_pairs_per_scan=1, reject_outlier_edges=False, use_reflectors=False),
        "reference": _cfg(reference_scan=2, use_reflectors=False),
        "reference_high": _cfg(reference_scan=9, use_reflectors=False),
        "multiview": _cfg(refine_multiview=True, refinement_points_per_scan=20000, use_reflectors=False),
        "multiview_capped": _cfg(refine_multiview=True, refinement_max_shift=-1.0, refinement_rounds=1,
                                 refinement_voxel_sizes=(0.1,), refinement_max_distances=(0.3,),
                                 use_reflectors=False),
        "screened_out": _cfg(max_coarse_stem_rmse=0.0, recover_unregistered=False, use_reflectors=False),
        "screened_targets": _cfg(max_coarse_stem_rmse=0.0, recover_unregistered=False),
    }.items():
        r, messages = _run(scans, cfg)
        out.update(_survey_out(r, name, messages))
    return out


def survey_inputs():
    scans = _prepared()
    out = {}
    stemless = list(scans)
    stemless[2] = dataclasses.replace(stemless[2], stem_map=StemMap([], name=stemless[2].name))
    priors = [_relative(k) for k in range(4)]
    wrong = list(priors)
    wrong[2] = wrong[2].copy()
    wrong[2][:3, 3] += [10.0, 0.0, 0.0]
    off = tf.se3_exp(np.array([0.0, 0.0, np.radians(1.0), 0.25, -0.20, 0.0])) @ priors[2]
    off[2, 3] += 1.5
    unusable = [dataclasses.replace(scans[0], error="broken", icp_points=np.zeros((0, 3), np.float32))] + list(scans[1:])
    runs = {
        "fixed": (scans, _cfg(use_reflectors=False), {"fixed": {0: priors[0], 1: priors[1]}}),
        "fixed_ref": (scans, _cfg(use_reflectors=False, reference_scan=3), {"fixed": {1: priors[1], 2: priors[2]}}),
        "priors_wrong": (scans, _cfg(use_reflectors=False), {"priors": wrong}),
        "priors_rotation": (scans, _cfg(use_reflectors=False, max_prior_rotation=0.001), {"priors": priors}),
        "priors_place": (stemless, _cfg(use_reflectors=False), {"priors": [priors[0], priors[1], off, priors[3]]}),
        "priors_partial": (stemless, _cfg(use_reflectors=False), {"priors": [priors[0], None, off, None]}),
        "priors_refused": (stemless, _cfg(use_reflectors=False, max_prior_shift=0.01),
                           {"priors": [priors[0], priors[1], off, priors[3]]}),
        "pairs": (scans, _cfg(use_reflectors=False), {"pairs": [(0, 1), (1, 2)]}),
        "pairs_norecover": (scans, _cfg(use_reflectors=False, recover_unregistered=False), {"pairs": [(0, 1), (1, 2)]}),
        "reroot": (scans, _cfg(use_reflectors=False), {"pairs": [(1, 2), (2, 3)]}),
        "gnss": (scans, _cfg(use_reflectors=False, max_pair_distance=12.0),
                 {"approximate_positions": np.array([[0, 0, 0], [5, 0, 0], [500, 500, 0], [np.nan] * 3])}),
        "gnss_norecover": (scans, _cfg(use_reflectors=False, max_pair_distance=12.0, recover_unregistered=False),
                           {"approximate_positions": np.array([[0, 0, 0], [5, 0, 0], [500, 500, 0], [9, 0, 0]])}),
        "stranger": (list(scans[:3]) + [_stranger()], _cfg(use_reflectors=False), {}),
        "stranger_one_round": (list(scans[:3]) + [_stranger()], _cfg(use_reflectors=False, recovery_rounds=0,
                                                                       recovery_neighbours=0), {"pairs": [(0, 1)]}),
        "unusable": (unusable, _cfg(use_reflectors=False), {}),
        "tie_in_pairwise": (scans, _cfg(use_reflectors=False, min_icp_fitness=0.6),
                            {"fixed": {k: priors[k] for k in range(3)}, "pairs": []}),
        "tie_in_combined": (scans, _cfg(use_reflectors=False, min_icp_fitness=0.7),
                            {"fixed": {k: priors[k] for k in range(3)}, "pairs": []}),
        "too_few_stems": (scans, _cfg(use_reflectors=False, min_match_inliers=100),
                          {"fixed": {k: priors[k] for k in range(3)}, "pairs": []}),
        "single": (scans[:1], _cfg(), {}),
    }
    for name, (s, cfg, kw) in runs.items():
        r, messages = _run(s, cfg, **kw)
        out.update(_survey_out(r, name, messages))
    return out


@functools.lru_cache(maxsize=None)
def _sparse():
    sc = cs.survey(2, [(-12, -10), (0, -10), (12, -10), (-6, 2), (6, 2), (0, 12)], max_range=13.0)
    return tuple(pl.prepare_scan(c, _cfg(), name=f"s{k}") for k, c in enumerate(sc.clouds))


def sparse():
    """A survey whose scans barely reach each other: rejected edges and recovery."""
    out = {}
    for name, cfg in (("default", _cfg()), ("kept", _cfg(reject_outlier_edges=False, workers=0)),
                      ("unscreened", _cfg(screen_pairs=False, max_pairs_per_scan=2, recovery_rounds=3))):
        r, messages = _run(_sparse(), cfg)
        out.update(_survey_out(r, name, messages))
    return out


def coregister():
    sc = _scene()
    out = {}
    tilt = tf.se3_exp(np.array([np.radians(80.0), np.radians(20.0), 0.0, 0.0, 0.0, 0.0]))
    clouds = list(sc.clouds[:3]) + [tf.transform_points(tilt, sc.clouds[3])]
    targets = [[Reflector(*p) for p in t] for t in sc.targets]
    targets[3] = [Reflector(*p) for p in tf.transform_points(tilt, sc.targets[3])]
    messages = []
    r = pl.coregister([PointCloud(c) for c in clouds], _cfg(), names=["a", "", "c", "d"], reflectors=targets,
                      levelling=[None, None, None, tf.invert(tilt)], progress=messages.append)
    out.update(_survey_out(r, "levelled", messages))
    for k, f in enumerate(r.scans):
        out.update(_scan_out(f, f"levelled_scan{k}"))
    messages = []
    r = pl.coregister(clouds, _cfg(use_reflectors=False), progress=messages.append)
    out.update(_survey_out(r, "blind", messages))
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k, c in enumerate(sc.clouds[:3]):
            paths.append(Path(d) / f"s{k}.laz")
            io.write(PointCloud(c), paths[-1])
        empty = Path(d) / "aborted.laz"
        io.write(PointCloud(np.zeros((0, 3))), empty)
        broken = Path(d) / "corrupt.laz"
        broken.write_bytes(b"LASF garbage that is not a point cloud")
        for name, files, cfg in (("files", [empty] + paths + [broken], _cfg(workers=2)),
                                 ("files_one", [str(p) for p in paths], _cfg(workers=1, use_reflectors=False))):
            messages = []
            r = pl.coregister(files, cfg, progress=messages.append)
            out.update(_survey_out(r, name, [m.replace(d, "<dir>") for m in messages], strip=d))
            out[f"{name}_errors"] = _strings(s.error.replace(d, "<dir>") for s in r.scans)
            out[f"{name}_sources"] = _strings(str(s.source).replace(d, "<dir>") for s in r.scans)
            merged = pl.merge_clouds(files, r, voxel=0.1)
            _put(out, f"{name}_merged", merged.xyz, 31)
            _put(out, f"{name}_merged_ids", merged.attrs["scan_id"], 31)
    return out


def merge():
    sc = _scene()
    scans = _prepared()
    r, _ = _run(scans, _cfg(use_reflectors=False), pairs=[(0, 1), (1, 2)])
    r.registered[3] = False
    out = {}
    for name, kw in (("default", {}), ("coarse", {"voxel": 0.3}), ("raw", {"voxel": None}),
                     ("all", {"voxel": 0.25, "only_registered": False})):
        clouds = [c[::4] for c in sc.clouds]
        m = pl.merge_clouds([PointCloud(c) for c in clouds[:2]] + clouds[2:], r, **kw)
        _put(out, f"{name}_xyz", m.xyz, 11)
        _put(out, f"{name}_ids", m.attrs["scan_id"], 11)
    r.registered = [False] * 4
    m = pl.merge_clouds(sc.clouds, r)
    out["nothing_xyz"] = m.xyz
    out["nothing_ids"] = m.attrs["scan_id"]
    return out


def results():
    """Result objects built by hand: summaries, reports and saved files of odd values."""
    out = {}
    icp = ICPResult(np.eye(4), 0.1234, 0.0456, 321, 7, True, [1.0, 0.5])
    names = ["a", "b", "été", "d"]
    scans = [pl.ScanFeatures(n, 1234567 * (k + 1), None, StemMap([]), np.zeros((0, 3), np.float32),
                             source=Path(f"/data/{n}.rxp") if k % 2 else None, error="bad" if k == 3 else "")
             for k, n in enumerate(names)]
    m = match_stem_maps(_prepared()[0].stem_map, _prepared()[1].stem_map)
    pairs = [
        pl.PairResult(0, 1, "a", "b", transform=_relative(1), match=m, icp=icp, success=True, reason="fine",
                      matched_source=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 1.0, 0.5]]),
                      matched_target=np.array([[0.05, 0.0, 0.0], [1.0, 2.1, 0.0], [3.0, 1.0, 0.7]]),
                      coarse_stem_rmse=0.0125, fine_stem_rmse=0.0375, fitness_above=0.3333, ground_offset=-0.125),
        pl.PairResult(1, 2, "b", "été", reason="nothing", coarse_stem_rmse=np.inf, used_icp=False),
        pl.PairResult(0, 2, "a", "été", success=True, icp=icp, used_icp=False, trusted=True,
                      reflector_match=ReflectorMatch(np.eye(4), 5, 0.002, np.zeros((5, 2), int), True),
                      matched_source=np.ones((2, 3)), matched_target=np.ones((2, 3)) * 2, reason="targets"),
        pl.PairResult(2, 1, "été", "b", success=True, icp=icp, matched_source=np.ones((1, 3)),
                      matched_target=np.full((1, 3), 3.0), coarse_stem_rmse=0.5, fine_stem_rmse=np.nan),
    ]
    opt = OptimisationResult([np.eye(4)] * 4, 12, False, 123456.789, 0.000123456, [1, 5, 0], np.array([1.0]))
    poses = [np.eye(4), _relative(1), _relative(2), tf.se3_exp(np.r_[0.1, 0.2, 0.3, 1.0, 2.0, 3.0])]
    for name, kw in (("full", {"optimisation": opt, "edge_to_pair": [0, 2, 3], "seconds": 12.345}),
                     ("plain", {"edge_to_pair": [], "seconds": 0.0}),
                     ("clean", {"optimisation": OptimisationResult(poses, 3, True, 1e-20, 12345678.0),
                                "edge_to_pair": [0, 2, 3], "seconds": 1.0})):
        r = pl.SurveyResult(scans, pairs, poses, 1, registered=[True, True, False, False], **kw)
        out.update(_survey_out(r, name))
        out[f"{name}_timed_report"] = r.report()
        with tempfile.TemporaryDirectory() as d:
            out[f"{name}_timed_saved"] = r.save(Path(d) / "t.json").read_text()
    for k, p in enumerate(pairs):
        out[f"summary_{k}"] = p.summary()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.json"
        path.write_text(json.dumps({"scans": [{"name": "x", "registered": 1, "world_from_scan": np.eye(4).tolist()},
                                              {"name": "y", "registered": False, "world_from_scan": []},
                                              {"name": "x", "registered": True,
                                               "world_from_scan": (2 * np.eye(4)).tolist()},
                                              {"name": "z", "world_from_scan": []}]}))
        loaded = pl.load_transforms(path)
        out["loaded_names"] = _strings(loaded)
        out["loaded"] = np.array(list(loaded.values()))
    out["config"] = json.dumps(pl.CoregConfig().to_dict(), sort_keys=True)
    return out


CASES = {
    "prepare": prepare,
    "reading": reading,
    "helpers": helpers,
    "pairs": pairs,
    "place": place,
    "survey": survey,
    "survey_inputs": survey_inputs,
    "sparse": sparse,
    "coregister": coregister,
    "merge": merge,
    "results": results,
}
