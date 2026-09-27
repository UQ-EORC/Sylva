"""Write inputs and Python outputs for the R coregistration pipeline tests.

    python tests/parity/export_r_coreg_pipeline.py

r/sylva/tests/testthat/test-coreg-pipeline-python.R reads the same scans
from these files, runs the R API and requires the Python results to 1e-9.
Coordinates are multiples of 2^-12, written in full, so that both languages
read exactly the same numbers. Indices are written as Python gives them
(from 0).
"""

import dataclasses
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity import coreg_scenes as cs  # noqa: E402
from parity.export_r_fixtures import OUT, num, r_value  # noqa: E402
from sylva.coreg import pipeline as pl  # noqa: E402
from sylva.coreg import transforms as tf  # noqa: E402
from sylva.coreg.reflectors import Reflector  # noqa: E402
from sylva.coreg.stems import StemMap  # noqa: E402

D = OUT / "coreg_pipeline"
TIMING = re.compile(r"\(\d+\.\d s\)|in \d+\.\d s")


def value(v):
    """R source for nested outputs: strings, lists, integers and empty arrays too."""
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "list(" + ", ".join(value(x) for x in v) + ")"
    if isinstance(v, dict):
        return "list(" + ", ".join(f"{k} = {value(x)}" for k, x in v.items()) + ")"
    if isinstance(v, np.ndarray) and v.dtype.kind in "US":
        return "c(" + ", ".join(json.dumps(str(x)) for x in v) + ")" if v.size else "character()"
    if isinstance(v, np.ndarray) and v.size == 0:
        return f"matrix(0, 0, {v.shape[1]})" if v.ndim == 2 else "double()"
    if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
        return num(v)
    return r_value(v)


def exact_csv(name, a):
    """Rows of multiples of 2^-12, each written exactly."""
    lines = [",".join(np.format_float_positional(v, trim="-") for v in row) for row in np.asarray(a)]
    (D / name).write_text("\n".join(lines) + "\n")


def config(**kw):
    return dataclasses.replace(pl.CoregConfig(verbose=False, workers=2, min_points_per_scan=500), **kw)


def scan_out(f):
    return {
        "name": f.name, "n_points": f.n_points, "error": f.error,
        "stems": np.array([[s.x, s.y, s.z, s.dbh, s.quality] for s in f.stem_map]).reshape(-1, 5),
        "n_icp": len(f.icp_points), "icp_head": np.asarray(f.icp_points[:20], float),
        "icp_sum": np.asarray(f.icp_points, float).sum(axis=0), "heights_head": np.asarray(f.icp_heights[:20], float),
        "heights_sum": float(np.asarray(f.icp_heights, float).sum()),
        "elevation": f.ground.elevation, "ground_origin": f.ground.origin, "observed": f.ground.observed.astype(float),
        "reflectors": np.array([[r.x, r.y, r.z] for r in f.reflectors]).reshape(-1, 3),
        "levelling": f.levelling, "origin": f.origin,
    }


def pair_out(p):
    return {
        "ij": np.array([p.i, p.j], float), "success": p.success, "reason": p.reason, "summary": p.summary(),
        "transform": p.transform, "coarse": p.coarse_transform, "fitness": p.fitness, "rmse": p.rmse,
        "fitness_above": p.fitness_above, "ground_offset": p.ground_offset, "used_icp": p.used_icp,
        "trusted": p.trusted, "n_stem_matches": p.n_stem_matches,
        "stem_rmse": np.array([p.coarse_stem_rmse, p.fine_stem_rmse]),
        "matched_source": np.asarray(p.matched_source, float).reshape(-1, 3),
        "icp_n": p.icp.n_correspondences if p.icp else 0,
    }


def survey_out(r, messages):
    with tempfile.TemporaryDirectory() as d:
        text = r.save(Path(d) / "t.json").read_text()
    c = r.consistency()
    return {
        "poses": np.array(r.poses), "registered": np.array(r.registered), "reference": r.reference,
        "edge_to_pair": np.array(r.edge_to_pair, float), "summaries": np.array([p.summary() for p in r.pairs]),
        "report": TIMING.sub("", r.report()), "saved": re.sub(r'"seconds": [^,]*,', '"seconds": ?,', text),
        "consistency": np.array([[i, j, v] for (i, j), v in c.items()]).reshape(-1, 3),
        "log": np.array([TIMING.sub("", m) for m in messages]),
        "optimisation": np.array([r.optimisation.iterations, r.optimisation.initial_error,
                                  r.optimisation.final_error]) if r.optimisation else np.zeros(0),
    }


def main():
    if D.exists():
        shutil.rmtree(D)
    D.mkdir(parents=True)
    stand = cs.stand(5, size=30.0, n_trees=22, spacing=2.6)
    targets = [(-3, -1, 1.2), (4, 1, 0.8), (1, 4, 1.5), (-5, 3, 1.0), (6, 3, 1.1)]
    sc = cs.survey(5, [(-4, -3), (4, -2), (0, 5)], the_stand=stand, targets=targets, max_range=14.0, ground=5000,
                   under=600, stem_density=3500.0, crown=60)
    clouds = [np.round(c * 4096) / 4096 for c in sc.clouds]
    refl = [np.round(t * 4096) / 4096 for t in sc.targets]
    for k, (c, t) in enumerate(zip(clouds, refl, strict=True)):
        exact_csv(f"scan{k}.csv", c)
        exact_csv(f"targets{k}.csv", t)
    level = np.round(tf.se3_exp(np.array([0.02, -0.03, 0.4, 0.0, 0.0, 0.0])) * 4096) / 4096
    exact_csv("levelling.csv", level)
    cfg = config()
    expected = {}
    scans = [pl.prepare_scan(c, cfg, name=f"s{k}", reflectors=[Reflector(*p) for p in refl[k]])
             for k, c in enumerate(clouds)]
    expected["scans"] = [scan_out(s) for s in scans]
    expected["levelled"] = scan_out(pl.prepare_scan(clouds[1], cfg, name="lev", levelling=level, origin=[0, 0, 0]))
    expected["few"] = pl.prepare_scan(clouds[0][:300], cfg).error
    stems_only = config(use_reflectors=False)
    truth = tf.invert(sc.truth[1]) @ sc.truth[0]
    expected["truth01"] = truth
    expected["pairs"] = {
        "default": pair_out(pl.register_pair(scans[0], scans[1], stems_only)),
        "reverse": pair_out(pl.register_pair(scans[2], scans[0], stems_only, i=2, j=0)),
        "initial": pair_out(pl.register_pair(scans[0], scans[1], stems_only, initial=truth)),
        "targets": pair_out(pl.register_pair(scans[0], scans[1], cfg)),
        "trusted": pair_out(pl.register_pair(scans[0], scans[1], config(min_icp_fitness=0.99,
                                                                        trusted_reflector_matches=3))),
        "rival": pair_out(pl.register_pair(scans[0], scans[2], config(max_match_ambiguity=-1.0,
                                                                      use_reflectors=False), j=2)),
    }
    surveys = {}
    runs = {
        "default": (scans, stems_only, {}),
        "targets": (scans, cfg, {}),
        "fixed": (scans, stems_only, {"fixed": {0: tf.invert(sc.truth[0]) @ sc.truth[0],
                                                1: tf.invert(sc.truth[0]) @ sc.truth[1]}}),
        "pairs": (scans, config(use_reflectors=False, workers=1), {"pairs": [(0, 1)]}),
    }
    stemless = list(scans)
    stemless[2] = dataclasses.replace(scans[2], stem_map=StemMap([], name="s2"))
    priors = [tf.invert(sc.truth[0]) @ sc.truth[k] for k in range(3)]
    off = tf.se3_exp(np.array([0.0, 0.0, 0.01, 0.2, -0.1, 0.0])) @ priors[2]
    off[2, 3] += 0.8
    runs["priors"] = (stemless, stems_only, {"priors": [priors[0], priors[1], off]})
    for name, (s, c, kw) in runs.items():
        messages = []
        r = pl.coregister_prepared(s, c, progress=messages.append, **kw)
        surveys[name] = survey_out(r, messages)
    expected["surveys"] = surveys
    expected["priors"] = {"prior": off, "true": priors}
    r, used = pl.place_from_prior(scans[2], scans[:2], priors[:2], off, stems_only)
    expected["placed"] = {"pair": pair_out(r), "used": np.array(used, float)}
    messages = []
    r = pl.coregister([c for c in clouds], config(use_reflectors=False), names=["a", "b", ""],
                      progress=messages.append)
    expected["coregister"] = survey_out(r, messages)
    merged = pl.merge_clouds(clouds, r, voxel=0.1)
    expected["merged"] = {"n": len(merged), "sum": merged.xyz.sum(axis=0), "head": merged.xyz[:20],
                          "ids": np.bincount(merged.attrs["scan_id"]).astype(float)}
    (D / "expected.R").write_text("expected <- " + value(expected) + "\n")
    print(f"wrote {D}")


if __name__ == "__main__":
    main()
