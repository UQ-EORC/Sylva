"""Parity cases for sylva.coreg.reflectors."""

import json
import tempfile
from pathlib import Path

import numpy as np

from sylva.coreg import reflectors as rf
from sylva.coreg import transforms as tf


def _flat(targets, prefix):
    return {
        f"{prefix}_xyz": np.array([[t.x, t.y, t.z] for t in targets]).reshape(-1, 3),
        f"{prefix}_reflectance": np.array([t.reflectance for t in targets], float),
        f"{prefix}_diameter": np.array([t.diameter for t in targets], float),
        f"{prefix}_n_points": np.array([t.n_points for t in targets], int),
        f"{prefix}_name": np.array([t.name for t in targets], dtype="U64"),
    }


def _match(m, prefix):
    return {
        f"{prefix}_transform": m.transform,
        f"{prefix}_n_inliers": m.n_inliers,
        f"{prefix}_rmse": m.rmse,
        f"{prefix}_correspondences": np.asarray(m.correspondences).reshape(-1, 2),
        f"{prefix}_success": m.success,
    }


def _targets(positions):
    return [rf.Reflector(float(p[0]), float(p[1]), float(p[2])) for p in positions]


def detect():
    """Bright clusters in a dim cloud: targets, one too large, one too sparse."""
    rng = np.random.default_rng(10)
    scene = rng.uniform(-20, 20, (20000, 3))
    refl = rng.normal(-10, 3, len(scene))
    parts, values = [scene], [refl]
    for c in rng.uniform(-15, 15, (6, 3)):
        blob = c + rng.normal(0, 0.03, (40, 3))
        parts.append(blob)
        values.append(rng.uniform(6, 20, 40))
    parts.append(np.array([2.0, 2.0, 2.0]) + rng.uniform(-0.6, 0.6, (400, 3)))  # a bright sign
    values.append(rng.uniform(6, 9, 400))
    parts.append(np.array([-5.0, 5.0, 1.0]) + rng.normal(0, 0.02, (4, 3)))  # too few returns
    values.append(np.full(4, 12.0))
    xyz, reflectance = np.vstack(parts), np.concatenate(values)
    out = _flat(rf.detect_reflectors(xyz, reflectance), "default")
    out.update(_flat(rf.detect_reflectors(xyz, reflectance, min_reflectance=8.0, cluster_radius=0.1,
                                          min_points=5, max_extent=2.0), "tuned"))
    out["none"] = len(rf.detect_reflectors(xyz, None))
    out["dim"] = len(rf.detect_reflectors(xyz, reflectance - 100))
    return out


def match():
    rng = np.random.default_rng(11)
    out = {}
    positions = rng.uniform(-20, 20, (7, 3))
    T = tf.se3_exp(np.array([0.05, -0.03, 1.2, 4.0, -3.0, 0.5]))
    moved = tf.transform_points(tf.invert(T), positions) + rng.normal(0, 0.002, (7, 3))
    out.update(_match(rf.match_reflectors(_targets(moved), _targets(positions)), "full"))
    # Partial overlap, with extra targets on both sides.
    extra_src = rng.uniform(-20, 20, (3, 3))
    extra_dst = rng.uniform(-20, 20, (2, 3))
    src = _targets(np.vstack([moved[:5], extra_src]))
    dst = _targets(np.vstack([extra_dst, positions[1:]]))
    out.update(_match(rf.match_reflectors(src, dst), "partial"))
    out.update(_match(rf.match_reflectors(src, dst, tolerance=0.01, min_inliers=4,
                                          distance_tolerance=0.01), "tight"))
    out.update(_match(rf.match_reflectors(_targets(moved[:2]), _targets(positions)), "few"))
    out.update(_match(rf.match_reflectors(_targets(rng.uniform(-20, 20, (6, 3))),
                                          _targets(rng.uniform(-20, 20, (6, 3)))), "unrelated"))
    line = np.column_stack([np.linspace(0, 10, 5), np.zeros(5), np.zeros(5)])
    out.update(_match(rf.match_reflectors(_targets(line), _targets(line + 1.0)), "collinear"))
    # A regular lattice: several triangles fit, the best must win the same way.
    grid = np.array([[x, y, 0.1 * x] for x in range(3) for y in range(3)], float) * 5.0
    out.update(_match(rf.match_reflectors(_targets(grid[:7]), _targets(grid[2:] + 0.5)), "lattice"))
    return out


def read():
    out = {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tpl = d / "scan.tpl"
        tpl.write_text(json.dumps([
            {"name": "TP00", "reflectance": 27.3, "diameter": 0.057, "pointcount": 415,
             "positionCartesian": {"x": 4.52, "y": -1.08, "z": -1.28}},
            {"name": "broken"},
            {"name": "strings", "positionCartesian": {"x": "1.5", "y": " -2 ", "z": 3}},
            {"positionCartesian": {"x": 1.0, "y": 2.0, "z": 3.0}},
            {"name": 7, "pointcount": 12.9, "reflectance": True,
             "positionCartesian": {"x": 1e3, "y": -0.0, "z": 2.5e-7}},
            {"name": "bad", "positionCartesian": {"x": "one", "y": 2.0, "z": 3.0}},
            {"name": "nulls", "positionCartesian": None},
            "not a dict",
            {"name": "list", "positionCartesian": [1, 2, 3]},
        ]))
        out.update(_flat(rf.read_tiepoint_list(tpl), "tpl"))
        (d / "empty.tpl").write_text("")
        (d / "object.tpl").write_text('{"x": 1}')
        (d / "bad.tpl").write_text('[{"positionCartesian": {"x": 1, "y": 2, "z": 3}},')
        out["tpl_empty"] = len(rf.read_tiepoint_list(d / "empty.tpl"))
        out["tpl_object"] = len(rf.read_tiepoint_list(d / "object.tpl"))
        out["tpl_bad"] = len(rf.read_tiepoint_list(d / "bad.tpl"))
        out["tpl_missing"] = len(rf.read_tiepoint_list(d / "missing.tpl"))
        rfl = d / "ScanPos002.rfl"
        rfl.write_text(
            "RieglRflID=1.1\n"
            "Reflector0=before,columns,are,known\n"
            "ReflectorIdx=name,index,status,x,y,z,r,theta,phi,reflectance,diameter,points,linkname\n"
            "Reflector0=ScanPos002/a.rxp,0,5,18.406487,1.751866,-1.807602,18.58,95.58,5.44,14.69,0.0997,19060,$Nolink\r\n"
            "Reflector1=ScanPos002/b.rxp,1,5,23.672012,2.619638,-2.476889,23.94,95.94,6.31,18.98,0.0840,14397,$Nolink\n"
            "Reflector2=broken\n"
            "  Reflector3 = c , 2, 5, 1.0, 2.0, 3.0, 0, 0, 0, bright, , 12.7\n"
            "Reflector4=d,3,5,1,2\n"
            "Other=1,2,3\n"
            "Reflector5=e,4,5,-1e2,2E1,.5,0,0,0,1,2\n"
        )
        out.update(_flat(rf.read_reflector_list(rfl), "rfl"))
        out["rfl_missing"] = len(rf.read_reflector_list(d / "missing.rfl"))
    return out


CASES = {"detect": detect, "match": match, "read": read}
