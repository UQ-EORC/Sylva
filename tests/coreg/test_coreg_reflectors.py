# Tests of reflector matching.
import json

import numpy as np
import pytest

from sylva.coreg import (
    Reflector,
    detect_reflectors,
    match_reflectors,
    read_reflector_list,
    read_tiepoint_list,
    se3_exp,
    transform_difference,
    transform_points,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def _targets(positions) -> list[Reflector]:
    return [Reflector(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in positions]


def test_match_recovers_a_full_6dof_transform(rng):
    """Three targets fix all six degrees of freedom - stems only manage four."""
    positions = rng.uniform(-20, 20, (7, 3))
    T = se3_exp(np.array([0.05, -0.03, 1.2, 4.0, -2.0, 0.5]))
    match = match_reflectors(_targets(positions), _targets(transform_points(T, positions)))
    assert match.success and match.n_inliers == 7
    rotation, translation = transform_difference(match.transform, T)
    assert np.degrees(rotation) < 1e-6 and translation < 1e-6
    assert match.rmse < 1e-9


def test_match_with_partial_overlap(rng):
    """Only some targets are visible from both positions."""
    shared = rng.uniform(-20, 20, (4, 3))
    T = se3_exp(np.array([0.0, 0.0, 0.7, 3.0, 1.0, 0.0]))
    source = _targets(np.vstack([shared, rng.uniform(-20, 20, (3, 3))]))
    target = _targets(np.vstack([transform_points(T, shared), rng.uniform(-20, 20, (3, 3))]))
    match = match_reflectors(source, target)
    assert match.success and match.n_inliers >= 4
    assert transform_difference(match.transform, T)[1] < 0.01


def test_too_few_targets_is_a_clean_failure(rng):
    two = _targets(rng.uniform(-10, 10, (2, 3)))
    assert not match_reflectors(two, two).success
    assert not match_reflectors([], []).success


def test_unrelated_sets_do_not_match(rng):
    a = _targets(rng.uniform(-30, 30, (6, 3)))
    b = _targets(rng.uniform(-30, 30, (6, 3)))
    assert not match_reflectors(a, b).success


def test_collinear_targets_are_rejected():
    """Three targets in a line fix no rotation about that line."""
    line = _targets([[0, 0, 0], [5, 0, 0], [10, 0, 0]])
    assert not match_reflectors(line, line).success


def test_detect_reflectors_from_reflectance(rng):
    """Targets stand out because they return far more energy than the scene."""
    scene = rng.uniform(-10, 10, (5000, 3))
    scene_reflectance = rng.normal(-12, 4, 5000)
    blobs, values = [], []
    for centre in ([3.0, 1.0, 2.0], [-4.0, 5.0, 1.0]):
        blobs.append(rng.normal(centre, 0.02, (40, 3)))
        values.append(rng.normal(20, 1.0, 40))
    xyz = np.vstack([scene] + blobs)
    reflectance = np.concatenate([scene_reflectance] + values)

    found = detect_reflectors(xyz, reflectance, min_reflectance=5.0)
    assert len(found) == 2
    centres = sorted(f.position.tolist() for f in found)
    assert np.allclose(centres[0], [-4.0, 5.0, 1.0], atol=0.05)
    assert np.allclose(centres[1], [3.0, 1.0, 2.0], atol=0.05)


def test_detect_needs_reflectance(rng):
    assert detect_reflectors(rng.normal(size=(100, 3)), None) == []


def test_read_tiepoint_list(tmp_path):
    path = tmp_path / "scan.tpl"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "TP00",
                    "reflectance": 27.3,
                    "diameter": 0.057,
                    "pointcount": 415,
                    "positionCartesian": {"x": 4.52, "y": -1.08, "z": -1.28},
                },
                {"name": "broken"},  # no position: skipped, not fatal
            ]
        )
    )
    targets = read_tiepoint_list(path)
    assert len(targets) == 1
    assert targets[0].name == "TP00"
    assert np.allclose(targets[0].position, [4.52, -1.08, -1.28])


def test_read_tiepoint_list_tolerates_rubbish(tmp_path):
    """An empty or unreadable list means no targets, which is normal."""
    (tmp_path / "empty.tpl").write_text("")
    assert read_tiepoint_list(tmp_path / "empty.tpl") == []
    assert read_tiepoint_list(tmp_path / "missing.tpl") == []


def test_read_riscan_reflector_list(tmp_path):
    """RiSCAN PRO's .rfl: the columns are named by ReflectorIdx."""
    path = tmp_path / "ScanPos002.rfl"
    path.write_text(
        "RieglRflID=1.1\n"
        "ReflectorIdx=name,index,status,x,y,z,r,theta,phi,reflectance,diameter,points,linkname\n"
        "Reflector0=ScanPos002/a.rxp,0,5,18.406487,1.751866,-1.807602,18.58,95.58,5.44,14.69,0.0997,19060,$Nolink\n"
        "Reflector1=ScanPos002/b.rxp,1,5,23.672012,2.619638,-2.476889,23.94,95.94,6.31,18.98,0.0840,14397,$Nolink\n"
        "Reflector2=broken\n"
    )
    found = read_reflector_list(path)
    assert len(found) == 2
    assert np.allclose(found[0].position, [18.406487, 1.751866, -1.807602])
    assert np.isclose(found[1].reflectance, 18.98) and found[1].n_points == 14397
    assert read_reflector_list(tmp_path / "missing.rfl") == []
