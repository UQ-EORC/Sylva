import numpy as np
import pytest
from conftest import make_stem

from sylva import PointCloud, coreg


def _stand(seed=0, n=45, size=60.0):
    """Stems of varied diameter on a gently sloping ground, with a little low noise."""
    rng = np.random.default_rng(seed)
    xy = []
    while len(xy) < n:
        p = rng.uniform(0, size, 2)
        if all(np.hypot(*(p - q)) > 3.0 for q in xy):
            xy.append(p)
    parts = []
    for x, y in xy:
        r = rng.uniform(0.08, 0.3)
        z0 = 0.03 * x
        parts.append(make_stem(rng, x, y, r, 6.0, z0=z0, density=1500, noise=0.004))
    g = rng.uniform(0, size, (120_000, 2))
    parts.append(np.column_stack([g, 0.03 * g[:, 0] + rng.normal(0, 0.01, len(g))]))
    return np.vstack(parts)


def _scan(world, centre, radius, T):
    """Points within ``radius`` of ``centre``, in a frame whose world_from_scan is ``T``."""
    keep = np.hypot(world[:, 0] - centre[0], world[:, 1] - centre[1]) < radius
    return PointCloud(coreg._transform(coreg.invert(T), world[keep]))


def _pose(yaw_deg, t):
    T = np.eye(4)
    a = np.radians(yaw_deg)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = t
    return T


@pytest.fixture(scope="module")
def survey():
    world = _stand()
    truth = [_pose(0, [0, 0, 0]), _pose(35, [22, 4, 1.5]), _pose(-80, [12, 24, -0.8])]
    centres = [(22, 22), (38, 26), (28, 40)]
    scans = [
        coreg.prepare_scan(_scan(world, c, 22.0, T), name=f"s{k}")
        for k, (c, T) in enumerate(zip(centres, truth, strict=True))
    ]
    return scans, truth


def _err(A, B):
    d = coreg.invert(A) @ B
    return float(np.linalg.norm(d[:3, 3])), float(np.degrees(np.linalg.norm(coreg.se3_log(d)[:3])))


def test_se3_round_trip():
    xi = np.array([0.1, -0.2, 0.3, 1.0, 2.0, -3.0])
    np.testing.assert_allclose(coreg.se3_log(coreg.se3_exp(xi)), xi, atol=1e-9)


def test_register_pair_recovers_a_known_transform(survey):
    scans, truth = survey
    assert all(s.usable for s in scans)
    r = coreg.register_pair(scans[1], scans[0], i=1, j=0)
    assert r.success, r.reason
    want = coreg.invert(truth[0]) @ truth[1]  # scan-0 frame from scan-1 frame
    dt, dr = _err(r.transform, want)
    assert dt < 0.03 and dr < 0.2, (dt, dr)
    assert r.n_stems >= 5 and r.fitness > 0.3


def test_register_scans_and_place_against_fixed(survey):
    scans, truth = survey
    res = coreg.register_scans(scans, log=None)
    assert res.registered.all()
    for k in (1, 2):
        dt, dr = _err(res.poses[k], truth[k])
        assert dt < 0.03 and dr < 0.2, (k, dt, dr)
    # Scans 0 and 1 trusted, scan 2 placed into their frame.
    r, used = coreg.place_scan(scans[2], scans[:2], truth[:2])
    assert r.success, r.reason
    dt, dr = _err(r.transform, truth[2])
    assert dt < 0.03 and dr < 0.2 and set(used) == {0, 1}


def test_pose_graph_spreads_loop_closure_error():
    truth = [_pose(0, [0, 0, 0]), _pose(10, [10, 0, 0]), _pose(20, [10, 10, 0])]
    g = coreg.PoseGraph(3)
    rng = np.random.default_rng(1)
    for i, j in [(1, 0), (2, 1), (2, 0)]:
        rel = coreg.invert(truth[j]) @ truth[i]
        noisy = rel @ coreg.se3_exp(np.r_[rng.normal(0, 1e-3, 3), rng.normal(0, 0.02, 3)])
        g.add_edge(i, j, noisy, rmse=0.02, fitness=0.5, n_correspondences=1000)
    assert g.optimise() == []
    for k in (1, 2):
        assert _err(g.poses[k], truth[k])[0] < 0.05
