# Tests of the pose graph.
import numpy as np
import pytest

from sylva.coreg import (
    PoseGraph,
    default_information,
    invert,
    plane_edge_information,
    se3_exp,
    se3_log,
    transform_difference,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def _chain(n, rng, spacing=8.0):
    poses = [np.eye(4)]
    for _ in range(n - 1):
        poses.append(se3_exp(np.concatenate([rng.normal(0, 0.05, 3), rng.normal(0, spacing, 3)])))
    return poses


def _fill(graph, poses, rng, noise_r=0.002, noise_t=0.01, complete=True):
    n = len(poses)
    for i in range(n):
        for j in range(i + 1, n):
            if not complete and j - i > 1:
                continue
            noise = se3_exp(np.concatenate([rng.normal(0, noise_r, 3), rng.normal(0, noise_t, 3)]))
            graph.add_edge(
                i,
                j,
                noise @ invert(poses[j]) @ poses[i],
                rmse=0.01,
                fitness=0.5,
                n_correspondences=1000,
            )


def test_recovers_poses_from_noisy_measurements(rng):
    poses = _chain(6, rng)
    graph = PoseGraph(6, reference=0)
    _fill(graph, poses, rng)
    result = graph.optimise()
    assert result.converged
    assert result.final_error < result.initial_error
    for i in range(6):
        rotation, translation = transform_difference(graph.poses[i], poses[i])
        assert np.degrees(rotation) < 0.5
        assert translation < 0.06


def test_rejects_a_grossly_wrong_edge(rng):
    poses = _chain(6, rng)
    graph = PoseGraph(6, reference=0)
    _fill(graph, poses, rng)
    graph.add_edge(
        1,
        4,
        se3_exp(np.array([0.3, 0.2, 1.0, 5.0, -3.0, 1.0])),
        rmse=0.01,
        fitness=0.5,
        n_correspondences=1000,
    )
    bad_index = len(graph.edges) - 1

    result = graph.optimise()
    assert bad_index in result.rejected_edges
    for i in range(6):
        assert transform_difference(graph.poses[i], poses[i])[1] < 0.08


def test_outlier_rejection_never_disconnects_the_graph(rng):
    """A wrong edge that is also the only link must be kept, not orphaned."""
    poses = _chain(4, rng)
    graph = PoseGraph(4, reference=0)
    _fill(graph, poses, rng, complete=False)  # a bare chain 0-1-2-3
    graph.edges[-1].transform = se3_exp(np.array([0.5, 0.1, 0.9, 4.0, 2.0, 1.0]))
    result = graph.optimise()
    assert 2 not in result.rejected_edges or len(result.rejected_edges) == 0
    assert all(np.all(np.isfinite(p)) for p in graph.poses)


def test_reference_pose_stays_fixed(rng):
    poses = _chain(5, rng)
    graph = PoseGraph(5, reference=2)
    _fill(graph, poses, rng)
    graph.optimise()
    assert np.allclose(graph.poses[2], np.eye(4))


def test_relative_matches_the_measurement(rng):
    poses = _chain(4, rng)
    graph = PoseGraph(4, reference=0)
    _fill(graph, poses, rng, noise_r=0.0, noise_t=0.0)
    graph.optimise()
    expected = invert(poses[2]) @ poses[1]
    assert transform_difference(graph.relative(1, 2), expected)[1] < 1e-3


def test_initialise_spans_the_tree(rng):
    poses = _chain(5, rng)
    graph = PoseGraph(5, reference=0)
    _fill(graph, poses, rng, noise_r=0.0, noise_t=0.0, complete=False)
    graph.initialise()
    for i in range(5):
        assert transform_difference(graph.poses[i], poses[i])[1] < 1e-6


def test_components_detects_a_split_survey():
    graph = PoseGraph(4)
    graph.add_edge(0, 1, np.eye(4))
    graph.add_edge(2, 3, np.eye(4))
    assert graph.components() == [[0, 1], [2, 3]]


def test_empty_graph_optimises_trivially():
    result = PoseGraph(3).optimise()
    assert result.converged and result.iterations == 0


def test_self_edges_and_bad_indices_are_rejected():
    graph = PoseGraph(3)
    with pytest.raises(ValueError):
        graph.add_edge(1, 1, np.eye(4))
    with pytest.raises(ValueError):
        graph.add_edge(0, 5, np.eye(4))
    with pytest.raises(ValueError):
        PoseGraph(2, reference=7)


def test_information_grows_with_quality():
    good = default_information(rmse=0.005, fitness=0.9, n_correspondences=10000)
    poor = default_information(rmse=0.05, fitness=0.2, n_correspondences=100)
    assert np.trace(good) > np.trace(poor)


def test_fixed_nodes_hold_and_anchor_the_rest(rng):
    """Sylva's addition: several scans held at trusted poses, the rest solved into their frame."""
    poses = _chain(5, rng)
    graph = PoseGraph(5, reference=0, fixed={1: poses[1], 3: poses[3]})
    graph.poses[0] = poses[0]
    _fill(graph, poses, rng)
    graph.optimise()
    assert np.allclose(graph.poses[1], poses[1]) and np.allclose(graph.poses[3], poses[3])
    for i in (2, 4):
        assert transform_difference(graph.poses[i], poses[i])[1] < 0.06


def test_rejects_a_wrong_edge_while_a_scan_is_unregistered(rng):
    """Outlier edges are rejected even while a node is unreachable, so that
    rejection also works in surveys that register only in part."""
    poses = _chain(6, rng)
    graph = PoseGraph(7, reference=0)  # node 6: a scan nothing registered to
    _fill(graph, poses, rng)
    graph.add_edge(
        1,
        4,
        se3_exp(np.array([0.3, 0.2, 1.0, 5.0, -3.0, 1.0])),
        rmse=0.01,
        fitness=0.5,
        n_correspondences=1000,
    )
    bad_index = len(graph.edges) - 1
    result = graph.optimise()
    assert bad_index in result.rejected_edges
    for i in range(6):
        assert transform_difference(graph.poses[i], poses[i])[1] < 0.08


def test_poses_never_rest_on_a_rejected_edge(rng):
    """The last rejection pass is solved again without what it rejected."""
    poses = _chain(6, rng)
    graph = PoseGraph(6, reference=0)
    _fill(graph, poses, rng)
    graph.add_edge(
        1,
        4,
        se3_exp(np.array([0.3, 0.2, 1.0, 5.0, -3.0, 1.0])),
        rmse=0.01,
        fitness=0.5,
        n_correspondences=1000,
    )
    result = graph.optimise(max_rejection_passes=1)
    assert result.rejected_edges
    clean = PoseGraph(6, reference=0)
    for k, e in enumerate(graph.edges):
        if k not in result.rejected_edges:
            clean.add_edge(e.i, e.j, e.transform, information=e.information)
    clean.optimise(reject_outliers=False)
    for i in range(6):
        assert transform_difference(graph.poses[i], clean.poses[i])[1] < 1e-3  # 15 mm if kept


def _plane_hessian(points, normals):
    a = np.column_stack([np.cross(points, normals), normals])
    return a.T @ a


def test_plane_edge_information_is_expressed_in_the_residual_frame(rng):
    """For the truth se3_exp(d) @ Z, the edge residual carries d's information."""
    p = rng.uniform(-10, 10, (400, 3))
    n = rng.normal(size=(400, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    H = _plane_hessian(p, n)
    Z = se3_exp(np.array([0.1, -0.2, 0.7, 12.0, -4.0, 1.5]))
    info = plane_edge_information(H, 0.01, 400, Z, patch_points=1.0)
    info_target = H / 400 * 400 / 0.01**2
    for _ in range(5):
        d = rng.normal(0, 1e-4, 6)
        r = se3_log(invert(Z) @ se3_exp(d) @ Z)
        assert np.isclose(r @ info @ r, d @ info_target @ d, rtol=1e-3)


def test_flat_ground_leaves_the_horizontal_free(rng):
    """Correspondences on level ground fix height, roll and pitch only."""
    p = np.column_stack([rng.uniform(-15, 15, (2000, 2)), np.zeros(2000)])
    n = np.tile([0.0, 0.0, 1.0], (2000, 1))
    info = plane_edge_information(_plane_hessian(p, n), 0.01, 2000, np.eye(4))
    scale = np.trace(info)
    for axis in (2, 3, 4):  # yaw, x, y
        assert info[axis, axis] < 1e-6 * scale
    for axis in (0, 1, 5):  # roll, pitch, z
        assert info[axis, axis] > 1e-3 * scale


def test_information_grows_with_overlap_not_without_bound(rng):
    p = rng.uniform(-10, 10, (1000, 3))
    n = rng.normal(size=(1000, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    H = _plane_hessian(p, n)
    small = plane_edge_information(H / 10, 0.01, 100, np.eye(4))
    large = plane_edge_information(H, 0.01, 1000, np.eye(4))
    assert np.allclose(large, 10 * small, rtol=1e-6)
    sharp = plane_edge_information(H, 0.0001, 1000, np.eye(4), min_sigma=0.005)
    assert np.allclose(sharp, plane_edge_information(H, 0.005, 1000, np.eye(4)))
