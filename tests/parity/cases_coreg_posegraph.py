"""Parity cases for sylva.coreg.posegraph."""

import numpy as np

from sylva.coreg import posegraph as pg
from sylva.coreg import transforms as tf


def _chain(n, rng, spacing=8.0):
    poses = [np.eye(4)]
    for _ in range(n - 1):
        poses.append(tf.se3_exp(np.concatenate([rng.normal(0, 0.05, 3), rng.normal(0, spacing, 3)])))
    return poses


def _fill(graph, poses, rng, noise_r=0.002, noise_t=0.01, complete=True):
    n = len(poses)
    for i in range(n):
        for j in range(i + 1, n):
            if not complete and j - i > 1:
                continue
            noise = tf.se3_exp(np.concatenate([rng.normal(0, noise_r, 3), rng.normal(0, noise_t, 3)]))
            graph.add_edge(i, j, noise @ tf.invert(poses[j]) @ poses[i], rmse=0.01,
                           fitness=float(rng.uniform(0.3, 0.9)), n_correspondences=int(rng.integers(100, 2000)))


def _result(r, prefix):
    return {
        f"{prefix}_poses": np.array(r.poses),
        f"{prefix}_iterations": r.iterations,
        f"{prefix}_converged": r.converged,
        f"{prefix}_initial_error": r.initial_error,
        f"{prefix}_final_error": r.final_error,
        f"{prefix}_rejected": np.array(r.rejected_edges, int),
        f"{prefix}_edge_errors": np.asarray(r.edge_errors),
    }


def information():
    rng = np.random.default_rng(20)
    out = {
        "default": pg.default_information(0.02, 0.6, 1500),
        "default_floor": pg.default_information(0.0, 3.0, 0, extent=0.0),
        "default_low": pg.default_information(0.5, 0.0, -4, extent=30.0),
    }
    for k in range(3):
        a = rng.normal(size=(500, 6))
        w = rng.uniform(0.2, 1.0, 500)
        H = (a * w[:, None]).T @ a
        T = tf.se3_exp(rng.normal(0, 1, 6))
        out[f"adjoint_{k}"] = pg.adjoint(T)
        out[f"plane_{k}"] = pg.plane_edge_information(H, 0.002 * (k + 1), 500 * (k + 1), T)
        out[f"plane_opts_{k}"] = pg.plane_edge_information(H, 0.01, 50, T, patch_points=0.5, min_sigma=0.02)
    # Flat ground only: the horizontal directions are left free.
    n = np.tile([0.0, 0.0, 1.0], (200, 1))
    p = rng.uniform(-10, 10, (200, 3))
    a = np.column_stack([np.cross(p, n), n])
    out["plane_flat"] = pg.plane_edge_information(a.T @ a, 0.004, 200, np.eye(4))
    return out


def solve():
    rng = np.random.default_rng(21)
    out = {}
    poses = _chain(6, rng)
    graph = pg.PoseGraph(6, reference=0)
    _fill(graph, poses, rng)
    out.update(_result(graph.optimise(), "noisy"))
    out["noisy_relative"] = graph.relative(1, 4)
    out["noisy_residual"] = graph.residual(graph.edges[3])
    out["noisy_total"] = graph.total_error()
    # A grossly wrong edge, rejected.
    poses = _chain(6, rng)
    graph = pg.PoseGraph(6, reference=0)
    _fill(graph, poses, rng)
    graph.edges[4].transform = tf.se3_exp(np.array([0.0, 0.0, 0.8, 5.0, -4.0, 0.0])) @ graph.edges[4].transform
    out.update(_result(graph.optimise(), "outlier"))
    # The same without rejection, with a tighter Huber kernel.
    graph.poses = [np.eye(4) for _ in range(6)]
    out.update(_result(graph.optimise(reject_outliers=False, huber_delta=1.0, max_iterations=7), "no_reject"))
    # A chain with short loops: rejection must never cut a node off.
    poses = _chain(5, rng)
    graph = pg.PoseGraph(5, reference=2)
    _fill(graph, poses, rng, complete=False)
    for i, j in [(0, 2), (1, 3), (2, 4), (0, 4), (1, 4), (0, 3)]:
        noise = tf.se3_exp(np.concatenate([rng.normal(0, 0.002, 3), rng.normal(0, 0.01, 3)]))
        graph.add_edge(i, j, noise @ tf.invert(poses[j]) @ poses[i], rmse=0.002, fitness=0.4,
                       n_correspondences=300)
    graph.edges[5].transform = tf.se3_exp(np.array([0.0, 0.0, 0.5, 3.0, 0.0, 0.0])) @ graph.edges[5].transform
    out.update(_result(graph.optimise(), "chain"))
    # Fixed nodes, a scan left unregistered, a wrong edge.
    poses = _chain(7, rng)
    graph = pg.PoseGraph(7, reference=0, fixed={3: poses[3] @ tf.invert(poses[0])})
    for i in range(6):
        for j in range(i + 1, 6):
            noise = tf.se3_exp(np.concatenate([rng.normal(0, 0.002, 3), rng.normal(0, 0.01, 3)]))
            info = pg.default_information(0.01, 0.5, 800) if (i + j) % 2 else None
            graph.add_edge(i, j, noise @ tf.invert(poses[j]) @ poses[i], information=info,
                           fitness=0.5, n_correspondences=800, label=f"{i}-{j}")
    graph.edges[2].transform = tf.se3_exp(np.array([0.3, 0.0, 0.0, 0.0, 2.0, 0.0])) @ graph.edges[2].transform
    out.update(_result(graph.optimise(outlier_sigma=3.0, max_rejection_passes=3), "fixed"))
    out["fixed_components"] = np.array([len(c) for c in graph.components()])
    # A starting guess that is already set is not replaced by the spanning tree.
    poses = _chain(4, rng)
    graph = pg.PoseGraph(4)
    _fill(graph, poses, rng)
    graph.poses = [p @ tf.se3_exp(rng.normal(0, 0.01, 6)) for p in poses]
    out.update(_result(graph.optimise(tolerance=1e-10), "warm"))
    empty = pg.PoseGraph(3)
    out.update(_result(empty.optimise(), "empty"))
    return out


def structure():
    rng = np.random.default_rng(22)
    poses = _chain(8, rng)
    graph = pg.PoseGraph(8, reference=1, fixed={5: poses[5]})
    for i, j, w in [(0, 1, 0.5), (1, 2, 0.9), (0, 2, 0.9), (2, 3, 0.2), (3, 0, 0.7), (6, 7, 1.0), (5, 4, 0.3)]:
        noise = tf.se3_exp(rng.normal(0, 0.005, 6))
        graph.add_edge(i, j, noise @ tf.invert(poses[j]) @ poses[i], fitness=w, n_correspondences=100)
    comps = graph.components()
    out = {
        "components_sizes": np.array([len(c) for c in comps]),
        "components_flat": np.concatenate([np.array(c) for c in comps]),
        "initialise_1": np.array(graph.initialise()),
        "initialise_3": np.array(graph.initialise(reference=3)),
        "reference": graph.reference,
    }
    out["residuals"] = np.array([graph.residual(e) for e in graph.edges])
    out["residuals_other"] = np.array([graph.residual(e, poses) for e in graph.edges])
    out["total"] = graph.total_error()
    out["total_subset"] = graph.total_error(poses, graph.edges[2:5])
    return out


CASES = {"information": information, "solve": solve, "structure": structure}
