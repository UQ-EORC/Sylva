# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Multi-scan global optimisation.

Pairwise registration leaves a survey inconsistent: going A->B->C->A does not
return to the start, and the drift shows up as doubled stems. A pose graph
finds the poses that best explain every pairwise measurement at once,
spreading the closure error over the loop (Lu & Milios 1997).

Nodes are ``world_from_scan`` poses; edges are measured relative transforms
with an information matrix saying how far, and in which directions, each is
trusted: an ICP edge's comes from its point-to-plane correspondences
(:func:`plane_edge_information`), so a pair matched mostly on flat ground
holds height, roll and pitch firmly and the horizontal position loosely. The solver is
Levenberg-Marquardt (Levenberg 1944; Marquardt 1963) on SE(3) with a Huber
(1964) kernel and an explicit outlier pass,
because in a forest a pairwise match can be confidently and completely wrong
when two parts of a stand have similar stem patterns. The graph here is a
container; the solve runs in the Rust core.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import _core
from .transforms import _mat4, identity, invert

__all__ = [
    "OptimisationResult",
    "PoseGraph",
    "PoseGraphEdge",
    "default_information",
    "plane_edge_information",
]


def default_information(
    rmse: float, fitness: float, n_correspondences: int, extent: float = 15.0
) -> np.ndarray:
    """Diagonal information matrix from registration quality.

    Translation precision scales as the residual over the square root of the
    correspondence count; rotation precision is that positional uncertainty
    spread over the scan's ``extent`` (m).

    Parameters
    ----------
    rmse, fitness, n_correspondences
        ICP quality of the edge.
    extent
        Spatial extent of a scan (m).

    Returns
    -------
    numpy.ndarray
        ``(6, 6)`` information, rotation block first.
    """
    return _core.coreg_default_information(
        float(rmse), float(fitness), int(n_correspondences), float(extent)
    )


def adjoint(T: np.ndarray) -> np.ndarray:
    """``(6, 6)`` adjoint of a transform on ``[omega, v]`` twists:
    ``T @ se3_exp(xi) @ invert(T) == se3_exp(adjoint(T) @ xi)``."""
    return _core.coreg_adjoint(_mat4(T))


def plane_edge_information(
    hessian: np.ndarray,
    sigma: float,
    n: int,
    transform: np.ndarray,
    *,
    patch_points: float = 100.0,
    min_sigma: float = 0.005,
) -> np.ndarray:
    """Edge information from the point-to-plane correspondences of an ICP.

    The Gauss-Newton matrix of the correspondences
    (:class:`~sylva.coreg.icp.PlaneInformation`) says in which directions the
    surfaces pin the transform; it is scaled to the residual and carried from
    the target frame, where ICP perturbs the transform, into the frame of the
    edge residual. Neighbouring residuals share the error of the surface they
    sample, so ``n`` correspondences count as ``n / patch_points``
    independent ones: the information of an edge grows with its overlap, but
    not to the point where millimetres of disagreement between two edges
    look like outliers.

    Parameters
    ----------
    hessian
        ``(6, 6)`` ``sum w a a^T`` for updates ``se3_exp(xi) @ transform``.
    sigma
        Weighted RMS point-to-plane residual (m); floored at ``min_sigma``.
    n
        Correspondences behind ``hessian``.
    transform
        The measured ``target_from_source`` transform.
    patch_points
        Correspondences per independent observation.
    min_sigma
        Floor on ``sigma`` (m): a residual of a millimetre says the surfaces
        agree, not that the pose is known to a millimetre.

    Returns
    -------
    numpy.ndarray
        ``(6, 6)`` information of the edge residual ``se3_log(invert(transform)
        @ target_from_source)``, rotation block first.
    """
    return _core.coreg_plane_edge_information(
        np.ascontiguousarray(np.asarray(hessian, dtype=float).reshape(6, 6)),
        float(sigma),
        int(n),
        _mat4(transform),
        float(patch_points),
        float(min_sigma),
    )


@dataclass(eq=False)  # edges hold arrays and are tracked by identity
class PoseGraphEdge:
    """A measured relative transform.

    ``transform`` maps scan ``i`` into scan ``j`` (``j_from_i``).
    """

    i: int
    j: int
    transform: np.ndarray
    information: np.ndarray | None = None
    fitness: float = 1.0
    rmse: float = 0.01
    n_correspondences: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        self.transform = np.asarray(self.transform, dtype=float).reshape(4, 4)
        if self.information is None:
            self.information = default_information(self.rmse, self.fitness, self.n_correspondences)
        self.information = np.asarray(self.information, dtype=float).reshape(6, 6)

    @property
    def weight(self) -> float:
        """Scalar trust of the edge, for the spanning-tree initialisation."""
        return float(self.fitness * max(self.n_correspondences, 1))


@dataclass
class OptimisationResult:
    """Diagnostics of :meth:`PoseGraph.optimise`."""

    poses: list[np.ndarray]
    iterations: int
    converged: bool
    initial_error: float
    final_error: float
    rejected_edges: list[int] = field(default_factory=list)
    edge_errors: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __repr__(self) -> str:
        return (
            f"OptimisationResult(iterations={self.iterations}, converged={self.converged}, "
            f"error {self.initial_error:.4g} -> {self.final_error:.4g}, "
            f"rejected={len(self.rejected_edges)})"
        )


class PoseGraph:
    """A pose graph over scan poses.

    ``poses[i]`` is ``world_from_scan_i``. The world frame is that of the
    ``reference`` node, held fixed. Other nodes can be held fixed too, at
    poses given in ``fixed`` (scans already trusted, which new scans are
    registered into).

    Parameters
    ----------
    n_nodes
        Number of scans.
    reference
        Node whose pose defines the world frame.
    fixed
        ``{node: world_from_scan}`` of further nodes held fixed.
    """

    def __init__(
        self, n_nodes: int, reference: int = 0, fixed: dict[int, np.ndarray] | None = None
    ) -> None:
        if n_nodes < 1:
            raise ValueError("a pose graph needs at least one node")
        if not 0 <= reference < n_nodes:
            raise ValueError("reference node index out of range")
        self.n_nodes = n_nodes
        self.reference = reference
        self.fixed = {int(k): np.asarray(v, float).copy() for k, v in (fixed or {}).items()}
        self.poses: list[np.ndarray] = [identity() for _ in range(n_nodes)]
        for k, pose in self.fixed.items():
            self.poses[k] = pose.copy()
        self.edges: list[PoseGraphEdge] = []

    def __repr__(self) -> str:
        return f"PoseGraph(nodes={self.n_nodes}, edges={len(self.edges)})"

    def _anchors(self) -> set[int]:
        return {self.reference, *self.fixed}

    def add_edge(
        self,
        i: int,
        j: int,
        transform: np.ndarray,
        *,
        information: np.ndarray | None = None,
        fitness: float = 1.0,
        rmse: float = 0.01,
        n_correspondences: int = 0,
        label: str = "",
    ) -> PoseGraphEdge:
        """Add a measurement that scan ``i`` maps into scan ``j`` by ``transform``.

        Returns
        -------
        PoseGraphEdge
        """
        if not (0 <= i < self.n_nodes and 0 <= j < self.n_nodes):
            raise ValueError("edge endpoints out of range")
        if i == j:
            raise ValueError("self-edges are not allowed")
        edge = PoseGraphEdge(i, j, transform, information, fitness, rmse, n_correspondences, label)
        self.edges.append(edge)
        return edge

    def components(self) -> list[list[int]]:
        """Connected components, as sorted lists of nodes."""
        i, j = [int(e.i) for e in self.edges], [int(e.j) for e in self.edges]
        return [list(c) for c in _core.coreg_posegraph_components(self.n_nodes, i, j)]

    def initialise(self, reference: int | None = None) -> list[np.ndarray]:
        """Initialise the poses by walking a maximum-weight spanning tree.

        Chaining the strongest edges first starts the solve close to the
        optimum, which matters because the SE(3) cost is not convex. Nodes in
        components without an anchor keep the identity.

        Returns
        -------
        list of numpy.ndarray
        """
        if reference is not None:
            self.reference = reference
        i, j, T, _, w = _edge_arrays(self.edges)
        nodes, fixed = self._fixed_arrays()
        poses = _core.coreg_posegraph_initialise(
            self.n_nodes, i, j, T, w, int(self.reference), nodes, fixed
        )
        self.poses = [np.array(p) for p in poses]
        return self.poses

    def _fixed_arrays(self) -> tuple[list[int], np.ndarray]:
        nodes = list(self.fixed)
        poses = np.array([self.fixed[k] for k in nodes], dtype=float).reshape(-1, 4, 4)
        return nodes, poses

    def residual(self, edge: PoseGraphEdge, poses: list[np.ndarray] | None = None) -> np.ndarray:
        """6-vector error of one edge under ``poses`` (default: the current ones)."""
        poses = poses or self.poses
        i, j, T, _, _ = _edge_arrays([edge])
        return _core.coreg_posegraph_residuals(i, j, T, _poses_array(poses))[0]

    def total_error(
        self, poses: list[np.ndarray] | None = None, edges: list[PoseGraphEdge] | None = None
    ) -> float:
        """Sum of squared Mahalanobis edge errors over ``edges`` (default: all)."""
        poses = poses or self.poses
        edges = self.edges if edges is None else edges
        i, j, T, info, _ = _edge_arrays(edges)
        return float(_core.coreg_posegraph_total_error(i, j, T, info, _poses_array(poses)))

    def optimise(
        self,
        max_iterations: int = 200,
        *,
        tolerance: float = 1e-6,
        huber_delta: float = 3.0,
        reject_outliers: bool = True,
        outlier_sigma: float = 5.0,
        max_rejection_passes: int = 2,
    ) -> OptimisationResult:
        """Levenberg-Marquardt over all edges, with outlier rejection.

        Parameters
        ----------
        max_iterations
            Cap per rejection pass; a 62-scan survey needed more than 60.
        tolerance
            Stop when an iteration improves the error by less than this
            fraction of it.
        huber_delta
            Mahalanobis distance beyond which an edge is down-weighted.
        reject_outliers
            Remove edges whose error stays far above the median, then solve
            again. An edge is never removed if that would cut a node off
            from the anchors it reached.
        outlier_sigma
            How many robust standard deviations above the median is an outlier.
        max_rejection_passes
            Solve-and-reject passes.

        Returns
        -------
        OptimisationResult
        """
        if not self.edges:
            return OptimisationResult(self.poses, 0, True, 0.0, 0.0)
        i, j, T, info, w = _edge_arrays(self.edges)
        nodes, fixed = self._fixed_arrays()
        d = _core.coreg_posegraph_optimise(
            self.n_nodes,
            i,
            j,
            T,
            info,
            w,
            int(self.reference),
            nodes,
            fixed,
            _poses_array(self.poses),
            int(max_iterations),
            float(tolerance),
            float(huber_delta),
            bool(reject_outliers),
            float(outlier_sigma),
            int(max_rejection_passes),
        )
        self.poses = [np.array(p) for p in d["poses"]]
        return OptimisationResult(
            self.poses,
            int(d["iterations"]),
            bool(d["converged"]),
            float(d["initial_error"]),
            float(d["final_error"]),
            [int(k) for k in d["rejected_edges"]],
            np.asarray(d["edge_errors"]),
        )

    def relative(self, i: int, j: int) -> np.ndarray:
        """Optimised transform mapping scan ``i`` into scan ``j``'s frame."""
        return invert(self.poses[j]) @ self.poses[i]


def _edge_arrays(edges: list[PoseGraphEdge]):
    """``(i, j, transforms, information, weights)`` of edges, for the core."""
    i = [int(e.i) for e in edges]
    j = [int(e.j) for e in edges]
    T = np.array([e.transform for e in edges], dtype=float).reshape(-1, 4, 4)
    info = np.array([e.information for e in edges], dtype=float).reshape(-1, 6, 6)
    w = [float(e.weight) for e in edges]
    return i, j, T, info, w


def _poses_array(poses: list[np.ndarray]) -> np.ndarray:
    return np.array([np.asarray(p, dtype=float) for p in poses], dtype=float).reshape(-1, 4, 4)
