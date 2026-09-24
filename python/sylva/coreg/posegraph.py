# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Multi-scan global optimisation.

Pairwise registration leaves a survey inconsistent: going A->B->C->A does not
return to the start, and the drift shows up as doubled stems. A pose graph
finds the poses that best explain every pairwise measurement at once,
spreading the closure error over the loop (Lu & Milios 1997).

Nodes are ``world_from_scan`` poses; edges are measured relative transforms
with an information matrix saying how far each is trusted. The solver is
Levenberg-Marquardt (Levenberg 1944; Marquardt 1963) on SE(3) with a Huber
(1964) kernel and an explicit outlier pass,
because in a forest a pairwise match can be confidently and completely wrong
when two parts of a stand have similar stem patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms import identity, invert, se3_exp, se3_log

__all__ = ["OptimisationResult", "PoseGraph", "PoseGraphEdge", "default_information"]


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
    n = max(int(n_correspondences), 1)
    sigma_t = max(rmse, 1e-4) / np.sqrt(n)
    sigma_r = sigma_t / max(extent, 1e-3)
    scale = max(min(fitness, 1.0), 1e-3)
    info = np.eye(6)
    info[:3, :3] *= scale / sigma_r**2
    info[3:, 3:] *= scale / sigma_t**2
    return info


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
        adjacency = _adjacency(self.edges, self.n_nodes)
        seen: set[int] = set()
        out = []
        for start in range(self.n_nodes):
            if start in seen:
                continue
            stack, group = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                group.append(node)
                for nb in adjacency[node]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            out.append(sorted(group))
        return out

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
        adjacency: dict[int, list[tuple[float, PoseGraphEdge]]] = {
            i: [] for i in range(self.n_nodes)
        }
        for e in self.edges:
            adjacency[e.i].append((e.weight, e))
            adjacency[e.j].append((e.weight, e))
        self.poses = [identity() for _ in range(self.n_nodes)]
        for k, pose in self.fixed.items():
            self.poses[k] = pose.copy()
        visited = set(self._anchors())
        frontier = sorted((t for a in sorted(visited) for t in adjacency[a]), key=lambda t: -t[0])
        while frontier:
            frontier.sort(key=lambda t: -t[0])
            _, edge = frontier.pop(0)
            if edge.i in visited and edge.j in visited:
                continue
            if edge.i in visited:
                known, unknown = edge.i, edge.j
                self.poses[unknown] = self.poses[known] @ invert(edge.transform)
            else:
                known, unknown = edge.j, edge.i
                self.poses[unknown] = self.poses[known] @ edge.transform
            visited.add(unknown)
            frontier.extend(adjacency[unknown])
        return self.poses

    def residual(self, edge: PoseGraphEdge, poses: list[np.ndarray] | None = None) -> np.ndarray:
        """6-vector error of one edge under ``poses`` (default: the current ones)."""
        poses = poses or self.poses
        return se3_log(invert(edge.transform) @ invert(poses[edge.j]) @ poses[edge.i])

    def total_error(
        self, poses: list[np.ndarray] | None = None, edges: list[PoseGraphEdge] | None = None
    ) -> float:
        """Sum of squared Mahalanobis edge errors over ``edges`` (default: all)."""
        poses = poses or self.poses
        edges = self.edges if edges is None else edges
        return float(sum((r := self.residual(e, poses)) @ e.information @ r for e in edges))

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
            from the anchors.
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
        free_nodes = [k for k in range(self.n_nodes) if k not in self._anchors()]
        if all(np.allclose(self.poses[k], identity()) for k in free_nodes):
            self.initialise()
        initial_error = self.total_error()
        rejected: list[int] = []
        iterations = 0
        converged = False
        for _ in range(max_rejection_passes if reject_outliers else 1):
            active = [e for k, e in enumerate(self.edges) if k not in rejected]
            if not active:
                break
            used, converged = self._run_lm(active, max_iterations, tolerance, huber_delta)
            iterations += used
            if not reject_outliers:
                break
            new = self._find_outliers(active, outlier_sigma)
            if not new:
                break
            rejected.extend(self.edges.index(e) for e in new)
        errors = np.array(
            [
                float(np.sqrt(max((r := self.residual(e)) @ e.information @ r, 0.0)))
                for e in self.edges
            ]
        )
        dropped = set(rejected)
        kept = [e for k, e in enumerate(self.edges) if k not in dropped]
        return OptimisationResult(
            self.poses,
            iterations,
            converged,
            initial_error,
            self.total_error(edges=kept),
            sorted(dropped),
            errors,
        )

    def _run_lm(
        self, edges: list[PoseGraphEdge], max_iterations: int, tolerance: float, huber_delta: float
    ) -> tuple[int, bool]:
        anchors = self._anchors()
        slot = {
            node: k for k, node in enumerate(n for n in range(self.n_nodes) if n not in anchors)
        }
        dim = 6 * len(slot)
        if dim == 0:
            return 0, True
        lam = 1e-4
        error = self._error(edges, self.poses, huber_delta)
        improvement = 0.0
        for iteration in range(max_iterations):
            H = np.zeros((dim, dim))
            b = np.zeros(dim)
            for edge in edges:
                e_vec = self.residual(edge)
                omega = edge.information * _huber_weight(e_vec, edge.information, huber_delta)
                Ji, Jj = self._jacobians(edge)
                blocks = [(slot[n], J) for n, J in ((edge.i, Ji), (edge.j, Jj)) if n in slot]
                for si, Ja in blocks:
                    b[6 * si : 6 * si + 6] -= Ja.T @ omega @ e_vec
                    for sj, Jb in blocks:
                        H[6 * si : 6 * si + 6, 6 * sj : 6 * sj + 6] += Ja.T @ omega @ Jb
            diagonal = np.maximum(H.diagonal(), 1e-12)
            for _ in range(12):  # damping search
                try:
                    delta = np.linalg.solve(H + np.diag(lam * diagonal), b)
                except np.linalg.LinAlgError:
                    delta = None
                if delta is None or not np.all(np.isfinite(delta)):
                    lam *= 10.0
                    continue
                candidate = [p.copy() for p in self.poses]
                for node, k in slot.items():
                    candidate[node] = candidate[node] @ se3_exp(delta[6 * k : 6 * k + 6])
                new_error = self._error(edges, candidate, huber_delta)
                if new_error <= error:
                    self.poses = candidate
                    improvement = error - new_error
                    error = new_error
                    lam = max(lam * 0.5, 1e-12)
                    break
                lam *= 10.0
            else:
                return iteration + 1, False
            if improvement < tolerance * max(error, 1.0):
                return iteration + 1, True
        return max_iterations, False

    def _error(
        self, edges: list[PoseGraphEdge], poses: list[np.ndarray], huber_delta: float
    ) -> float:
        total = 0.0
        for edge in edges:
            e_vec = self.residual(edge, poses)
            chi2 = float(e_vec @ edge.information @ e_vec)
            total += (
                2.0 * huber_delta * np.sqrt(chi2) - huber_delta**2
                if chi2 > huber_delta**2
                else chi2
            )
        return total

    def _jacobians(self, edge: PoseGraphEdge, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
        """Numerical Jacobians of an edge residual; graphs here have tens of nodes."""
        base = self.residual(edge)
        Ji = np.zeros((6, 6))
        Jj = np.zeros((6, 6))
        for k in range(6):
            step = np.zeros(6)
            step[k] = eps
            perturbation = se3_exp(step)
            poses = list(self.poses)
            poses[edge.i] = self.poses[edge.i] @ perturbation
            Ji[:, k] = (self.residual(edge, poses) - base) / eps
            poses = list(self.poses)
            poses[edge.j] = self.poses[edge.j] @ perturbation
            Jj[:, k] = (self.residual(edge, poses) - base) / eps
        return Ji, Jj

    def _find_outliers(self, edges: list[PoseGraphEdge], sigma: float) -> list[PoseGraphEdge]:
        """Edges whose Mahalanobis error is far above the median, never disconnecting the graph."""
        if len(edges) < 4:
            return []
        errors = np.array(
            [float(np.sqrt(max((r := self.residual(e)) @ e.information @ r, 0.0))) for e in edges]
        )
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median))) * 1.4826
        if mad <= 1e-9:
            return []
        threshold = median + sigma * mad
        candidates = sorted(
            (e for e, err in zip(edges, errors, strict=True) if err > threshold),
            key=lambda e: -float(np.sqrt(max((r := self.residual(e)) @ e.information @ r, 0.0))),
        )
        kept = list(edges)
        removed = []
        for candidate in candidates:
            trial = [e for e in kept if e is not candidate]
            if _is_connected(trial, self.n_nodes, self._anchors()):
                kept = trial
                removed.append(candidate)
        return removed

    def relative(self, i: int, j: int) -> np.ndarray:
        """Optimised transform mapping scan ``i`` into scan ``j``'s frame."""
        return invert(self.poses[j]) @ self.poses[i]


def _adjacency(edges: list[PoseGraphEdge], n: int) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for e in edges:
        adjacency[e.i].append(e.j)
        adjacency[e.j].append(e.i)
    return adjacency


def _huber_weight(e_vec: np.ndarray, information: np.ndarray, delta: float) -> float:
    chi2 = float(e_vec @ information @ e_vec)
    return 1.0 if chi2 <= delta**2 else float(delta / np.sqrt(max(chi2, 1e-12)))


def _is_connected(edges: list[PoseGraphEdge], n_nodes: int, anchors: set[int]) -> bool:
    """Would every node still reach an anchor through ``edges``?

    Like tlsalign, this asks for *every* node: while any scan is unregistered
    no edge is ever rejected.
    """
    adjacency = _adjacency(edges, n_nodes)
    seen = set(anchors)
    stack = list(anchors)
    while stack:
        for nb in adjacency[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == n_nodes
