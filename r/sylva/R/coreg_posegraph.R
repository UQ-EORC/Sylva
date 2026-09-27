# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# The graph is a plain list; the solve runs in the Rust core. Nodes and edges
# are numbered from 1 here and from 0 in the core.

#' Edge information matrices
#'
#' `default_information()`: a diagonal information matrix from registration
#' quality (translation precision is the residual over the square root of
#' the correspondence count, rotation precision that spread over the scan's
#' `extent`). `plane_edge_information()`: the information of an ICP edge
#' from its point-to-plane Gauss-Newton matrix, scaled to the residual,
#' with `n` correspondences counting as `n / patch_points` independent ones,
#' and carried into the frame of the edge residual. `adjoint()`: the 6 x 6
#' adjoint of a transform on twists.
#'
#' @param rmse,fitness,n_correspondences Registration quality of the edge.
#' @param extent Spatial extent of a scan (m).
#' @param hessian 6 x 6 `sum w a a^T` of the correspondences.
#' @param sigma Weighted RMS point-to-plane residual (m).
#' @param n Correspondences behind `hessian`.
#' @param transform,T The measured `target_from_source` transform.
#' @param patch_points Correspondences per independent observation.
#' @param min_sigma Floor on `sigma` (m).
#' @return A 6 x 6 matrix, rotation block first.
#' @export
default_information <- function(rmse, fitness, n_correspondences, extent = 15) {
  core_coreg_default_information(as.double(rmse), as.double(fitness), as.double(n_correspondences), as.double(extent))
}

#' @rdname default_information
#' @export
plane_edge_information <- function(hessian, sigma, n, transform, patch_points = 100, min_sigma = 0.005) {
  h <- as.matrix(hessian)
  storage.mode(h) <- "double"
  core_coreg_plane_edge_information(h, as.double(sigma), as.double(n), as_transform(transform),
                                    as.double(patch_points), as.double(min_sigma))
}

#' @rdname default_information
#' @export
adjoint <- function(T) core_coreg_adjoint(as_transform(T))

#' A pose graph over scan poses
#'
#' `poses[[k]]` is `world_from_scan_k`; the world frame is that of the
#' `reference` node, held fixed with any nodes in `fixed`. `add_edge()` adds
#' a measurement that scan `i` maps into scan `j` by `transform` and returns
#' the graph; `pose_graph_edge()` builds such an edge. `components()` lists
#' the connected components; `initialise()` sets the poses by walking a
#' maximum-weight spanning tree from the anchors; `residual()` and
#' `total_error()` give edge errors; `optimise()` solves the graph;
#' `relative()` is the transform mapping scan `i` into scan `j`.
#'
#' @param n_nodes Number of scans.
#' @param reference Node whose pose defines the world frame.
#' @param fixed Further nodes held fixed: a list of 4 x 4 poses named by
#'   node number, e.g. `list("3" = T3)`.
#' @param graph A `sylva_pose_graph`.
#' @param i,j Edge endpoints (nodes, from 1).
#' @param transform The measured `j_from_i` transform.
#' @param information 6 x 6 information matrix; from
#'   [default_information()] if `NULL`.
#' @param fitness,rmse,n_correspondences Registration quality of the edge.
#' @param label Free text.
#' @return `pose_graph()` and `add_edge()` a `sylva_pose_graph`.
#' @export
pose_graph <- function(n_nodes, reference = 1, fixed = NULL) {
  n_nodes <- as.integer(n_nodes)
  if (n_nodes < 1) stop("a pose graph needs at least one node")
  if (reference < 1 || reference > n_nodes) stop("reference node index out of range")
  fixed <- lapply(or_else(fixed, list()), as_transform)
  nodes <- as.integer(names(fixed))
  if (length(fixed) && (anyNA(nodes) || any(nodes < 1 | nodes > n_nodes))) stop("fixed nodes must be named by node number")
  names(fixed) <- as.character(nodes)
  poses <- rep(list(diag(4)), n_nodes)
  poses[nodes] <- fixed
  structure(list(n_nodes = n_nodes, reference = as.integer(reference), fixed = fixed, poses = poses, edges = list()),
            class = "sylva_pose_graph")
}

or_else <- function(a, b) if (is.null(a)) b else a

#' @rdname pose_graph
#' @export
pose_graph_edge <- function(i, j, transform, information = NULL, fitness = 1, rmse = 0.01, n_correspondences = 0,
                            label = "") {
  if (is.null(information)) information <- default_information(rmse, fitness, n_correspondences)
  information <- matrix(as.double(information), 6, 6)
  structure(list(i = as.integer(i), j = as.integer(j), transform = as_transform(transform), information = information,
                 fitness = as.double(fitness), rmse = as.double(rmse), n_correspondences = n_correspondences,
                 label = as.character(label)),
            class = "sylva_pose_graph_edge")
}

edge_weight <- function(e) e$fitness * max(e$n_correspondences, 1)

#' @rdname pose_graph
#' @export
add_edge <- function(graph, i, j, transform, information = NULL, fitness = 1, rmse = 0.01, n_correspondences = 0,
                     label = "") {
  g <- unclass(graph)
  if (!(i >= 1 && i <= g$n_nodes && j >= 1 && j <= g$n_nodes)) stop("edge endpoints out of range")
  if (i == j) stop("self-edges are not allowed")
  g$edges[[length(g$edges) + 1]] <- pose_graph_edge(i, j, transform, information, fitness, rmse, n_correspondences, label)
  structure(g, class = "sylva_pose_graph")
}

graph_edges <- function(edges) {
  list(i = vapply(edges, function(e) e$i - 1, 0), j = vapply(edges, function(e) e$j - 1, 0),
       transforms = lapply(edges, `[[`, "transform"), information = lapply(edges, `[[`, "information"),
       weights = vapply(edges, edge_weight, 0))
}

fixed_args <- function(g) list(as.double(names(g$fixed)) - 1, unname(g$fixed))

#' @rdname pose_graph
#' @export
components <- function(graph) {
  g <- unclass(graph)
  e <- graph_edges(g$edges)
  lapply(core_coreg_posegraph_components(g$n_nodes, e$i, e$j), function(c) as.integer(c) + 1L)
}

#' @rdname pose_graph
#' @export
initialise <- function(graph, reference = NULL) {
  g <- unclass(graph)
  if (!is.null(reference)) g$reference <- as.integer(reference)
  e <- graph_edges(g$edges)
  f <- fixed_args(g)
  g$poses <- core_coreg_posegraph_initialise(g$n_nodes, e$i, e$j, e$transforms, e$weights, g$reference - 1, f[[1]], f[[2]])
  structure(g, class = "sylva_pose_graph")
}

edge_list <- function(g, edges) {
  if (is.null(edges)) return(g$edges)
  if (inherits(edges, "sylva_pose_graph_edge")) return(list(edges))
  if (is.numeric(edges)) return(g$edges[edges])
  edges
}

#' @rdname pose_graph
#' @param edge An edge (its number in the graph, or a `sylva_pose_graph_edge`).
#' @param poses Poses to evaluate at (default: the graph's).
#' @export
residual <- function(graph, edge, poses = NULL) {
  g <- unclass(graph)
  e <- graph_edges(edge_list(g, edge))
  if (length(e$i) != 1) stop("residual() takes one edge")
  core_coreg_posegraph_residuals(e$i, e$j, e$transforms, lapply(or_else(poses, g$poses), as_transform))[1, ]
}

#' @rdname pose_graph
#' @param edges Edges to sum over (numbers or edges; default: all).
#' @export
total_error <- function(graph, poses = NULL, edges = NULL) {
  g <- unclass(graph)
  e <- graph_edges(edge_list(g, edges))
  core_coreg_posegraph_total_error(e$i, e$j, e$transforms, e$information, lapply(or_else(poses, g$poses), as_transform))
}

#' @rdname pose_graph
#' @export
relative <- function(graph, i, j) {
  p <- unclass(graph)$poses
  invert(p[[j]]) %*% p[[i]]
}

#' Solve a pose graph
#'
#' Levenberg-Marquardt over all edges on SE(3), with a Huber kernel and
#' solve-and-reject passes that drop edges far above the median error
#' without cutting a node off from the anchors it reached. If every free
#' node is still at the identity the poses are first set by [initialise()].
#' For other objects `optimise()` is [stats::optimise()].
#'
#' @param f A `sylva_pose_graph`.
#' @param max_iterations Cap per rejection pass.
#' @param tolerance Stop when an iteration improves the error by less than
#'   this fraction of it.
#' @param huber_delta Mahalanobis distance beyond which an edge is
#'   down-weighted.
#' @param reject_outliers Remove edges whose error stays far above the
#'   median, then solve again.
#' @param outlier_sigma Robust standard deviations above the median that
#'   make an outlier.
#' @param max_rejection_passes Solve-and-reject passes.
#' @param ... Passed to [stats::optimise()] for other objects.
#' @return A `sylva_optimisation_result`: `poses`, `iterations`,
#'   `converged`, `initial_error`, `final_error`, `rejected_edges` (edge
#'   numbers), `edge_errors`, and `graph`, the graph at the solution.
#' @export
optimise <- function(f, ...) UseMethod("optimise")

#' @export
optimise.default <- function(f, ...) stats::optimise(f, ...)

#' @rdname optimise
#' @export
optimise.sylva_pose_graph <- function(f, max_iterations = 200, tolerance = 1e-6, huber_delta = 3, reject_outliers = TRUE,
                                      outlier_sigma = 5, max_rejection_passes = 2, ...) {
  g <- unclass(f)
  if (!length(g$edges)) {
    return(structure(list(poses = g$poses, iterations = 0L, converged = TRUE, initial_error = 0, final_error = 0,
                          rejected_edges = integer(), edge_errors = double(), graph = f),
                     class = "sylva_optimisation_result"))
  }
  e <- graph_edges(g$edges)
  fx <- fixed_args(g)
  r <- core_coreg_posegraph_optimise(g$n_nodes, e$i, e$j, e$transforms, e$information, e$weights, g$reference - 1,
                                     fx[[1]], fx[[2]], lapply(g$poses, as_transform), as.double(max_iterations),
                                     as.double(tolerance), as.double(huber_delta), isTRUE(reject_outliers),
                                     as.double(outlier_sigma), as.double(max_rejection_passes))
  g$poses <- r$poses
  r$iterations <- as.integer(r$iterations)
  r$rejected_edges <- as.integer(r$rejected_edges) + 1L
  r$graph <- structure(g, class = "sylva_pose_graph")
  structure(r, class = "sylva_optimisation_result")
}

#' @export
print.sylva_pose_graph <- function(x, ...) {
  cat(sprintf("<sylva_pose_graph> %d nodes, %d edges\n", x$n_nodes, length(x$edges)))
  invisible(x)
}

#' @export
print.sylva_optimisation_result <- function(x, ...) {
  cat(sprintf("<sylva_optimisation_result> %d iterations, converged=%s, error %.4g -> %.4g, rejected=%d\n",
              x$iterations, x$converged, x$initial_error, x$final_error, length(x$rejected_edges)))
  invisible(x)
}
