# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# ------------------------------------------------------------------------ ICP

#' Settings of coregistration ICP
#'
#' The defaults are tlsalign's; see the Python package's `ICPConfig`. The
#' pyramid should span only the error left by the coarse stem match.
#' `distances()` gives the correspondence cut-off of each level (5 x voxel
#' where `max_distances` is `NULL`).
#'
#' @param voxel_sizes Voxel size of each pyramid level (m).
#' @param max_distances Correspondence cut-off per level (m), or `NULL`.
#' @param max_iterations Iterations per level.
#' @param method `"point_to_plane"` or `"point_to_point"`.
#' @param robust `"huber"`, `"tukey"` or `"none"`.
#' @param robust_scale Lower bound (m) of the adaptive robust scale.
#' @param trim_fraction,trim_ramp Trimming, phased in over `trim_ramp`
#'   iterations per level.
#' @param min_planarity,normal_neighbours Planarity gate on target points.
#' @param translation_tolerance,rotation_tolerance Convergence.
#' @param fitness_threshold Distance (m) within which a source point counts
#'   as fitted.
#' @param damping Gauss-Newton damping.
#' @param max_points Cap per level after thinning.
#' @param plateau_tolerance,plateau_patience Early stop on a plateau.
#' @param seed Seed of the random caps.
#' @param config A `sylva_icp_config`.
#' @return A `sylva_icp_config`.
#' @export
icp_config <- function(voxel_sizes = c(0.30, 0.15, 0.07, 0.05), max_distances = c(0.80, 0.40, 0.20, 0.12),
                       max_iterations = 30, method = "point_to_plane", robust = "huber", robust_scale = 0.05,
                       trim_fraction = 0.85, trim_ramp = 3, min_planarity = 0.25, normal_neighbours = 20,
                       translation_tolerance = 1e-4, rotation_tolerance = 2e-5, fitness_threshold = 0.10,
                       damping = 1e-6, max_points = 120000, plateau_tolerance = 0, plateau_patience = 3, seed = 0) {
  structure(as.list(environment()), class = "sylva_icp_config")
}

#' @rdname icp_config
#' @export
distances <- function(config) {
  cfg <- unclass(config)
  if (is.null(cfg$max_distances)) return(5 * cfg$voxel_sizes)
  if (length(cfg$max_distances) != length(cfg$voxel_sizes)) stop("max_distances must have one entry per voxel size")
  cfg$max_distances
}

icp_args <- function(config) {
  cfg <- unclass(or_else(config, icp_config()))
  cfg$voxel_sizes <- as.double(cfg$voxel_sizes)
  cfg$max_distances <- as.double(distances(cfg))
  cfg
}

#' A target's ICP pyramid, built once and reused
#'
#' Every ICP rebuilds the target's voxel pyramid, normals and search trees;
#' registering several scans against a prepared target gives exactly the
#' results of passing the points.
#'
#' @param points n x 3 target points.
#' @param config [icp_config()]; the pyramid is built for its `voxel_sizes`,
#'   `max_points`, `method`, `normal_neighbours` and `seed`.
#' @return A `sylva_icp_target`.
#' @export
icp_target <- function(points, config = NULL) {
  structure(list(ptr = core_coreg_icp_target(as_points(points), icp_args(config))), class = "sylva_icp_target")
}

#' @export
length.sylva_icp_target <- function(x) as.integer(core_coreg_icp_target_len(unclass(x)$ptr))

target_args <- function(target) {
  if (inherits(target, "sylva_icp_target")) list(NULL, unclass(target)$ptr) else list(as_points(target, "target"), NULL)
}

plane_info <- function(d) if (is.null(d)) NULL else structure(d, class = "sylva_plane_information")

#' Fine registration by robust ICP
#'
#' Point-to-plane ICP with a planarity gate, Huber weights with an adaptive
#' scale and a phased-in trim, over a coarse-to-fine voxel pyramid: the
#' Python package's `sylva.coreg.icp`. `plane_information()` gives the
#' point-to-plane information of a transform ICP did not produce, as at the
#' end of `coreg_icp()`. `evaluate_registration()` scores a registration
#' without changing it.
#'
#' @param source n x 3 points in their own frame.
#' @param target n x 3 points in their own frame, or an [icp_target()].
#' @param initial Initial source-to-target transform (identity if `NULL`).
#' @param transform Source-to-target transform.
#' @param config [icp_config()].
#' @param threshold Inlier distance (m).
#' @param max_points Random cap on each cloud after thinning.
#' @param voxel Voxel centroids at this size first (m); `NULL` to skip.
#' @param seed Seed of the random cap.
#' @return `coreg_icp()`: a `sylva_icp_result` (`transform`, `fitness`,
#'   `inlier_rmse`, `n_correspondences`, `iterations`, `converged`,
#'   `history`, `information`); `plane_information()`: a
#'   `sylva_plane_information` (`hessian`, `sigma`, `n`) or `NULL`;
#'   `evaluate_registration()`: `list(fitness, inlier_rmse, n_inliers)`.
#' @export
coreg_icp <- function(source, target, initial = NULL, config = NULL) {
  t <- target_args(target)
  r <- core_coreg_icp(as_points(source, "source"), t[[1]], t[[2]], if (is.null(initial)) NULL else as_transform(initial),
                      icp_args(config))
  r$information <- plane_info(r$information)
  for (k in c("n_correspondences", "iterations")) r[[k]] <- as.integer(r[[k]])
  structure(r, class = "sylva_icp_result")
}

#' @rdname coreg_icp
#' @export
plane_information <- function(source, target, transform, config = NULL) {
  t <- target_args(target)
  plane_info(core_coreg_plane_information(as_points(source, "source"), t[[1]], t[[2]], as_transform(transform),
                                          icp_args(config)))
}

#' @rdname coreg_icp
#' @export
evaluate_registration <- function(source, target, transform, threshold = 0.10, max_points = 200000, voxel = 0.05,
                                  seed = 0) {
  r <- core_coreg_evaluate(as_points(source, "source"), as_points(target, "target"), as_transform(transform),
                           as.double(threshold), as.double(max_points), if (is.null(voxel)) NULL else as.double(voxel),
                           as.double(seed))
  r$n_inliers <- as.integer(r$n_inliers)
  r
}

#' @export
print.sylva_icp_result <- function(x, ...) {
  cat(sprintf("<sylva_icp_result> fitness=%.3f, rmse=%.1f mm, n=%d, iters=%d, converged=%s\n", x$fitness,
              x$inlier_rmse * 1000, x$n_correspondences, x$iterations, x$converged))
  invisible(x)
}

# ------------------------------------------------------------------- geometry

#' Nearest-neighbour search
#'
#' A k-d tree over fixed 2-D or 3-D points; `query()` gives the nearest
#' point to each query within `distance_upper_bound` (exclusive), `Inf` and
#' `NA` where there is none.
#'
#' @param points n x 2 or n x 3 points.
#' @param tree A `sylva_kd_tree`.
#' @param queries Query points of the tree's dimension.
#' @param distance_upper_bound Only neighbours closer than this count.
#' @return `kd_tree()`: a `sylva_kd_tree`; `query()`: `list(distance, index)`
#'   with 1-based indices.
#' @export
kd_tree <- function(points) {
  p <- as.matrix(points)
  storage.mode(p) <- "double"
  structure(core_coreg_kd_tree(p), class = "sylva_kd_tree")
}

#' @rdname kd_tree
#' @export
query <- function(tree, queries, distance_upper_bound = Inf) {
  t <- unclass(tree)
  q <- if (is.null(dim(queries))) matrix(as.double(queries), nrow = 1) else as.matrix(queries)
  storage.mode(q) <- "double"
  r <- core_coreg_kd_query(t$tree, t$dim, q, as.double(distance_upper_bound))
  idx <- as.integer(r$index) + 1L
  idx[idx > t$n] <- NA_integer_
  list(distance = r$distance, index = idx)
}

#' @export
length.sylva_kd_tree <- function(x) as.integer(unclass(x)$n)

#' Local geometry for coregistration
#'
#' `coreg_voxel_downsample()`: one point per voxel (the centroid, or the
#' first), ordered by voxel. `coreg_estimate_normals()`: unit normals and
#' planarity `(l1 - l0) / l2` by local PCA over `k` neighbours (the point
#' itself included), neighbours beyond `radius` left out.
#' `planar_filter()`: the locally planar points (stems, ground, logs), the
#' most useful step before ICP in a forest.
#'
#' @param points n x 3 points.
#' @param voxel Voxel size (m); for `planar_filter()` voxel centroids at
#'   this size first, `NULL` to skip.
#' @param centroid Average the points of each voxel; otherwise keep the first.
#' @param return_counts Also return how many points fed each output point.
#' @param k Neighbours.
#' @param radius Neighbourhood radius (m), or `NULL`.
#' @param min_planarity Planarity a point needs.
#' @return A matrix, or a list (`points` and `counts`; `normals` and
#'   `planarity`).
#' @export
coreg_voxel_downsample <- function(points, voxel, centroid = TRUE, return_counts = FALSE) {
  if (voxel <= 0) stop("voxel size must be positive")
  r <- core_coreg_voxel_centroids(as_points(points), as.double(voxel), isTRUE(centroid))
  if (return_counts) list(points = r$points, counts = as.integer(r$counts)) else r$points
}

#' @rdname coreg_voxel_downsample
#' @export
coreg_estimate_normals <- function(points, k = 20, radius = NULL) {
  core_coreg_estimate_normals(as_points(points), as.double(k), if (is.null(radius)) NULL else as.double(radius))
}

#' @rdname coreg_voxel_downsample
#' @export
planar_filter <- function(points, min_planarity = 0.35, voxel = 0.05, k = 20, radius = 0.15) {
  core_coreg_planar_filter(as_points(points), as.double(min_planarity), if (is.null(voxel)) NULL else as.double(voxel),
                           as.double(k), if (is.null(radius)) NULL else as.double(radius))
}

# --------------------------------------------------------------------- ground

#' The terrain model coregistration measures heights from
#'
#' tlsalign's raster DTM: a low percentile of z per cell, deep pits
#' rejected, gaps filled from the nearest observed cell, a grey opening and
#' smoothing, then a slope limit. `ground_model()` builds one from its
#' pieces (`elevation[row, col]`, rows along y); `height_at()` interpolates
#' it bilinearly (clamped at the edges), `support()` says whether locations
#' were backed by ground returns, `normalise()` gives heights above ground
#' and `slope_deg()` the mean terrain slope, a sanity check on the fit.
#'
#' @param points n x 3 points, z up.
#' @param cell_size Resolution (m).
#' @param percentile Per-cell z percentile taken as ground.
#' @param max_slope Largest rise over run between neighbouring cells.
#' @param smooth_cells,opening_cells Windows (cells) of the smoothing and the
#'   grey opening.
#' @param min_points_per_cell Points a cell needs to count as observed.
#' @param pit_depth,pit_window Pit rejection.
#' @param max_points Fit from at most this many points (`NULL`: all).
#' @param seed Seed of that draw.
#' @param elevation Terrain height matrix.
#' @param origin `c(x0, y0)`, the centre of cell `[1, 1]` when sampling.
#' @param observed Logical matrix: cells backed by ground returns.
#' @param model A `sylva_ground_model`.
#' @param xy n x 2 locations.
#' @return `fit_ground()` and `ground_model()`: a `sylva_ground_model`.
#' @export
fit_ground <- function(points, cell_size = 0.5, percentile = 5, max_slope = 1, smooth_cells = 3, opening_cells = 5,
                       min_points_per_cell = 1, pit_depth = 3, pit_window = 9, max_points = 8000000, seed = 0) {
  g <- core_coreg_fit_ground(as_points(points), as.double(cell_size), as.double(percentile), as.double(max_slope),
                             as.double(smooth_cells), as.double(opening_cells), as.double(min_points_per_cell),
                             as.double(pit_depth), as.double(pit_window), as.double(or_else(max_points, 0)),
                             as.double(seed))
  ground_model(g$elevation, g$origin, cell_size, g$observed)
}

#' @rdname fit_ground
#' @export
ground_model <- function(elevation, origin, cell_size, observed) {
  elevation <- as.matrix(elevation)
  storage.mode(elevation) <- "double"
  observed <- as.matrix(observed)
  storage.mode(observed) <- "logical"
  structure(list(elevation = elevation, origin = as.double(origin), cell_size = as.double(cell_size),
                 observed = observed), class = "sylva_ground_model")
}

as_xy <- function(xy) {
  xy <- if (is.null(dim(xy))) matrix(as.double(xy), ncol = 2, byrow = TRUE) else as.matrix(xy)
  storage.mode(xy) <- "double"
  xy[, 1:2, drop = FALSE]
}

#' @rdname fit_ground
#' @export
height_at <- function(model, xy) {
  m <- unclass(model)
  core_coreg_ground_height(m$elevation, m$origin[1], m$origin[2], m$cell_size, as_xy(xy))
}

#' @rdname fit_ground
#' @export
support <- function(model, xy) {
  m <- unclass(model)
  core_coreg_ground_support(m$observed, m$origin[1], m$origin[2], m$cell_size, as_xy(xy))
}

#' @rdname fit_ground
#' @export
normalise <- function(model, points) {
  p <- as_points(points)
  if (!nrow(p)) return(double())
  p[, 3] - height_at(model, p[, 1:2, drop = FALSE])
}

#' @rdname fit_ground
#' @export
slope_deg <- function(model) {
  m <- unclass(model)
  core_coreg_ground_slope_deg(m$elevation, m$cell_size)
}

#' @export
print.sylva_ground_model <- function(x, ...) {
  m <- unclass(x)
  cat(sprintf("<sylva_ground_model> %d x %d cells of %.2f m, %.0f%% observed\n", nrow(m$elevation), ncol(m$elevation),
              m$cell_size, 100 * mean(m$observed)))
  invisible(x)
}
