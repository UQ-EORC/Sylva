# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

#' Joint multi-view refinement
#'
#' Solves all poses together from point-to-plane and stem correspondences,
#' the reference held fixed, as the Python package's
#' `sylva.coreg.refine_joint`. Levels run coarse to fine; at each,
#' correspondences are associated `rounds` times under the current poses and
#' each association is followed by `iterations` Gauss-Newton steps with Huber
#' reweighting. Stems are paired afresh at every association, mutually
#' nearest within `stem_radius` horizontally, and carry `stem_weight` of each
#' pair's weight. A weak prior ties every pose to its starting value, and a
#' step is kept only if it lowers the robust cost, capped at
#' `max_step_translation` and `max_step_rotation_deg`. Random subsets (above
#' `points_per_scan` and `correspondences_per_pair`) are NumPy's for the
#' same `seed`, so results equal the Python package's.
#'
#' @param points Per scan, its planar ICP points (n x 3) in its own frame.
#' @param poses Per scan, its current 4 x 4 `world_from_scan`.
#' @param edges Accepted pairs: a two-column matrix (or a list of pairs) of
#'   scan numbers.
#' @param stems Per scan, its stem positions (n x 3) in its own frame, or
#'   `NULL`.
#' @param reference Scan held fixed.
#' @param voxel_sizes,max_distances Voxel size and correspondence cut-off of
#'   each level (m).
#' @param rounds,iterations Associations per level and steps per association.
#' @param points_per_scan,correspondences_per_pair Random caps.
#' @param min_planarity,normal_neighbours,max_normal_angle_deg Correspondence
#'   gates.
#' @param stem_weight,stem_radius,stem_scale Stem constraints.
#' @param min_voxel_points Voxels with fewer points are dropped.
#' @param robust_scale Lower bound (m) of the Huber scale.
#' @param prior_translation,prior_rotation_deg One-sigma pose priors.
#' @param max_step_translation,max_step_rotation_deg Step caps.
#' @param seed Seed of the random caps.
#' @param log Called with each progress message, or `NULL`.
#' @return A `sylva_joint_refinement`: `poses`, `shifts` (m), `rotations`
#'   (degrees), `residual_before`, `residual_after` (median absolute
#'   point-to-plane residual, m) and `correspondences`.
#' @export
refine_joint <- function(points, poses, edges, stems, reference, voxel_sizes = c(0.10, 0.05, 0.03),
                         max_distances = c(0.30, 0.15, 0.08), rounds = 3, iterations = 3, points_per_scan = 400000,
                         correspondences_per_pair = 20000, min_planarity = 0.25, normal_neighbours = 20,
                         max_normal_angle_deg = 45, stem_weight = 0.05, stem_radius = 0.15, stem_scale = 0.03,
                         min_voxel_points = 1, robust_scale = 0.02, prior_translation = 0.03, prior_rotation_deg = 0.3,
                         max_step_translation = 0.05, max_step_rotation_deg = 0.5, seed = 0, log = NULL) {
  n <- length(poses)
  if (length(points) < n) stop("refine_joint needs the points of every scan")
  if (is.list(edges) && !is.data.frame(edges)) edges <- do.call(rbind, lapply(edges, as.double))
  edges <- matrix(as.double(edges), ncol = 2)
  pts <- lapply(points[seq_len(n)], function(p) if (length(p)) as_points(p) else matrix(0, 0, 3))
  st <- lapply(stems, function(s) if (is.null(s) || !length(s)) matrix(0, 0, 3) else as_points(s, "stems"))
  r <- core_coreg_refine_joint(pts, lapply(poses, as_transform), edges[, 1] - 1, edges[, 2] - 1, st,
                               as.double(reference) - 1, as.double(voxel_sizes), as.double(max_distances),
                               as.double(rounds), as.double(iterations), as.double(points_per_scan),
                               as.double(correspondences_per_pair), as.double(min_planarity),
                               as.double(normal_neighbours), as.double(max_normal_angle_deg), as.double(stem_weight),
                               as.double(stem_radius), as.double(stem_scale), as.double(min_voxel_points),
                               as.double(robust_scale), as.double(prior_translation), as.double(prior_rotation_deg),
                               as.double(max_step_translation), as.double(max_step_rotation_deg), as.double(seed),
                               log)
  r$correspondences <- as.integer(r$correspondences)
  structure(r, class = "sylva_joint_refinement")
}

#' @export
print.sylva_joint_refinement <- function(x, ...) {
  cat(sprintf("<sylva_joint_refinement> %d scans, largest shift %.3f m, median residual %.2f -> %.2f cm\n",
              length(x$poses), max(c(0, x$shifts)), x$residual_before * 100, x$residual_after * 100))
  invisible(x)
}
