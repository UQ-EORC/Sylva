# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

reflector_frame <- function(l) {
  data.frame(x = l$x, y = l$y, z = l$z, reflectance = l$reflectance, diameter = l$diameter,
             n_points = as.integer(l$n_points), name = as.character(l$name), stringsAsFactors = FALSE)
}

#' Retro-reflective targets
#'
#' Targets are a data frame with one row per target and columns `x`, `y`,
#' `z` (the scan's own frame), `reflectance`, `diameter`, `n_points` and
#' `name`, the fields of the Python package's `Reflector`. `reflector()`
#' builds one from vectors. `read_tiepoint_list()` reads a RIEGL `.tpl`
#' tie-point list (JSON), `read_reflector_list()` a RiSCAN PRO `.rfl`
#' reflector list; both give no rows for a missing or unreadable file.
#'
#' @param x,y,z Target positions.
#' @param reflectance,diameter,n_points,name Target attributes.
#' @param path The file.
#' @return A data frame of targets.
#' @export
reflector <- function(x, y, z, reflectance = NaN, diameter = NaN, n_points = 0L, name = "") {
  reflector_frame(list(x = as.double(x), y = as.double(y), z = as.double(z),
                       reflectance = rep_len(as.double(reflectance), length(x)),
                       diameter = rep_len(as.double(diameter), length(x)),
                       n_points = rep_len(n_points, length(x)), name = rep_len(as.character(name), length(x))))
}

#' @rdname reflector
#' @export
read_tiepoint_list <- function(path) reflector_frame(core_coreg_read_tiepoint_list(path.expand(path)))

#' @rdname reflector
#' @export
read_reflector_list <- function(path) reflector_frame(core_coreg_read_reflector_list(path.expand(path)))

#' Find retro-reflective targets by their return strength
#'
#' Returns at least `min_reflectance` bright are clustered single-link at
#' `cluster_radius`; clusters of `min_points` or more and no wider than
#' `max_extent` are targets, each placed at its reflectance-weighted
#' centroid. `min_reflectance` is the setting that matters: in a scan with
#' targets the reflectance histogram is bimodal, so check it first.
#'
#' @param xyz n x 3 points (or a `sylva_cloud`).
#' @param reflectance Reflectance (dB) of each point; `NULL` finds nothing.
#' @param min_reflectance,cluster_radius,min_points,max_extent Detection
#'   settings.
#' @return A data frame of targets (see [reflector()]).
#' @export
detect_reflectors <- function(xyz, reflectance, min_reflectance = 5, cluster_radius = 0.15, min_points = 8,
                              max_extent = 0.5) {
  p <- as_points(xyz, "xyz")
  if (is.null(reflectance) || nrow(p) == 0) return(reflector(double(), double(), double()))
  reflector_frame(core_coreg_detect_reflectors(p, as.double(reflectance), as.double(min_reflectance),
                                               as.double(cluster_radius), as.double(min_points),
                                               as.double(max_extent)))
}

target_positions <- function(t) {
  if (is.data.frame(t)) return(as_points(as.matrix(t[, c("x", "y", "z")])))
  as_points(t)
}

#' Align two target sets with no initial guess
#'
#' Triangles of source targets are searched exhaustively against congruent
#' triangles of target targets; three correspondences fix all six degrees of
#' freedom, and each candidate is scored by how many targets it brings
#' within `tolerance` of a partner.
#'
#' When several triangles bring the same targets into agreement, their fits
#' differ only by rounding and the one kept is decided by it, so the order
#' of `correspondences` (not their set, nor the transform) can differ from
#' the Python package's.
#'
#' @param source,target Targets (data frames from [reflector()] or n x 3
#'   matrices).
#' @param tolerance How far (m) a matched target may sit from its partner.
#' @param min_inliers Targets that must correspond.
#' @param distance_tolerance Agreement (m) of triangle sides.
#' @return A `sylva_reflector_match`: `transform` (source into target),
#'   `n_inliers`, `rmse`, `correspondences` (1-based `source`, `target` rows)
#'   and `success`.
#' @export
match_reflectors <- function(source, target, tolerance = 0.05, min_inliers = 3, distance_tolerance = 0.03) {
  m <- core_coreg_match_reflectors(target_positions(source), target_positions(target), as.double(tolerance),
                                   as.double(min_inliers), as.double(distance_tolerance))
  m$correspondences <- m$correspondences + 1
  storage.mode(m$correspondences) <- "integer"
  colnames(m$correspondences) <- c("source", "target")
  m$n_inliers <- as.integer(m$n_inliers)
  structure(m, class = "sylva_reflector_match")
}

#' @export
print.sylva_reflector_match <- function(x, ...) {
  cat(sprintf("<sylva_reflector_match> success=%s, inliers=%d, rmse=%.1f mm\n", x$success, x$n_inliers,
              x$rmse * 1000))
  invisible(x)
}
