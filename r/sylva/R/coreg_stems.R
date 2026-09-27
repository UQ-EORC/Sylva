# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

stem_columns <- c("x", "y", "z", "dbh", "axis_x", "axis_y", "axis_z", "reference_height", "n_slices", "n_points",
                  "rmse", "coverage", "lean_deg")

stem_frame <- function(l) {
  d <- as.data.frame(l[stem_columns])
  d$n_slices <- as.integer(d$n_slices)
  d$n_points <- as.integer(d$n_points)
  d
}

#' Stems and stem maps
#'
#' A stem map is the stems of one scan, the features a coarse alignment is
#' found from: a `sylva_stem_map` holding `stems`, a data frame with one row
#' per stem (the fields of the Python package's `Stem`: `x`, `y`, `z` at
#' `reference_height` above the ground in the scan's own frame, `dbh`, the
#' upward axis `axis_x`, `axis_y`, `axis_z`, `n_slices`, `n_points`,
#' `rmse`, `coverage`, `lean_deg`), a `name` and an optional terrain model.
#' `coreg_stem()` builds stem rows from vectors (`stem` is taken by
#' graphics); `stem_map()` wraps them.
#'
#' `positions()`, `xy()`, `diameters()`, `qualities()` and `axes()` read a
#' map; `sorted_by_quality()` puts the best first, `top()` keeps the `n`
#' best and `transformed()` moves every stem by a rigid transform.
#' `save_stem_map()` writes the JSON of the Python package's
#' `StemMap.save()` and `load_stem_map()` reads it.
#'
#' Equal qualities keep their order in `sorted_by_quality()`; NumPy's sort
#' (which the Python package uses) does not promise that, so maps whose
#' stems tie may be ordered differently in the two languages.
#'
#' @param x,y,z,dbh Stem position and diameter (m).
#' @param axis Upward axis, a 3-vector (or a 3-column matrix, one row per stem).
#' @param reference_height,n_slices,n_points,rmse,coverage,lean_deg Stem fit.
#' @param stems A data frame of stems (from `coreg_stem()`).
#' @param name Name of the map.
#' @param ground A terrain model ([fit_ground()]) or `NULL`.
#' @param map A `sylva_stem_map`.
#' @param n Stems to keep.
#' @param T A 4 x 4 transform.
#' @param path The JSON file.
#' @export
coreg_stem <- function(x, y, z, dbh, axis = c(0, 0, 1), reference_height = 1.3, n_slices = 0L, n_points = 0L, rmse = 0,
                 coverage = 0, lean_deg = 0) {
  k <- length(x)
  axis <- if (k) matrix(as.double(axis), ncol = 3, nrow = k, byrow = is.null(dim(axis))) else matrix(0, 0, 3)
  stem_frame(list(x = as.double(x), y = rep_len(as.double(y), k), z = rep_len(as.double(z), k),
                  dbh = rep_len(as.double(dbh), k), axis_x = axis[, 1], axis_y = axis[, 2], axis_z = axis[, 3],
                  reference_height = rep_len(as.double(reference_height), k), n_slices = rep_len(n_slices, k),
                  n_points = rep_len(n_points, k), rmse = rep_len(as.double(rmse), k),
                  coverage = rep_len(as.double(coverage), k), lean_deg = rep_len(as.double(lean_deg), k)))
}

#' @rdname coreg_stem
#' @export
stem_map <- function(stems = NULL, name = "", ground = NULL) {
  if (is.null(stems)) stems <- coreg_stem(double(), double(), double(), double())
  structure(list(stems = stem_frame(stems), name = as.character(name), ground = ground), class = "sylva_stem_map")
}

#' @export
length.sylva_stem_map <- function(x) nrow(unclass(x)$stems)

#' @export
print.sylva_stem_map <- function(x, ...) {
  cat(sprintf("<sylva_stem_map> name=%s, %d stems\n", dQuote(unclass(x)$name, FALSE), length(x)))
  invisible(x)
}

#' @export
`[.sylva_stem_map` <- function(x, i) {
  m <- unclass(x)
  m$stems <- m$stems[i, , drop = FALSE]
  rownames(m$stems) <- NULL
  structure(m, class = "sylva_stem_map")
}

#' @rdname coreg_stem
#' @export
positions <- function(map) {
  s <- unclass(map)$stems
  cbind(s$x, s$y, s$z)
}

#' @rdname coreg_stem
#' @export
xy <- function(map) positions(map)[, 1:2, drop = FALSE]

#' @rdname coreg_stem
#' @export
diameters <- function(map) unclass(map)$stems$dbh

#' @rdname coreg_stem
#' @export
qualities <- function(map) {
  s <- unclass(map)$stems
  core_coreg_stem_quality(as.double(s$rmse), as.double(s$coverage), as.double(s$n_slices))
}

#' @rdname coreg_stem
#' @export
axes <- function(map) {
  s <- unclass(map)$stems
  cbind(s$axis_x, s$axis_y, s$axis_z)
}

#' @rdname coreg_stem
#' @export
sorted_by_quality <- function(map) map[order(-qualities(map))]

#' @rdname coreg_stem
#' @export
top <- function(map, n) {
  s <- sorted_by_quality(map)
  s[seq_len(min(n, length(s)))]
}

#' @rdname coreg_stem
#' @export
transformed <- function(map, T) {
  m <- unclass(map)
  if (!nrow(m$stems)) return(stem_map(name = m$name))
  p <- transform_points(T, positions(map))
  a <- transform_vectors(T, axes(map))
  s <- m$stems
  s$x <- p[, 1]
  s$y <- p[, 2]
  s$z <- p[, 3]
  s$axis_x <- a[, 1]
  s$axis_y <- a[, 2]
  s$axis_z <- a[, 3]
  stem_map(s, name = m$name)
}

#' @rdname coreg_stem
#' @export
save_stem_map <- function(map, path) {
  m <- unclass(map)
  core_coreg_write_stem_map(path.expand(path), m$name, as.list(m$stems))
  invisible(path)
}

#' @rdname coreg_stem
#' @export
load_stem_map <- function(path) {
  r <- core_coreg_read_stem_map(path.expand(path))
  stem_map(stem_frame(r$stems), name = r$name)
}

#' A stem map from plain vectors
#'
#' For tests and external stem lists: every stem gets 5 slices, 200 points,
#' a 5 mm residual and half its circumference seen.
#'
#' @param xy n x 2 positions.
#' @param dbh Diameters (m); 0.3 if `NULL`.
#' @param z Elevations; 0 if `NULL`.
#' @param name Name of the map.
#' @return A `sylva_stem_map`.
#' @export
stem_map_from_arrays <- function(xy, dbh = NULL, z = NULL, name = "") {
  xy <- matrix(as.double(xy), ncol = 2)
  k <- nrow(xy)
  stem_map(coreg_stem(xy[, 1], xy[, 2], if (is.null(z)) rep(0, k) else z, if (is.null(dbh)) rep(0.3, k) else dbh,
                n_slices = 5L, n_points = 200L, rmse = 0.005, coverage = 0.5), name = name)
}

#' Settings of stem detection
#'
#' The defaults are tlsalign's and suit plot-scale TLS (10 to 30 m range,
#' stems 5 to 120 cm). In dense understorey raise `min_slices` and
#' `min_coverage`; for buttressed stems raise `slice_min_height` above the
#' buttresses.
#'
#' @param slice_min_height,slice_max_height,slice_thickness,slice_step Stem
#'   band and slices (m above ground).
#' @param reference_height Height of the reported positions and diameters.
#' @param min_radius,max_radius Stem radius range (m).
#' @param cluster_cell,min_cluster_points,max_cluster_extent Slice clustering.
#' @param ransac_iterations,ransac_tolerance,max_circles_per_cluster,min_circle_inliers,min_coverage,max_circle_rmse Circle fits.
#' @param link_radius,link_radius_ratio,min_slices,max_lean_deg Linking
#'   circles up a stem.
#' @param seed Seed of the circle fits.
#' @return A `sylva_stem_detection_config`.
#' @export
stem_detection_config <- function(slice_min_height = 1.0, slice_max_height = 5.0, slice_thickness = 0.3,
                                  slice_step = 0.4, reference_height = 1.3, min_radius = 0.025, max_radius = 0.60,
                                  cluster_cell = 0.06, min_cluster_points = 12, max_cluster_extent = 2.0,
                                  ransac_iterations = 120, ransac_tolerance = 0.02, max_circles_per_cluster = 3,
                                  min_circle_inliers = 10, min_coverage = 0.12, max_circle_rmse = 0.02,
                                  link_radius = 0.20, link_radius_ratio = 0.45, min_slices = 3, max_lean_deg = 25.0,
                                  seed = 0) {
  structure(as.list(environment()), class = "sylva_stem_detection_config")
}

#' Detect the stems of one scan
#'
#' Horizontal slices through the stem band are clustered and fitted with
#' RANSAC circles, the circles are linked up the stem, and each chain gives
#' an axis, a diameter at breast height and a quality: the Python package's
#' `sylva.coreg.detect_stems` (tlsalign's detector).
#'
#' @param points n x 3 points in the scan's own (levelled) frame.
#' @param ground Terrain model ([fit_ground()]); fitted if `NULL`.
#' @param config [stem_detection_config()].
#' @param name Name of the map.
#' @param heights Heights above ground of `points`, if already known.
#' @return A `sylva_stem_map`, best first, with `z` on the scan's own terrain.
#' @export
coreg_detect_stems <- function(points, ground = NULL, config = NULL, name = "", heights = NULL) {
  cfg <- unclass(or_else(config, stem_detection_config()))
  p <- as_points(points)
  if (nrow(p) < 100) return(stem_map(name = name, ground = ground))
  if (is.null(heights)) {
    if (is.null(ground)) ground <- fit_ground(p)
    heights <- normalise(ground, p)
  }
  s <- stem_frame(core_coreg_detect_stems(p, as.double(heights), cfg))
  if (!is.null(ground) && nrow(s)) s$z <- height_at(ground, cbind(s$x, s$y)) + cfg$reference_height
  sorted_by_quality(stem_map(s, name = name, ground = ground))
}

#' Settings of stem-map matching
#'
#' The defaults are tlsalign's; see the Python package's `MatchConfig`.
#'
#' @param min_pair_distance,max_pair_distance Stem pairs outside this range
#'   (m) are ignored.
#' @param pair_distance_tolerance Tolerance on pair separation (m).
#' @param inlier_tolerance Horizontal distance (m) at which two stems count
#'   as the same tree.
#' @param diameter_rel_tolerance,diameter_abs_tolerance,use_diameters
#'   Diameter agreement.
#' @param max_stems The best stems of each scan taken into matching.
#' @param max_hypotheses,min_inliers,early_exit_inliers Search budget and
#'   acceptance.
#' @param distinct_translation,distinct_yaw_deg How different a rival must be.
#' @param refine_iterations Refinement passes of the winner.
#' @param seed Unused (kept for the Python signature).
#' @return A `sylva_match_config`.
#' @export
match_config <- function(min_pair_distance = 2.0, max_pair_distance = 35.0, pair_distance_tolerance = 0.25,
                         inlier_tolerance = 0.40, diameter_rel_tolerance = 0.30, diameter_abs_tolerance = 0.04,
                         use_diameters = TRUE, max_stems = 70, max_hypotheses = 60000, min_inliers = 4,
                         early_exit_inliers = 40, distinct_translation = 1.0, distinct_yaw_deg = 5.0,
                         refine_iterations = 6, seed = 0) {
  structure(as.list(environment()), class = "sylva_match_config")
}

stem_match <- function(m) {
  m$correspondences <- m$correspondences + 1
  storage.mode(m$correspondences) <- "integer"
  colnames(m$correspondences) <- c("source", "target")
  for (k in c("n_inliers", "n_source", "n_target")) m[[k]] <- as.integer(m[[k]])
  if (!is.null(m$rival)) m$rival <- stem_match(m$rival)
  structure(m, class = "sylva_match_result")
}

#' Align two stem maps with no initial guess
#'
#' Yaw and translation aligning `source` stems onto `target` stems, found by
#' a near-exhaustive search over stem pairs of equal separation.
#'
#' @param source,target `sylva_stem_map`s.
#' @param config [match_config()].
#' @param result A `sylva_match_result`.
#' @return A `sylva_match_result`: `transform` (source into target),
#'   `n_inliers`, `inlier_rmse`, `score`, `correspondences` (1-based stem
#'   rows), `n_source`, `n_target`, `success`, `ambiguity` and `rival`.
#'   `inlier_fraction()` is the inliers over the smaller map.
#' @export
match_stem_maps <- function(source, target, config = NULL) {
  cfg <- unclass(or_else(config, match_config()))
  cfg$use_diameters <- isTRUE(cfg$use_diameters)
  stem_match(core_coreg_match_stem_maps(positions(source), as.double(diameters(source)), qualities(source),
                                        positions(target), as.double(diameters(target)), qualities(target), cfg))
}

#' @rdname match_stem_maps
#' @export
inlier_fraction <- function(result) result$n_inliers / max(min(result$n_source, result$n_target), 1)

#' @export
print.sylva_match_result <- function(x, ...) {
  cat(sprintf("<sylva_match_result> success=%s, inliers=%d, rmse=%.3f m, ambiguity=%.2f\n", x$success, x$n_inliers,
              x$inlier_rmse, x$ambiguity))
  invisible(x)
}
