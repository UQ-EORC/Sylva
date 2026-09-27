# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

#' Synthetic scenes
#'
#' Small scenes for examples, tutorials and tests, as the Python package's
#' `sylva.synthetic` (`terrain_height`, `tree`, `leaf_area`, `forest` and
#' `scan`, prefixed here with `synthetic_`). Nothing here is a forest model;
#' the shapes are simple enough that the right answer is known (stem
#' positions, diameters, heights, leaf area). The random draws are those of
#' the Python package, so a seed gives the same scene, to the bit, in both
#' languages.
#'
#' `synthetic_terrain_height()` is the ground of the scenes, `slope * x +
#' 0.2 * sin(y / 3)`. `synthetic_tree()` is one tree: a tapered stem,
#' `n_branches` limbs in the upper half and small flat leaf discs (radius
#' `SYNTHETIC_LEAF_RADIUS`, 12 points each) around the limb ends, with
#' `classification` 5 for wood and 4 for leaves. `synthetic_leaf_area()` is
#' the true one-sided leaf area of such a cloud. `synthetic_forest()` puts
#' trees on sloped terrain (z is not normalised), with `classification` (2
#' ground, 4 leaf, 5 wood) and `tree_id` (0 for ground, then 1.. in row
#' order) as ground truth; the ground extends `margin` m beyond the `size` m
#' square.
#'
#' @param x,y Coordinates (m); for `synthetic_tree()` the stem base.
#' @param slope Rise per metre along x.
#' @param dbh Diameter at the base (m); the stem tapers to a quarter of it.
#' @param height Tree height (m).
#' @param z0 Stem base elevation.
#' @param n_branches Limbs.
#' @param leaf_points Approximate number of leaf points (12 per leaf disc).
#' @param seed Random seed; each tree of a forest gets `seed + i`.
#' @param cloud A `sylva_cloud` from `synthetic_tree()` or `synthetic_forest()`.
#' @param trees A data frame or matrix with columns `x`, `y`, `dbh` and
#'   `height`, one row per tree; `SYNTHETIC_DEFAULT_TREES` if `NULL`.
#' @param size Side of the plot square (m).
#' @param ground_points Points on the terrain.
#' @param margin Terrain beyond the plot edge (m).
#' @return `synthetic_terrain_height()`: heights, the shape of `x` and `y`
#'   recycled; `synthetic_leaf_area()`: m²; otherwise a `sylva_cloud`.
#' @export
synthetic_terrain_height <- function(x, y, slope = 0.05) {
  n <- max(length(x), length(y))
  z <- core_synthetic_terrain_height(rep_len(as.double(x), n), rep_len(as.double(y), n), as.double(slope))
  if (!is.null(dim(x))) dim(z) <- dim(x) else if (!is.null(dim(y))) dim(z) <- dim(y)
  z
}

#' @rdname synthetic_terrain_height
#' @export
synthetic_tree <- function(x = 0, y = 0, dbh = 0.3, height = 12, z0 = 0, n_branches = 6, leaf_points = 18000,
                           seed = 0) {
  as_cloud(core_synthetic_tree(as.double(x), as.double(y), as.double(dbh), as.double(height), as.double(z0),
                               as.double(n_branches), as.double(leaf_points), as.double(seed)))
}

#' @rdname synthetic_terrain_height
#' @export
synthetic_leaf_area <- function(cloud) {
  core_synthetic_leaf_area(as.double(attrs_of(cloud)$classification))
}

#' @rdname synthetic_terrain_height
#' @export
synthetic_forest <- function(trees = NULL, size = 20, ground_points = 40000, margin = 4, seed = 0) {
  if (is.null(trees)) trees <- SYNTHETIC_DEFAULT_TREES
  trees <- as.data.frame(trees)
  if (!nrow(trees)) trees <- data.frame(x = double(), y = double(), dbh = double(), height = double())
  if (is.null(names(trees)) || !all(c("x", "y", "dbh", "height") %in% names(trees))) {
    names(trees)[1:4] <- c("x", "y", "dbh", "height")
  }
  as_cloud(core_synthetic_forest(as.double(trees$x), as.double(trees$y), as.double(trees$dbh), as.double(trees$height),
                                 as.double(size), as.double(ground_points), as.double(margin), as.double(seed)))
}

#' A pseudo terrestrial scan
#'
#' Pulses are fired from `origin` on a regular zenith / azimuth grid. The
#' points falling in a pulse's angular cell are its candidate targets: the
#' nearest gives the first echo, and further ones at least
#' `echo_separation` m apart give up to `max_echoes` echoes. Cells without a
#' point are pulses with no return, which carry the free-space information
#' ray tracing needs. Echo attributes are copied from the points, so
#' `classification` and `tree_id` survive. As the Python package's
#' `sylva.synthetic.scan`.
#'
#' @param cloud Scene to scan, e.g. from `synthetic_forest()`.
#' @param origin Scanner position.
#' @param resolution_deg Angular step in zenith and azimuth (degrees).
#' @param max_zenith_deg Pulses are fired from straight up to this zenith.
#' @param max_echoes Echoes per pulse.
#' @param echo_separation Minimum range (m) between echoes of one pulse.
#' @return A `sylva_shots`, one pulse per angular cell, misses included.
#' @export
synthetic_scan <- function(cloud, origin = c(10, 10, 1.5), resolution_deg = 0.25, max_zenith_deg = 130, max_echoes = 2,
                           echo_separation = 0.5) {
  as_shots(core_synthetic_scan(unclass(cloud), as.double(origin), as.double(resolution_deg), as.double(max_zenith_deg),
                               as.double(max_echoes), as.double(echo_separation)))
}

#' @rdname synthetic_terrain_height
#' @format `SYNTHETIC_DEFAULT_TREES`: the four trees of `synthetic_forest()`.
#'   `SYNTHETIC_LEAF_RADIUS`: radius (m) of the leaf discs.
#' @export
SYNTHETIC_DEFAULT_TREES <- data.frame(x = c(5, 14, 8, 15.5), y = c(5, 6, 15, 15), dbh = c(0.30, 0.20, 0.45, 0.25),
                                      height = c(12, 9, 15, 11))

#' @rdname synthetic_terrain_height
#' @export
SYNTHETIC_LEAF_RADIUS <- 0.08
