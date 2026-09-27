# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

GROUND <- 2L

#' Classify ground points
#'
#' `classify_ground_csf()`: the Cloth Simulation Filter (Zhang et al. 2016).
#' The cloud is turned upside down and a cloth dropped onto it; points
#' within `class_threshold` of the settled cloth are ground. The default for
#' TLS plots; robust to understorey and moderate slopes.
#' `classify_ground_pmf()`: the Progressive Morphological Filter (Zhang et
#' al. 2003), which opens a minimum-height grid with growing windows; faster
#' but less robust under dense understorey.
#'
#' @param cloud Points in a frame with z up (project or projected
#'   coordinates, not a tilted scanner frame). Thin to 2-5 cm first for
#'   large plots.
#' @param cloth_resolution Cloth grid spacing (m).
#' @param rigidness 1 for steep terrain, 2 for moderate slopes, 3 for flat ground.
#' @param class_threshold Distance from the cloth (m) within which a point is ground.
#' @param iterations Maximum simulation steps.
#' @param time_step Simulation step.
#' @param cell_size Grid cell size (m).
#' @param max_window Largest window (m); should exceed the largest non-ground
#'   object footprint.
#' @param slope Terrain slope (rise over run) that grows the height
#'   threshold with window size.
#' @param initial_distance Height threshold (m) for the smallest window.
#' @param max_distance Upper limit on the height threshold (m).
#' @param return_mask Return the logical ground mask instead of the cloud.
#' @return The cloud with `classification` set to 2 (ground) or 1 (other),
#'   replacing any existing classification; or the mask (`TRUE` = ground).
#' @export
classify_ground_csf <- function(cloud, cloth_resolution = 0.5, rigidness = 2, class_threshold = 0.3,
                                iterations = 500, time_step = 0.65, return_mask = FALSE) {
  mask <- core_csf_ground_mask(xyz_of(cloud), as.double(cloth_resolution), as.integer(rigidness),
                               as.double(class_threshold), as.integer(iterations), as.double(time_step))
  if (return_mask) mask else with_class(cloud, mask)
}

#' @rdname classify_ground_csf
#' @export
classify_ground_pmf <- function(cloud, cell_size = 0.5, max_window = 10, slope = 0.3, initial_distance = 0.15,
                                max_distance = 2, return_mask = FALSE) {
  mask <- core_pmf_ground_mask(xyz_of(cloud), as.double(cell_size), as.double(max_window), as.double(slope),
                               as.double(initial_distance), as.double(max_distance))
  if (return_mask) mask else with_class(cloud, mask)
}

with_class <- function(cloud, mask) with_attrs(cloud, classification = ifelse(mask, GROUND, 1L))

#' Which points are ground
#'
#' @param cloud A cloud with a `classification` attribute (from a
#'   classifier here or from a LAS file).
#' @return `TRUE` where `classification == 2` (ASPRS ground).
#' @export
ground_mask <- function(cloud) {
  cl <- attrs_of(cloud)$classification
  if (is.null(cl)) stop("cloud has no 'classification' attribute; run classify_ground_* first", call. = FALSE)
  cl == GROUND
}

#' Digital terrain model from the ground points
#'
#' Each cell takes its lowest ground point. Empty cells (under stems,
#' behind occlusion) are filled from the nearest measured cells and
#' smoothed; measured cells keep their value.
#'
#' @param cloud A classified cloud (see `ground_mask()`).
#' @param resolution Cell size (m).
#' @param bounds `c(xmin, ymin, xmax, ymax)` of the grid; the extent of the
#'   ground points if `NULL`. Pass the plot extent to get identical grids
#'   across dates.
#' @return A `sylva_raster` of ground elevation with no missing cells.
#' @export
make_dtm <- function(cloud, resolution = 0.5, bounds = NULL) {
  g <- xyz_of(cloud)[ground_mask(cloud), , drop = FALSE]
  as_raster(core_make_dtm(g, as.double(resolution), if (is.null(bounds)) NULL else as.double(bounds)))
}

#' Height above ground
#'
#' `normalize_height()` adds z minus the interpolated DTM as an attribute,
#' keeping the coordinates; `flatten()` replaces z with it, for methods
#' that read z as height. Points outside the DTM take the edge value.
#'
#' @param cloud Points in the DTM's frame.
#' @param dtm Terrain from `make_dtm()`.
#' @param attr Name of the new attribute.
#' @return A `sylva_cloud`.
#' @export
normalize_height <- function(cloud, dtm, attr = "height") {
  a <- list(heights_above(cloud, dtm))
  names(a) <- attr
  do.call(with_attrs, c(list(cloud), a))
}

#' @rdname normalize_height
#' @export
flatten <- function(cloud, dtm) {
  xyz <- xyz_of(cloud)
  xyz[, 3] <- heights_above(cloud, dtm)
  as_cloud(list(xyz = xyz, attrs = attrs_of(cloud)))
}

heights_above <- function(cloud, dtm) do.call(core_heights_above, c(list(xyz_of(cloud)), raster_args(dtm)))

#' Canopy height model
#'
#' The highest point in each cell. No pit filling is applied, and TLS
#' under-samples the upper canopy far from the scanners.
#'
#' @param cloud Points with heights (see `heights()`).
#' @param resolution Cell size (m).
#' @param height_attr Attribute holding height above ground; z is used if absent.
#' @param bounds `c(xmin, ymin, xmax, ymax)`; the cloud extent if `NULL`.
#' @param min_height Cells with nothing above this height are 0.
#' @return A `sylva_raster` of maximum height per cell (m).
#' @export
make_chm <- function(cloud, resolution = 0.5, height_attr = "height", bounds = NULL, min_height = 0) {
  as_raster(core_make_chm(xyz_of(cloud), heights(cloud, height_attr), as.double(resolution),
                          if (is.null(bounds)) NULL else as.double(bounds), as.double(min_height)))
}
