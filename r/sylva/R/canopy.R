# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# Arrays keep the Python package's dimension order and cross to Rust as flat
# row-major vectors.
row_major <- function(a) if (is.null(dim(a))) as.double(a) else as.double(aperm(a, rev(seq_along(dim(a)))))
from_row_major <- function(v, dims) aperm(array(v, rev(dims)), rev(seq_along(dims)))

# ------------------------------------------------------------------ gap profiles

#' Gap probability profiles
#'
#' Gap probability by zenith ring, azimuth sector and height above ground,
#' pooled over scans (Jupp et al. 2009), as the Python package's
#' `GapProfile`. Start with `gap_profile()`, add each scan position with
#' `add_scan()`, then read `report()`.
#'
#' @param zenith_edges Ring edges (degrees from up).
#' @param n_azimuth Azimuth sectors per ring (for the clumping index).
#' @param height_bin Height resolution (m).
#' @param max_height Top of the profile (m).
#' @return A `sylva_gap_profile`: `hits[ring, sector, height]`,
#'   `shots[ring, sector]` and the same per scan.
#' @export
gap_profile <- function(zenith_edges = seq(5, 70, by = 5), n_azimuth = 36, height_bin = 0.5, max_height = 80) {
  nr <- length(zenith_edges) - 1
  nh <- ceiling(max_height / height_bin)
  structure(list(zenith_edges = as.double(zenith_edges), n_azimuth = as.integer(n_azimuth),
                 height_bin = as.double(height_bin), hits = array(0, c(nr, n_azimuth, nh)),
                 shots = matrix(0, nr, n_azimuth), scan_hits = list(), scan_shots = list(),
                 scan_low = list(), min_height = 0.5),
            class = "sylva_gap_profile")
}

#' Add a scan position to a gap profile
#'
#' @param profile A `sylva_gap_profile`.
#' @param shots The position's pulses (scanner or project frame).
#' @param echo_heights Height above ground of each echo.
#' @param fired_per_ring Pulses fired into each ring, for streams without
#'   the pulses that returned nothing (`fired_pulses_per_ring()`,
#'   `fired_pulses_from_points()`); `NULL` if `shots` include them.
#' @param min_height Echoes below this height are dropped.
#' @return The profile with the scan added.
#' @export
add_scan <- function(profile, shots, echo_heights, fired_per_ring = NULL, min_height = -Inf) {
  p <- unclass(profile)
  d <- dim(p$hits)
  h <- core_pgap_histogram(unclass(shots), as.double(echo_heights), p$zenith_edges, p$n_azimuth,
                           p$height_bin, d[3], as.double(min_height),
                           if (is.null(fired_per_ring)) NULL else as.double(fired_per_ring))
  hits <- from_row_major(h$hits, d)
  sh <- matrix(h$shots, d[1], d[2], byrow = TRUE)
  first <- floor(p$min_height / p$height_bin + 1e-9)
  p$hits <- p$hits + hits
  p$shots <- p$shots + sh
  p$scan_hits[[length(p$scan_hits) + 1]] <- apply(hits, c(1, 2), sum)
  p$scan_shots[[length(p$scan_shots) + 1]] <- sh
  p$scan_low[[length(p$scan_low) + 1]] <- if (first > 0) apply(hits[, , seq_len(first), drop = FALSE], c(1, 2), sum) else sh * 0
  structure(p, class = "sylva_gap_profile")
}

gap_args <- function(profile) {
  p <- unclass(profile)
  d <- dim(p$hits)
  list(p$zenith_edges, p$n_azimuth, p$height_bin, d[3], p$min_height, row_major(p$hits), row_major(p$shots))
}

scan_args <- function(profile) {
  p <- unclass(profile)
  low <- if (length(p$scan_low) == length(p$scan_hits)) lapply(p$scan_low, row_major) else list()
  list(lapply(p$scan_hits, row_major), lapply(p$scan_shots, row_major), low)
}

#' Gap probability, plant area and clumping from a gap profile
#'
#' `pgap()`: gap probability `[ring, height]`. `pai_profile()`: cumulative
#' plant area index below each height by the hinge (57.5 degrees), linear
#' (Jupp et al. 2009) or weighted (Miller) method; `pavd_profile()` its
#' height derivative. `clumping()`: the Lang and Xiang (1986) index over
#' scan-sector segments. `report()`: the plot summary.
#'
#' @param profile A `sylva_gap_profile`.
#' @param method `"hinge"`, `"linear"` or `"weighted"`.
#' @param zenith Zenith angle (degrees) of the ring used for clumping.
#' @param top_fraction Fraction of total PAI that sets canopy height.
#' @param saturation_gap Gap at 57.5 degrees below which the PAI is flagged
#'   as saturated (bounded by the pulse count, not measured).
#' @export
pgap <- function(profile) do.call(core_gap_pgap, gap_args(profile))

#' @rdname pgap
#' @export
pai_profile <- function(profile, method = c("hinge", "linear", "weighted")) {
  do.call(core_gap_pai_profile, c(gap_args(profile), list(match.arg(method), FALSE)))
}

#' @rdname pgap
#' @export
pavd_profile <- function(profile, method = c("hinge", "linear", "weighted")) {
  do.call(core_gap_pai_profile, c(gap_args(profile), list(match.arg(method), TRUE)))
}

#' @rdname pgap
#' @export
clumping <- function(profile, zenith = 57.5) {
  p <- unclass(profile)
  do.call(core_gap_clumping, c(list(p$zenith_edges, p$n_azimuth), scan_args(profile), list(as.double(zenith))))
}

#' Summaries of Sylva results
#'
#' @param x A result object (e.g. a `sylva_gap_profile`).
#' @param ... Passed to the method.
#' @export
report <- function(x, ...) UseMethod("report")

#' @rdname pgap
#' @export
report.sylva_gap_profile <- function(x, top_fraction = 0.99, saturation_gap = 0.005, ...) {
  do.call(core_gap_report, c(gap_args(x), scan_args(x), list(as.double(top_fraction), as.double(saturation_gap))))
}

#' @export
print.sylva_gap_profile <- function(x, ...) {
  p <- unclass(x)
  cat(sprintf("<sylva_gap_profile> %d rings x %d sectors x %d heights, %d scans, %s pulses\n",
              dim(p$hits)[1], dim(p$hits)[2], dim(p$hits)[3], length(p$scan_hits),
              format(sum(p$shots), big.mark = ",")))
  invisible(x)
}

# ---------------------------------------------------------------- fired pulses

scan_pattern <- function(pattern) lapply(pattern[c("theta_start", "theta_delta", "theta_count", "phi_count")], as.double)

#' Pulses fired into each zenith ring
#'
#' For RIEGL streams, which hold only the pulses that returned something.
#' `fired_pulses_per_ring()` uses the scan pattern (from a RiSCAN project)
#' and counts pulses per line on the downward lines; `fired_pulses_from_points()`
#' needs only the returns, for scans known by their points and SOP.
#' `shots_scanner` must be in the scanner frame.
#'
#' @param shots_scanner A `sylva_shots` in the scanner frame.
#' @param pattern Scan pattern: a list with `theta_start`, `theta_delta`,
#'   `theta_count` and `phi_count`.
#' @param zenith_edges Ring edges (degrees).
#' @param shot_stride The `shot_stride` the shots were read with.
#' @param ground_zenith Zenith range (degrees) where every pulse returns.
#' @param limit_quantile Upper quantile of the returns' zenith taken as the scan's lower limit.
#' @param field_of_view The scanner's vertical field of view (degrees; 100 for
#'   RIEGL's VZ scanners), which sets the upper limit; `NULL` reads it from the
#'   returns, right only where the canopy returns the most upward pulses.
#' @export
fired_pulses_per_ring <- function(shots_scanner, pattern, zenith_edges, shot_stride = 1, ground_zenith = c(100, 125)) {
  core_fired_pulses_per_ring(unclass(shots_scanner)$direction, scan_pattern(pattern), as.double(zenith_edges),
                             as.integer(shot_stride), as.double(ground_zenith))
}

#' @rdname fired_pulses_per_ring
#' @export
fired_pulses_from_points <- function(shots_scanner, zenith_edges, ground_zenith = c(100, 125), limit_quantile = 1e-5,
                                     field_of_view = 100) {
  core_fired_pulses_from_points(unclass(shots_scanner)$direction, as.double(zenith_edges), as.double(ground_zenith),
                                as.double(limit_quantile), if (is.null(field_of_view)) NULL else as.double(field_of_view))
}

#' @rdname fired_pulses_per_ring
#' @param quantile Quantile of per-line counts taken as fully sampled.
#' @export
pulses_per_line <- function(shots_scanner, pattern, quantile = 0.98, shot_stride = 1) {
  as.integer(core_pulses_per_line(unclass(shots_scanner)$direction, scan_pattern(pattern), as.double(quantile),
                                  as.integer(shot_stride)))
}

#' @rdname fired_pulses_per_ring
#' @param pulses_per_line Pulses per zenith line; the pattern's `phi_count` if `NULL`.
#' @export
expected_per_zenith <- function(pattern, zenith_edges, pulses_per_line = NULL) {
  round(core_expected_per_zenith(scan_pattern(pattern), as.double(zenith_edges), pulses_per_line))
}

#' Gap fraction by zenith ring
#'
#' `gap_fraction_zenith()` needs every fired pulse (a ray cloud, or shots
#' read with their misses); `gap_fraction_pattern()` takes the fired pulses
#' from the scan pattern.
#'
#' @param shots A `sylva_shots` of one scan position.
#' @param echo_heights Height above ground of each echo.
#' @param pattern Scan pattern (see `fired_pulses_per_ring()`).
#' @param min_height Echoes at or below this height do not block a pulse.
#' @param zenith_edges Ring edges (degrees); 0-90 in 5-degree rings if `NULL`.
#' @param pulses_per_line Pulses per zenith line; estimated if `NULL`.
#' @return A list with ring `centres` and `gap`.
#' @export
gap_fraction_pattern <- function(shots, echo_heights, pattern, min_height = 0, zenith_edges = NULL,
                                 pulses_per_line = NULL) {
  edges <- if (is.null(zenith_edges)) seq(0, 90, by = 5) else as.double(zenith_edges)
  if (is.null(pulses_per_line)) pulses_per_line <- sylva::pulses_per_line(shots, pattern)
  core_gap_fraction_pattern(unclass(shots), as.double(echo_heights), scan_pattern(pattern), as.double(min_height),
                            edges, as.double(pulses_per_line))
}

#' @rdname gap_fraction_pattern
#' @export
gap_fraction_zenith <- function(shots, echo_heights, min_height = 0, zenith_edges = NULL) {
  edges <- if (is.null(zenith_edges)) seq(0, 90, by = 5) else as.double(zenith_edges)
  core_gap_fraction_zenith(unclass(shots), as.double(echo_heights), as.double(min_height), edges)
}

# -------------------------------------------------------------- ground, heights

#' Ground plane around a scanner
#'
#' Plane `z = a x + b y + c` through the lowest point of every `cell` grid
#' cell, fitted with Huber-weighted least squares (after Calders et al. 2014).
#'
#' @param points A `sylva_cloud` or `n x 3` matrix.
#' @param cell Grid cell (m).
#' @param centre,radius Only points within `radius` of `centre` (xy).
#' @param iterations Reweighting iterations.
#' @return `c(a, b, c)`.
#' @export
fit_ground_plane <- function(points, cell = 1, centre = NULL, radius = NULL, iterations = 20) {
  xyz <- if (inherits(points, "sylva_cloud")) unclass(points)$xyz else as_xyz(points)
  core_fit_ground_plane(xyz, as.double(cell), if (is.null(centre)) NULL else as.double(centre),
                        if (is.null(radius)) NULL else as.double(radius), as.integer(iterations))
}

#' Point counts by height
#'
#' @param cloud A height-normalised `sylva_cloud`.
#' @param bin_size Bin height (m).
#' @param height_attr Attribute holding heights; z if absent.
#' @param max_height Top of the last bin; the highest point if `NULL`.
#' @return A list with bin `bins` (bottoms) and `counts`.
#' @export
vertical_profile <- function(cloud, bin_size = 0.5, height_attr = "height", max_height = NULL) {
  c <- unclass(cloud)
  h <- if (!is.null(c$attrs[[height_attr]])) c$attrs[[height_attr]] else c$xyz[, 3]
  core_vertical_profile(as.double(h), as.double(bin_size), if (is.null(max_height)) NULL else as.double(max_height))
}

# --------------------------------------------------------------- density grids

#' Ray-traced plant area density on a voxel grid
#'
#' As the Python package's `density_grid()`: arrays are `[k, j, i]` (z, y, x);
#' `density` is `2 (n-1)/n hits / path length` per voxel, `NA` where fewer
#' than `min_hits` hits.
#'
#' @param shots A `sylva_shots` including misses.
#' @param voxel_size Voxel edge (m).
#' @param origin,shape Grid corner and `c(nx, ny, nz)`; from the shots if `NULL`.
#' @param min_hits Hits a voxel needs for a density.
#' @export
density_grid <- function(shots, voxel_size, origin = NULL, shape = NULL, min_hits = 2) {
  g <- core_density_grid(unclass(shots), as.double(voxel_size), if (is.null(origin)) NULL else as.double(origin),
                         if (is.null(shape)) NULL else as.double(shape), as.integer(min_hits))
  d <- g$dim
  structure(list(n_rays = from_row_major(g$n_rays, d), n_hits = from_row_major(g$n_hits, d),
                 path_length = from_row_major(g$path_length, d), density = from_row_major(g$density, d),
                 profile = g$profile, origin = g$origin, voxel_size = g$voxel_size),
            class = "sylva_density_grid")
}

grid_arg <- function(grid) {
  g <- unclass(grid)
  list(dim = as.double(dim(g$density)), origin = g$origin, voxel_size = g$voxel_size, n_rays = row_major(g$n_rays),
       n_hits = row_major(g$n_hits), path_length = row_major(g$path_length), density = row_major(g$density))
}

#' Density-grid profiles above the terrain
#'
#' `height_above()`: each voxel centre's height above `dtm`. `mask_ground()`:
#' the grid with voxels less than `margin` above the terrain removed.
#' `profile_above_ground()`: plant area density by height above the terrain,
#' pooled (`2 sum hits / sum path` per bin) or averaged. `pai()`: the layer
#' means summed over height.
#'
#' @param grid A `sylva_density_grid`.
#' @param dtm A `sylva_raster` in the grid's frame.
#' @param margin Minimum voxel-centre height (m); one voxel if `NULL`.
#' @param bin_size Height bin (m); the voxel size if `NULL`.
#' @param max_height Top of the profile; the highest sampled voxel if `NULL`.
#' @param pooled Pool hits and path lengths per bin.
#' @export
height_above <- function(grid, dtm) {
  from_row_major(do.call(core_grid_height_above, c(list(grid_arg(grid)), raster_args(dtm))), dim(unclass(grid)$density))
}

#' @rdname height_above
#' @export
mask_ground <- function(grid, dtm, margin = NULL) {
  g <- unclass(grid)
  m <- if (is.null(margin)) g$voxel_size else margin
  r <- do.call(core_grid_mask_ground, c(list(grid_arg(grid)), raster_args(dtm), list(as.double(m))))
  g$density <- from_row_major(r$density, dim(g$density))
  g$profile <- r$profile
  structure(g, class = "sylva_density_grid")
}

#' @rdname height_above
#' @export
profile_above_ground <- function(grid, dtm, bin_size = NULL, max_height = NULL, margin = NULL, pooled = TRUE) {
  g <- unclass(grid)
  b <- if (is.null(bin_size)) g$voxel_size else bin_size
  m <- if (is.null(margin)) g$voxel_size else margin
  do.call(core_grid_profile_above_ground, c(list(grid_arg(grid)), raster_args(dtm),
                                            list(as.double(b), as.double(m), isTRUE(pooled),
                                                 if (is.null(max_height)) NULL else as.double(max_height))))
}

#' @rdname height_above
#' @export
pai <- function(grid) {
  g <- unclass(grid)
  sum(g$profile, na.rm = TRUE) * g$voxel_size
}
