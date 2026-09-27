# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

as_shots <- function(x) structure(x, class = "sylva_shots")

#' Laser pulses with their echoes
#'
#' One row per pulse, echoes in compressed sparse row form, as the Python
#' package's `Shots`: the echoes of pulse `i` are
#' `echo_range[echo_start[i] + seq_len(echo_count[i])]`. `echo_start` is
#' 0-based, as in the Python package and the `.shots` file format. A pulse
#' with `echo_count == 0` is a miss.
#'
#' @param origin,direction `n x 3` matrices: beam origin and unit direction.
#' @param echo_start,echo_count Offset (0-based) and number of each pulse's echoes.
#' @param echo_range Range of every echo from its origin.
#' @param echo_attrs Named list of per-echo attributes.
#' @return A `sylva_shots`.
#' @export
shots <- function(origin, direction, echo_start, echo_count, echo_range, echo_attrs = list()) {
  as_shots(core_shots_check(list(origin = as_xyz(origin), direction = as_xyz(direction),
                                 echo_start = as.double(echo_start), echo_count = as.double(echo_count),
                                 echo_range = as.double(echo_range), echo_attrs = echo_attrs)))
}

as_xyz <- function(m) {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  m
}

rxp_args <- function(library, drop_pseudo_echoes, min_range, max_range, stride, max_points, echoes, shot_stride) {
  list(library, isTRUE(drop_pseudo_echoes), as.double(min_range), as.double(max_range), as.integer(stride),
       if (is.null(max_points)) NULL else as.double(max_points), as.character(echoes), as.integer(shot_stride))
}

#' Read a RIEGL scan
#'
#' `read_rxp()` returns the points, `read_rxp_shots()` the pulses (echoes
#' grouped by timestamp; RIEGL streams hold only pulses that returned
#' something). Both need RiVLib (`find_rivlib()`), and return the scan in
#' its own frame.
#'
#' @param path The `.rxp` file.
#' @param library RiVLib's `libscanifc`, found automatically if `NULL`.
#' @param drop_pseudo_echoes Drop RIEGL's pseudo echoes.
#' @param min_range,max_range Range window (m).
#' @param stride,shot_stride Keep every `stride`-th point, or every
#'   `shot_stride`-th pulse with all its echoes.
#' @param max_points Stop after this many points.
#' @param echoes `"all"`, `"first"`, `"last"` or `"single"`.
#' @export
read_rxp <- function(path, library = NULL, drop_pseudo_echoes = TRUE, min_range = 0.5, max_range = Inf,
                     stride = 1, max_points = NULL, echoes = "all", shot_stride = 1) {
  a <- rxp_args(library, drop_pseudo_echoes, min_range, max_range, stride, max_points, echoes, shot_stride)
  as_cloud(do.call(core_read_rxp, c(list(path.expand(path)), a)))
}

#' @rdname read_rxp
#' @export
read_rxp_shots <- function(path, library = NULL, drop_pseudo_echoes = TRUE, min_range = 0.5, max_range = Inf,
                           stride = 1, max_points = NULL, echoes = "all", shot_stride = 1) {
  a <- rxp_args(library, drop_pseudo_echoes, min_range, max_range, stride, max_points, echoes, shot_stride)
  as_shots(do.call(core_read_rxp_shots, c(list(path.expand(path)), a)))
}

#' Pulses from a point cloud
#'
#' `shots_from_pointcloud()` groups adjacent points with equal `gps_time`
#' into one pulse from `origin` (the scanner position in the cloud's frame);
#' `shots_from_ray_cloud()` reads a ray cloud (points with a `ray_x/y/z`
#' origin per point).
#'
#' @param cloud A `sylva_cloud`.
#' @param origin Scanner position.
#' @export
shots_from_pointcloud <- function(cloud, origin = c(0, 0, 0)) {
  as_shots(core_shots_from_cloud(unclass(cloud), as.double(origin)))
}

#' @rdname shots_from_pointcloud
#' @export
shots_from_ray_cloud <- function(cloud) as_shots(core_shots_from_ray_cloud(unclass(cloud)))

#' Echoes of pulses as points
#'
#' @param shots A `sylva_shots`.
#' @return `to_pointcloud()`: a `sylva_cloud` of the echoes with their
#'   attributes; `echo_xyz()`: their coordinates as an `n x 3` matrix.
#' @export
to_pointcloud <- function(shots) as_cloud(core_shots_to_cloud(unclass(shots)))

#' @rdname to_pointcloud
#' @export
echo_xyz <- function(shots) core_shots_echo_xyz(unclass(shots))

#' Pulse directions
#'
#' @param shots A `sylva_shots`.
#' @return A list with `zenith` (degrees from up) and `azimuth` (degrees
#'   clockwise from +y), one per pulse.
#' @export
zenith_azimuth <- function(shots) core_shots_zenith_azimuth(unclass(shots))

#' Echo bookkeeping
#'
#' `shot_of_echo()`: the (1-based) pulse of every echo; `echo_rank()`: each
#' echo's order within its pulse (1 = first).
#'
#' @param shots A `sylva_shots`.
#' @export
shot_of_echo <- function(shots) core_shots_shot_of_echo(unclass(shots)) + 1

#' @rdname shot_of_echo
#' @export
echo_rank <- function(shots) core_shots_echo_rank(unclass(shots)) + 1L

#' @export
subset.sylva_shots <- function(x, keep, ...) {
  as_shots(core_shots_subset(unclass(x), as.logical(keep)))
}

#' @export
transform.sylva_shots <- function(`_data`, matrix, ...) {
  as_shots(core_shots_transform(unclass(`_data`), as_xyz(matrix)))
}

#' @export
length.sylva_shots <- function(x) nrow(unclass(x)$origin)

#' @export
print.sylva_shots <- function(x, ...) {
  cat(sprintf("<sylva_shots> %s pulses, %s echoes\n", format(length(x), big.mark = ","),
              format(length(unclass(x)$echo_range), big.mark = ",")))
  invisible(x)
}

#' Add the pulses that returned nothing
#'
#' RiVLib's point stream only holds pulses that returned something, so a
#' pulse that went to the sky is absent. Given the scan `pattern`, this
#' compares, per zenith line, the pulses fired against the shots observed
#' and adds the difference as shots without echoes, spread uniformly in
#' azimuth (the exact azimuths cannot be recovered). Ray-traced metrics then
#' see the free space these pulses sampled. The azimuths are drawn exactly
#' as the Python package draws them, so a seed gives the same shots in both.
#'
#' Call on shots in the scanner frame (before the SOP), as the pattern's
#' zenith lines are defined there; all shots must share one origin.
#' `read_shots(position, fill_missing = TRUE)` does this in the right order.
#'
#' @param shots A `sylva_shots` in the scanner frame.
#' @param pattern The scan pattern (`theta_start`, `theta_delta`,
#'   `theta_count`, `phi_count`), as in a scan position's `pattern`.
#' @param pulses_per_line Pulses fired per zenith line; estimated with
#'   `pulses_per_line()` if `NULL`.
#' @param seed Seed for the random azimuths.
#' @param shot_stride The `shot_stride` the shots were read with; the
#'   estimate of pulses per line is divided by it.
#' @return The input followed by the added misses (the input itself if none
#'   are missing).
#' @export
fill_missing <- function(shots, pattern, pulses_per_line = NULL, seed = 0, shot_stride = 1) {
  if (!is.null(pulses_per_line)) pulses_per_line <- as.double(pulses_per_line)
  filled <- core_shots_fill_missing(unclass(shots), scan_pattern(pattern), pulses_per_line, as.double(seed),
                                    as.integer(shot_stride))
  if (is.null(filled)) shots else as_shots(filled)
}

concatenate_shots <- function(parts) {
  as_shots(core_shots_concatenate(lapply(parts, unclass)))
}

#' The sylva shots file format
#'
#' `save_shots()` writes pulses to a Parquet file with one row per pulse, as
#' the Python package's `Shots.save()`: columns `scan` (index into the
#' scanner positions kept in the file metadata), `zenith` and `azimuth`
#' (rad), `range` (a list of echo ranges, empty for a pulse without a
#' return) and one list column per echo attribute, zstd compressed in row
#' groups of `row_group_size` pulses. Any Parquet reader (arrow, duckdb,
#' polars) opens it. `load_shots()` reads it back, `shots_file_info()`
#' describes it without reading the pulses.
#'
#' Angles and ranges are float32 unless `double`: echo positions then come
#' back to about 0.01 mm per 100 m of range. Origins within
#' `origin_tolerance` (m) of each other become one scanner position; `0`
#' keeps them exact.
#'
#' @param shots A `sylva_shots`.
#' @param path The `.parquet` file; `save_shots()` overwrites it.
#' @param double Store angles and ranges as float64.
#' @param row_group_size Pulses per row group, the unit of partial reads.
#' @param zstd_level zstd compression level (1 fast, 22 smallest).
#' @param origin_tolerance Distance (m) within which origins are merged.
#' @param groups Row groups to read (1-based, up to `n_groups` of
#'   `shots_file_info()`); all if `NULL`.
#' @return `load_shots()`: a `sylva_shots`. `shots_file_info()`: a list with
#'   `n_shots`, `n_echoes`, `n_groups`, echo `bounds` (minimum and maximum
#'   corners), `scans` (scanner positions, an `n x 3` matrix; empty when
#'   origins are stored per pulse) and `echo_attrs` (names).
#' @export
save_shots <- function(shots, path, double = FALSE, row_group_size = 2^20, zstd_level = 3, origin_tolerance = 1e-3) {
  core_write_shots(unclass(shots), path.expand(path), isTRUE(double), as.double(row_group_size),
                   as.integer(zstd_level), as.double(origin_tolerance))
  invisible(path)
}

#' @rdname save_shots
#' @export
load_shots <- function(path, groups = NULL) {
  if (!is.null(groups)) groups <- as.double(groups) - 1
  as_shots(core_read_shots(path.expand(path), groups))
}

#' @rdname save_shots
#' @export
shots_file_info <- function(path) {
  info <- core_shots_info(path.expand(path))
  names(info$bounds) <- c("min", "max")
  info
}
