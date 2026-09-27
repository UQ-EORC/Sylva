# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

#' Scan position of each point, from its sensor origin
#'
#' @param origins An `n x 3` matrix: the sensor position of each point.
#' @param tolerance Origins on the same `tolerance` (m) grid cell are one
#'   position.
#' @return Integer id per point, `0 .. n-1` in sorted order of the origins
#'   (the ids of the Python package).
#' @export
scan_ids_from_origins <- function(origins, tolerance = 0.05) {
  o <- as.matrix(origins)
  storage.mode(o) <- "double"
  as.integer(core_scan_ids_from_origins(o, as.double(tolerance)))
}

#' Stem-based noise and registration of a cloud
#'
#' Between 1 and 3 m a stem is the one surface in a forest scan whose shape
#' is known well enough to measure the scanner against. Each stem is cut
#' into thin slices and a circle fitted to every slice from all scans
#' together; every point's radial residual is then read per scan position.
#' `summary()` gives the plot-level figures: `sigma_total` (all points about
#' the shared circle), `sigma_within` (one scan's points about their own
#' median), `sigma_local` (after removing a smooth curve along the arc: range
#' noise and bark only) and the scans' horizontal offsets.
#'
#' @param cloud Registered multi-scan cloud (or one scan) covering 1-3 m of the stems.
#' @param scan_id Scan position per point: a vector, the name of an
#'   attribute, or `NULL` for a single scan (see `scan_ids_from_origins()`).
#' @param stems Stem positions as an `n x 2` matrix, a tree table, or `NULL`
#'   to detect them.
#' @param height_attr Attribute holding height above ground; the height
#'   above the lowest point if absent.
#' @param iterations Rounds of moving the scans back and refitting.
#' @param height_min,height_max,step,thickness Slices (m above ground).
#' @param min_radius,max_radius Accepted slice radii (m).
#' @param min_arc Degrees of a slice's circle that must be seen.
#' @param min_inlier_fraction Share of the points near a circle it must explain.
#' @param cut_min,cut_fraction Points further than `max(cut_min, cut_fraction * r)`
#'   from the circle are not stem.
#' @param min_scan_points Fewest points of one scan in a slice.
#' @return A `sylva_stem_noise`: data frames `slices`, `scan_slices` and
#'   `scans`, and `residual` per input point (`NaN` where not used).
#' @export
stem_noise <- function(cloud, scan_id = NULL, stems = NULL, height_attr = "height", iterations = 3,
                       height_min = 1, height_max = 3, step = 0.25, thickness = 0.1, min_radius = 0.05,
                       max_radius = 1, min_arc = 270, min_inlier_fraction = 0.5, cut_min = 0.05,
                       cut_fraction = 0.3, min_scan_points = 30) {
  c <- unclass(cloud)
  h <- if (!is.null(c$attrs[[height_attr]])) as.double(c$attrs[[height_attr]]) else c$xyz[, 3] - min(c$xyz[, 3])
  if (is.character(scan_id)) scan_id <- c$attrs[[scan_id]]
  if (is.null(stems)) {
    with_h <- cloud
    if (is.null(c$attrs[[height_attr]])) {
      w <- unclass(cloud)
      w$attrs[[height_attr]] <- h
      with_h <- as_cloud(w)
    }
    found <- detect_stems(with_h, height_attr = height_attr)
    stems <- cbind(found$x, found$y)
  } else if (is.data.frame(stems)) {
    stems <- cbind(stems$x, stems$y)
  }
  stems <- matrix(as.double(stems), ncol = 2)
  r <- core_stem_noise(c$xyz, h, stems, if (is.null(scan_id)) NULL else as.double(scan_id), as.double(height_min),
                       as.double(height_max), as.double(step), as.double(thickness), as.double(min_radius),
                       as.double(max_radius), as.double(min_arc), as.double(min_inlier_fraction),
                       as.double(cut_min), as.double(cut_fraction), as.double(min_scan_points),
                       as.double(iterations))
  structure(list(slices = as.data.frame(r$slices), scan_slices = as.data.frame(r$scan_slices),
                 scans = as.data.frame(r$scans), residual = r$residual),
            class = "sylva_stem_noise")
}

#' Plot-level quality figures from stem noise
#'
#' @param object A `sylva_stem_noise`.
#' @param min_scan_slices Scans measured in fewer stem slices than this are
#'   left out of the registration figures.
#' @param ... Unused.
#' @return A list, in metres: `n_stems`, `n_slices`, `n_scans`; the
#'   point-weighted medians `sigma_total`, `sigma_corrected`,
#'   `sigma_within`, `sigma_local`; `tail_fraction`; and with several scans
#'   `registration_rms`, `registration_max` and `worst_scan` over the
#'   `n_scans_registered` scans with enough slices. Only the counts are
#'   present if no slice qualified.
#' @export
summary.sylva_stem_noise <- function(object, min_scan_slices = 1, ...) {
  x <- unclass(object)
  core_stem_noise_summary(as.list(x$slices), as.list(x$scan_slices), as.list(x$scans), as.double(min_scan_slices))
}

#' @export
print.sylva_stem_noise <- function(x, ...) {
  x <- unclass(x)
  cat(sprintf("<sylva_stem_noise> %d stem slices, %d scans\n", nrow(x$slices), nrow(x$scans)))
  invisible(x)
}
