# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

as_position <- function(x) structure(x, class = "sylva_scan_position")

#' Read a RiSCAN PRO project
#'
#' Parses the project structure and matrices, reading no scan data. A
#' `.RiSCAN` directory with `project.rsp` gives each position's SOP and scan
#' pattern (falling back to `DAT/<pos>.DAT`); without it the legacy layout is
#' read: `all_sop.csv` (roll, pitch, yaw in degrees and x, y, z),
#' `project.pop` and `SCANS/ScanPos*` or `*.SCNPOS` folders. For a scanner
#' `.PROJ` (`*.SCNPOS`) each position also gets the scanner's attitude
#' ([levelling()]), its GNSS fix and its `.tpl` target list.
#'
#' The POP of geo-referenced projects is usually a geocentric (ECEF)
#' transform, so it is returned but only applied when asked for.
#'
#' @param path The project directory.
#' @return A `sylva_riscan_project`: a list with `path`, `name`, `pop` (the
#'   project-to-global matrix or `NULL`) and `positions`, a list of
#'   `sylva_scan_position` objects. Each position is a list with `name`,
#'   `rxp` (its first scan or `NULL`), `sop` (scanner-to-project matrix or
#'   `NULL`), `scans`, `instrument`, `pattern` (the scan pattern:
#'   `theta_start`, `theta_delta`, `theta_count`, `phi_start`, `phi_delta`,
#'   `phi_count`), `tiepoints`, `gnss` (latitude, longitude, altitude) and
#'   `attitude` (3 x 3 levelling rotation). The list is named by position,
#'   so `project$positions[[3]]` or `project$positions[["ScanPos003"]]` picks
#'   one; `summary()` tabulates them.
#' @examples
#' \dontrun{
#' project <- read_riscan_project("plot.RiSCAN")
#' clouds <- lapply(with_scans(project), read_points, shot_stride = 4)
#' }
#' @export
read_riscan_project <- function(path) {
  path <- path.expand(path)
  if (!dir.exists(path)) stop("not a project directory: ", path)
  p <- core_riscan_read_project(path)
  positions <- lapply(p$positions, as_position)
  names(positions) <- vapply(positions, function(q) q$name, "")
  structure(list(path = path, name = p$name, pop = p$pop, positions = positions), class = "sylva_riscan_project")
}

#' @export
length.sylva_riscan_project <- function(x) length(x$positions)

#' Scan positions of a project
#'
#' `position_names()` lists the positions in project order; `with_scans()`
#' returns the positions that have a scan.
#'
#' @param project A `sylva_riscan_project`.
#' @param require_sop Only positions that also have a SOP, so can be read
#'   into project coordinates. `FALSE` for registration, which does not need
#'   one.
#' @return `with_scans()`: a list of `sylva_scan_position`.
#' @export
with_scans <- function(project, require_sop = TRUE) {
  unname(Filter(function(p) !is.null(p$rxp) && (!is.null(p$sop) || !require_sop), project$positions))
}

#' @rdname with_scans
#' @export
position_names <- function(project) vapply(unname(project$positions), function(p) p$name, "")

#' Scanner positions of a project
#'
#' `origins()`: the SOP translations, one row per position that has a SOP
#' (positions without one are skipped). `gnss_positions()`: the scanners'
#' GNSS fixes in local metres ([gnss_to_local()]), one row per position, NA
#' where there was no fix.
#'
#' @param project A `sylva_riscan_project`.
#' @return An `n x 3` matrix.
#' @export
origins <- function(project) {
  rows <- lapply(unname(Filter(function(p) !is.null(p$sop), project$positions)), function(p) p$sop[1:3, 4])
  if (!length(rows)) return(matrix(numeric(0), 0, 3))
  do.call(rbind, rows)
}

#' @rdname origins
#' @export
gnss_positions <- function(project) gnss_to_local(lapply(unname(project$positions), function(p) p$gnss))

#' GNSS fixes as local metres
#'
#' An equirectangular projection about the survey's own centre, accurate to
#' well under a metre over a plot, far finer than the fixes themselves.
#'
#' @param coordinates `(latitude, longitude, altitude)` per position: a list
#'   of length-3 vectors or `NULL`, or an `n x 3` matrix with `NA` rows.
#' @return An `n x 3` matrix of east, north and altitude; `NA` (NaN) where
#'   there was no fix.
#' @export
gnss_to_local <- function(coordinates) {
  if (is.list(coordinates)) {
    coordinates <- do.call(rbind, lapply(coordinates, function(c) if (is.null(c)) rep(NA_real_, 3) else as.double(c[1:3])))
    if (is.null(coordinates)) coordinates <- matrix(numeric(0), 0, 3)
  }
  core_riscan_gnss_to_local(as_xyz(coordinates))
}

#' Tabulate a project's positions
#'
#' @param object A `sylva_riscan_project`.
#' @param ... Unused.
#' @return A data frame, one row per position: `name`, `rxp`, `n_scans`,
#'   `has_sop`, the SOP translation `x`, `y`, `z`, `instrument` and the GNSS
#'   fix `latitude`, `longitude`, `altitude`.
#' @export
summary.sylva_riscan_project <- function(object, ...) {
  p <- unname(object$positions)
  num <- function(f) vapply(p, f, 0)
  chr <- function(f) vapply(p, function(q) { v <- f(q); if (is.null(v)) NA_character_ else v }, "")
  data.frame(name = chr(function(q) q$name), rxp = chr(function(q) q$rxp),
             n_scans = vapply(p, function(q) length(q$scans), 0L),
             has_sop = vapply(p, function(q) !is.null(q$sop), TRUE),
             x = num(function(q) if (is.null(q$sop)) NA_real_ else q$sop[1, 4]),
             y = num(function(q) if (is.null(q$sop)) NA_real_ else q$sop[2, 4]),
             z = num(function(q) if (is.null(q$sop)) NA_real_ else q$sop[3, 4]),
             instrument = chr(function(q) q$instrument),
             latitude = num(function(q) if (is.null(q$gnss)) NA_real_ else q$gnss[1]),
             longitude = num(function(q) if (is.null(q$gnss)) NA_real_ else q$gnss[2]),
             altitude = num(function(q) if (is.null(q$gnss)) NA_real_ else q$gnss[3]),
             stringsAsFactors = FALSE)
}

#' @export
print.sylva_riscan_project <- function(x, ...) {
  cat(sprintf("<sylva_riscan_project> %s: %d positions, %d with a scan%s\n", x$name, length(x),
              length(with_scans(x, require_sop = FALSE)), if (is.null(x$pop)) "" else ", POP"))
  invisible(x)
}

#' @export
print.sylva_scan_position <- function(x, ...) {
  cat(sprintf("<sylva_scan_position> %s%s%s\n", x$name, if (is.null(x$rxp)) ", no scan" else paste0(": ", basename(x$rxp)),
              if (is.null(x$sop)) ", no SOP" else ""))
  invisible(x)
}

#' A scan position's matrices and pattern
#'
#' `sop()`: the scanner-to-project matrix (`NULL` if none). `pattern()`: the
#' angular scan pattern of its first scan (`NULL` if the project does not
#' record one). `levelling()`: the 4 x 4 rotation that levels the scan, from
#' the scanner's attitude (`NULL` if unknown). `origin()`: the scanner
#' position in project coordinates. `transform()`: the scanner-to-project
#' matrix (the identity without a SOP), or scanner-to-global with `pop`.
#'
#' @param position A `sylva_scan_position`.
#' @param _data A `sylva_scan_position`.
#' @param pop Project-to-global matrix (`project$pop`); the result is then
#'   `pop %*% sop`.
#' @param ... Unused.
#' @export
sop <- function(position) position$sop

#' @rdname sop
#' @export
pattern <- function(position) position$pattern

#' @rdname sop
#' @export
levelling <- function(position) {
  if (is.null(position$attitude)) return(NULL)
  m <- diag(4)
  m[1:3, 1:3] <- position$attitude
  m
}

#' @rdname sop
#' @export
origin <- function(position) if (is.null(position$sop)) NULL else position$sop[1:3, 4]

#' @rdname sop
#' @export
transform.sylva_scan_position <- function(`_data`, pop = NULL, ...) {
  m <- if (is.null(`_data`$sop)) diag(4) else `_data`$sop
  if (is.null(pop)) m else pop %*% m
}

#' The targets a scanner found
#'
#' @param position A `sylva_scan_position`.
#' @return A data frame of the reflective targets in the position's `.tpl`
#'   list, in the scanner's own frame: `x`, `y`, `z`, `reflectance`,
#'   `diameter`, `n_points`, `name` (no rows without a list).
#' @export
reflectors <- function(position) {
  t <- if (is.null(position$tiepoints)) {
    list(x = numeric(0), y = numeric(0), z = numeric(0), reflectance = numeric(0), diameter = numeric(0),
         n_points = numeric(0), name = character(0))
  } else {
    core_riscan_read_tiepoints(position$tiepoints)
  }
  as.data.frame(t, stringsAsFactors = FALSE)
}

#' Read a scan position
#'
#' `read_points()` reads the position's scan as points, `read_shots()` as
#' pulses, in project coordinates (or global ones with `pop`). Both need
#' RiVLib (see [read_rxp()]).
#'
#' @param position A `sylva_scan_position`.
#' @param pop Also apply this project-to-global matrix (`project$pop`).
#' @param fill_missing Reconstruct the no-return pulses from the scan
#'   pattern. Not yet available in R.
#' @param ... Passed to [read_rxp()] or [read_rxp_shots()] (`shot_stride`,
#'   `min_range`, `echoes`, ...).
#' @export
read_points <- function(position, pop = NULL, ...) {
  if (is.null(position$rxp)) stop("scan position ", position$name, " has no .rxp")
  cloud <- read_rxp(position$rxp, ...)
  as_cloud(core_riscan_transform_cloud(unclass(cloud), row_major(transform(position, pop))))
}

#' @rdname read_points
#' @export
read_shots <- function(position, pop = NULL, fill_missing = FALSE, ...) {
  if (is.null(position$rxp)) stop("scan position ", position$name, " has no .rxp")
  if (fill_missing) {
    if (is.null(position$pattern)) stop("scan position ", position$name, " has no scan pattern in project.rsp")
    stop("fill_missing is not yet available in R")
  }
  transform(read_rxp_shots(position$rxp, ...), transform(position, pop))
}

#' RiSCAN PRO's export filter
#'
#' `read_export_settings()` reads an export filter settings file: one line
#' per attribute, `name, minimum, maximum` (`;` also separates, `#` starts a
#' comment), names lower-cased without a leading `riegl.`.
#' `export_settings_mask()` keeps the points within every closed interval,
#' `range` measured from the scanner's origin; RIEGL's "not measured"
#' deviation (65535) counts as -1.
#'
#' @param path The settings file.
#' @param settings From `read_export_settings()`: a named list of
#'   `c(minimum, maximum)` for `range`, `deviation`, `reflectance` and
#'   `amplitude`.
#' @param xyz `n x 3` points in the scanner's frame.
#' @param attributes Named list of per-point `deviation`, `reflectance`,
#'   `amplitude` as the settings need.
#' @return `read_export_settings()`: the named list; `export_settings_mask()`:
#'   a logical keep-mask.
#' @export
read_export_settings <- function(path) {
  s <- core_riscan_read_export_settings(path.expand(path))
  stats::setNames(lapply(seq_along(s$name), function(k) c(s$min[k], s$max[k])), s$name)
}

#' @rdname read_export_settings
#' @export
export_settings_mask <- function(settings, xyz, attributes = list()) {
  missing <- setdiff(setdiff(names(settings), "range"), names(attributes))
  if (length(missing)) stop("export settings need attribute '", missing[1], "', which this source lacks")
  used <- lapply(attributes[intersect(names(attributes), names(settings))], as.double)
  core_riscan_export_settings_mask(names(settings), vapply(settings, function(b) as.double(b[1]), 0),
                                   vapply(settings, function(b) as.double(b[2]), 0), as_xyz(xyz), used)
}

#' The scan pattern's angular steps
#'
#' The polar step is the median step between consecutive records; the
#' azimuth step is the spacing of the azimuths within one polar row, the
#' median over a dozen rows. Falls back to 0.03 degrees, the usual VZ-series
#' setting, for a scan too small to tell. Computed in single precision, as
#' the scan stream records points.
#'
#' @param xyz `n x 3` points in the scanner's frame, in recording order.
#' @param sample Records used.
#' @return `c(theta_step, phi_step)` in degrees.
#' @export
angular_steps <- function(xyz, sample = 2e6) core_riscan_angular_steps(as_xyz(xyz), as.double(sample))

#' What RiSCAN PRO's RXP import keeps
#'
#' An approximate keep-mask, measured against RiSCAN's own databases on
#' TERN VZ-2000i scans. `"current"` (a current RiSCAN's import) drops
#' `range < min_range`; `"legacy"` (the older conversion) also drops an echo
#' weaker than `weak_db` with fewer than `min_neighbours` other echoes within
#' `window_steps` scan increments and `window_range` m of range; `"none"`
#' keeps everything.
#'
#' @param xyz `n x 3` points in the scanner's frame, in recording order.
#' @param amplitude Amplitude (dB) per point.
#' @param mode `"none"`, `"current"` or `"legacy"`.
#' @param min_range,window_steps,window_range,min_neighbours,weak_db See above.
#' @param steps `c(theta_step, phi_step)` of the scan pattern (degrees);
#'   estimated by [angular_steps()] if `NULL`.
#' @return A logical keep-mask.
#' @export
riscan_like_mask <- function(xyz, amplitude, mode = "current", min_range = 0.5, window_steps = 1.5,
                             window_range = 1.0, min_neighbours = 6, weak_db = 12.0, steps = NULL) {
  modes <- c("none", "current", "legacy")
  if (!mode %in% modes) stop("mode must be one of ", paste(modes, collapse = ", "), ", not ", mode)
  core_riscan_like_mask(as_xyz(xyz), as.double(amplitude), mode, as.double(min_range), as.double(window_steps),
                        as.double(window_range), as.double(min_neighbours), as.double(weak_db),
                        if (length(steps)) as.double(steps) else NULL)
}
