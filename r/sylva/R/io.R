# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

#' Read a point cloud
#'
#' The reader is chosen from the extension, matched case-insensitively:
#' `.las`/`.laz`, `.ply` (ASCII or binary, including raycloudtools ray
#' clouds), text (`.xyz`, `.txt`, `.csv`, `.asc`, `.pts`, see
#' `read_ascii()`) or a RIEGL `.rxp` scan (which needs RiVLib, see
#' `read_rxp()` for its options).
#'
#' LAS/LAZ attributes: `intensity`, `return_number`, `number_of_returns`,
#' `classification`, `scan_angle`, `user_data`, `point_source_id`, plus
#' `gps_time` and `red`/`green`/`blue` when the point format has them, and
#' every extra-bytes dimension under its own name. PLY: every vertex
#' property except x, y, z.
#'
#' @param path File to read.
#' @return A `sylva_cloud` in the file's frame.
#' @export
read_cloud <- function(path) as_cloud(core_read(path.expand(path)))

#' Write a point cloud
#'
#' The writer is chosen from the extension: `.las`, `.laz`, `.ply`, `.xyz`,
#' `.txt`, `.asc`, `.pts` (space separated) or `.csv` (comma separated). An
#' existing file is overwritten.
#'
#' LAS/LAZ: attributes that are not standard dimensions of `point_format`
#' are written as typed extra bytes, so `height`, `tree_id` and the like
#' survive a round trip. Text files get a header line naming the columns and
#' coordinates to 0.1 mm; integer attributes are written as integers.
#'
#' @param cloud A `sylva_cloud`.
#' @param path Output file.
#' @param point_format LAS point data record format (LAS/LAZ only). 6 (LAS
#'   1.4, with GPS time) is the default; use 7 or 8 to keep RGB.
#' @param scale LAS coordinate quantisation in metres (LAS/LAZ only); the
#'   offset is the floor of the minimum coordinate.
#' @param binary Binary little-endian PLY if `TRUE`, ASCII otherwise (PLY only).
#' @export
write_cloud <- function(cloud, path, point_format = 6, scale = 0.001, binary = TRUE) {
  invisible(core_write(unclass(cloud), path.expand(path), as.integer(point_format), as.double(scale), isTRUE(binary)))
}

#' Read a delimited text point cloud
#'
#' The delimiter (comma, semicolon, tab or whitespace) is detected from the
#' first data line. Lines starting with `#` are skipped, a leading line
#' holding a single integer (PTS point count) is skipped, and a non-numeric
#' first line is taken as a header.
#'
#' @param path Text file whose first three columns are x, y, z.
#' @param columns Names for the columns: either all of them (the first three
#'   are then ignored) or only those after x, y, z. Overrides a header line.
#'   Without names or header, extra columns are called `col3`, `col4`, ...
#' @return A `sylva_cloud` with the extra columns as double attributes.
#' @export
read_ascii <- function(path, columns = NULL) {
  as_cloud(core_read_ascii(path.expand(path), if (is.null(columns)) NULL else as.character(columns)))
}

#' Locate RiVLib's libscanifc
#'
#' @param hint A file, or a directory to search (three levels deep) first.
#' @return The path of the library found. After `hint`, the search order is
#'   `$RIVLIB_PATH`, `$RIVLIB_HOME`, `~/.local/lib`, `~/.local`, `~/lib`,
#'   `~/opt`, `~`, `/opt`, `/usr/local/lib`, `/usr/local` and `/usr/lib`,
#'   looking in subdirectories whose name contains "rivlib". An error if none
#'   is found; set `RIVLIB_PATH` to the extracted RiVLib directory.
#' @export
find_rivlib <- function(hint = NULL) core_find_rivlib(if (is.null(hint)) NULL else path.expand(hint))

#' Read a 4 x 4 transform file
#'
#' @param path Text file holding 16 whitespace-separated numbers in
#'   row-major order, as RiSCAN exports SOP and POP matrices (`.dat`, `.txt`).
#' @return A `4 x 4` matrix for `transform()`.
#' @export
read_matrix_file <- function(path) core_read_matrix_file(path.expand(path))
