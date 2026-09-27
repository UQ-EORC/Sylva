# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

as_cloud <- function(x) structure(x, class = "sylva_cloud")

# A cloud checked by the core (lengths, attribute types).
new_cloud <- function(xyz, attrs = list()) {
  xyz <- as.matrix(xyz)
  storage.mode(xyz) <- "double"
  dimnames(xyz) <- NULL
  as_cloud(core_cloud(list(xyz = xyz, attrs = attrs)))
}

# The n x 3 coordinates of a cloud, or a numeric matrix as one.
xyz_of <- function(x) {
  if (inherits(x, "sylva_cloud")) return(unclass(x)$xyz)
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  if (ncol(x) != 3) stop("expected a sylva_cloud or an n x 3 matrix", call. = FALSE)
  x
}

attrs_of <- function(x) unclass(x)$attrs

#' A point cloud
#'
#' Points with optional per-point attributes, as in the Python package's
#' `PointCloud`. Coordinates are metres in whatever frame the data came in
#' (scanner, project or projected CRS). Every function that takes points
#' returns a new cloud; nothing is modified in place.
#'
#' Attributes follow laspy's names so LAS files round-trip:
#' `classification` (2 = ground, from `classify_ground_csf()`), `height`
#' (from `normalize_height()`), `tree_id`, `scan_id` (from
#' `merge_scans()`), and `nx`, `ny`, `nz` in ray-cloud PLY files.
#'
#' A cloud is subset with `cloud[i]` (logical, positive or negative
#' positions), keeping all attributes; `cloud$x`, `cloud$y`, `cloud$z`,
#' `cloud$xyz` and `cloud$<attribute>` read it.
#'
#' @param xyz An `n x 3` numeric matrix (or anything `as.matrix` turns into one).
#' @param ... Per-point attributes as named vectors of length `n` (double,
#'   integer or logical).
#' @return A `sylva_cloud`: a list with `xyz` and `attrs`.
#' @examples
#' cloud <- point_cloud(matrix(runif(300), ncol = 3), intensity = rep(1L, 100))
#' tall <- cloud[cloud$z > 0.5]
#' @export
point_cloud <- function(xyz, ...) new_cloud(xyz, list(...))

#' @export
length.sylva_cloud <- function(x) nrow(unclass(x)$xyz)

#' @export
print.sylva_cloud <- function(x, ...) {
  a <- sort(names(attrs_of(x)))
  cat(sprintf("<sylva_cloud> %s points%s\n", format(length(x), big.mark = ","),
              if (length(a)) paste0(", attrs: ", paste(a, collapse = ", ")) else ""))
  invisible(x)
}

#' @export
`[.sylva_cloud` <- function(x, i) {
  x <- unclass(x)
  as_cloud(list(xyz = x$xyz[i, , drop = FALSE], attrs = lapply(x$attrs, `[`, i)))
}

#' @export
`$.sylva_cloud` <- function(x, name) cloud_field(x, name)

#' @export
`[[.sylva_cloud` <- function(x, i, ...) cloud_field(x, i)

cloud_field <- function(x, name) {
  x <- unclass(x)
  switch(name, xyz = x$xyz, attrs = x$attrs, x = x$xyz[, 1], y = x$xyz[, 2], z = x$xyz[, 3], x$attrs[[name]])
}

#' Axis-aligned bounds of a cloud
#'
#' @param cloud A `sylva_cloud`.
#' @return `list(min, max)`, two length-3 vectors.
#' @export
bounds <- function(cloud) {
  xyz <- xyz_of(cloud)
  list(min = apply(xyz, 2, min), max = apply(xyz, 2, max))
}

#' Height above ground for each point
#'
#' @param cloud A `sylva_cloud`.
#' @param attr Attribute holding normalised heights.
#' @return `attr` as double if the cloud has it, otherwise z. Falling back to
#'   z is only right for clouds that are already height-normalised (e.g. from
#'   `flatten()`).
#' @export
heights <- function(cloud, attr = "height") {
  a <- attrs_of(cloud)[[attr]]
  if (is.null(a)) xyz_of(cloud)[, 3] else as.double(a)
}

#' Add, replace or drop attributes
#'
#' @param cloud A `sylva_cloud`.
#' @param ... For `with_attrs()`, `name = vector` pairs of length `n`; for
#'   `without()`, attribute names (absent names are ignored).
#' @return A new cloud; existing attributes are kept unless replaced or dropped.
#' @export
with_attrs <- function(cloud, ...) {
  a <- attrs_of(cloud)
  new <- list(...)
  if (length(new) && (is.null(names(new)) || any(names(new) == ""))) stop("attributes must be named", call. = FALSE)
  a[names(new)] <- new
  new_cloud(xyz_of(cloud), a)
}

#' @rdname with_attrs
#' @export
without <- function(cloud, ...) {
  drop <- c(...)
  a <- attrs_of(cloud)
  as_cloud(list(xyz = xyz_of(cloud), attrs = a[setdiff(names(a), drop)]))
}

#' Apply a transform to a cloud
#'
#' @param _data A `sylva_cloud`.
#' @param matrix A `4 x 4` homogeneous matrix acting on column vectors, e.g.
#'   a RIEGL SOP from `read_matrix_file()` or the result of `icp()`.
#' @param ... Unused.
#' @return The transformed cloud. Attributes are carried over unchanged, so
#'   direction-like attributes (`nx`, `ny`, `nz`) are not rotated.
#' @export
transform.sylva_cloud <- function(`_data`, matrix, ...) {
  m <- as.matrix(matrix)
  if (!identical(dim(m), c(4L, 4L))) stop("matrix must be 4x4", call. = FALSE)
  storage.mode(m) <- "double"
  as_cloud(list(xyz = core_transform_xyz(xyz_of(`_data`), m), attrs = attrs_of(`_data`)))
}

#' Stack several clouds into one
#'
#' @param ... Clouds, or one list of clouds, in order.
#' @return All points; only attributes present in every input are kept, in
#'   name order. Use `merge_scans()` to also record which scan each point
#'   came from.
#' @export
concatenate <- function(...) {
  clouds <- list(...)
  if (length(clouds) == 1 && !inherits(clouds[[1]], "sylva_cloud")) clouds <- clouds[[1]]
  if (!length(clouds)) stop("no clouds to concatenate", call. = FALSE)
  common <- sort(Reduce(intersect, lapply(clouds, function(c) names(attrs_of(c)))))
  attrs <- lapply(stats::setNames(common, common), function(k) unlist(lapply(clouds, function(c) attrs_of(c)[[k]]), use.names = FALSE))
  as_cloud(list(xyz = do.call(rbind, lapply(clouds, xyz_of)), attrs = attrs))
}

#' Build a cloud from a matrix or data frame
#'
#' `from_array()` takes a numeric matrix whose first three columns are x, y
#' and z; `as_point_cloud()` also takes a data frame (columns named by
#' `xyz`) or a lidR `LAS` object.
#'
#' @param array An `n x (3 + k)` numeric matrix.
#' @param names Names for the extra columns; unnamed ones take the matrix's
#'   column names, or `col3`, `col4`, ... (0-based, as in the Python
#'   package and `read_ascii()`).
#' @return A `sylva_cloud`.
#' @export
from_array <- function(array, names = NULL) {
  array <- as.matrix(array)
  storage.mode(array) <- "double"
  k <- ncol(array) - 3
  if (k < 0) stop("array must have at least three columns", call. = FALSE)
  nm <- if (k) paste0("col", 3:(ncol(array) - 1)) else character()
  cn <- colnames(array)
  if (!is.null(cn) && k) nm <- ifelse(is.na(cn[-(1:3)]) | cn[-(1:3)] == "", nm, cn[-(1:3)])
  if (!is.null(names)) {
    given <- rep_len(NA_character_, k)
    given[seq_along(names)] <- names
    nm <- ifelse(is.na(given) | given == "", nm, given)
  }
  attrs <- stats::setNames(lapply(seq_len(k), function(j) array[, 3 + j]), nm)
  new_cloud(array[, 1:3, drop = FALSE], attrs)
}

#' @rdname from_array
#' @param x A matrix, data frame or lidR `LAS` object.
#' @param ... Passed on to methods (`xyz` for data frames).
#' @export
as_point_cloud <- function(x, ...) UseMethod("as_point_cloud")

#' @export
as_point_cloud.sylva_cloud <- function(x, ...) x

#' @export
as_point_cloud.default <- function(x, ...) from_array(x, ...)

#' @export
as_point_cloud.data.frame <- function(x, xyz = c("x", "y", "z"), ...) {
  if (!all(xyz %in% names(x))) stop("the data frame has no columns ", paste(xyz, collapse = ", "), call. = FALSE)
  rest <- setdiff(names(x), xyz)
  attrs <- lapply(as.list(x)[rest], function(v) if (is.factor(v)) as.integer(v) else v)
  new_cloud(as.matrix(x[xyz]), attrs)
}

#' @export
as.data.frame.sylva_cloud <- function(x, row.names = NULL, optional = FALSE, ...) {
  xyz <- xyz_of(x)
  df <- data.frame(x = xyz[, 1], y = xyz[, 2], z = xyz[, 3], row.names = row.names)
  a <- attrs_of(x)
  for (k in names(a)) df[[k]] <- a[[k]]
  df
}

# lidR's names for the LAS dimensions and Sylva's (laspy's) for them.
LAS_NAMES <- c(Intensity = "intensity", ReturnNumber = "return_number", NumberOfReturns = "number_of_returns",
               Classification = "classification", ScanAngleRank = "scan_angle", ScanAngle = "scan_angle",
               UserData = "user_data", PointSourceID = "point_source_id", gpstime = "gps_time",
               R = "red", G = "green", B = "blue", NIR = "nir", ScanDirectionFlag = "scan_direction_flag",
               EdgeOfFlightline = "edge_of_flight_line", Synthetic_flag = "synthetic",
               Keypoint_flag = "key_point", Withheld_flag = "withheld", Overlap_flag = "overlap",
               ScannerChannel = "scanner_channel")
LAS_INTEGER <- c("Intensity", "ReturnNumber", "NumberOfReturns", "Classification", "ScanAngleRank",
                 "UserData", "PointSourceID", "R", "G", "B", "NIR", "ScannerChannel")

#' @export
as_point_cloud.LAS <- function(x, ...) {
  d <- as.data.frame(x@data)
  rest <- setdiff(names(d), c("X", "Y", "Z"))
  nm <- ifelse(rest %in% names(LAS_NAMES), LAS_NAMES[rest], rest)
  new_cloud(as.matrix(d[c("X", "Y", "Z")]), stats::setNames(as.list(d[rest]), nm))
}

#' Convert a cloud to a lidR LAS object
#'
#' Needs the lidR package. Standard LAS dimensions are renamed to lidR's
#' names (`intensity` to `Intensity`, `gps_time` to `gpstime`, `red` to
#' `R`, ...); other attributes are kept under their own names.
#'
#' @param cloud A `sylva_cloud`.
#' @param crs Optional CRS for the LAS header (anything `sf::st_crs` reads).
#' @return A lidR `LAS`.
#' @export
as_las <- function(cloud, crs = NULL) {
  if (!requireNamespace("lidR", quietly = TRUE)) {
    stop("as_las() needs the lidR package: install.packages(\"lidR\")", call. = FALSE)
  }
  xyz <- xyz_of(cloud)
  d <- data.frame(X = xyz[, 1], Y = xyz[, 2], Z = xyz[, 3])
  a <- attrs_of(cloud)
  to_las <- stats::setNames(names(LAS_NAMES), LAS_NAMES)
  to_las["scan_angle"] <- "ScanAngleRank"
  for (k in names(a)) {
    v <- a[[k]]
    name <- if (k %in% names(to_las)) to_las[[k]] else k
    if (name == "ScanAngleRank" && any(v != round(v), na.rm = TRUE)) name <- "ScanAngle"
    if (name %in% LAS_INTEGER) v <- as.integer(v)
    if (name %in% c("Synthetic_flag", "Keypoint_flag", "Withheld_flag", "Overlap_flag")) v <- as.logical(v)
    d[[name]] <- v
  }
  las <- lidR::LAS(d)
  if (!is.null(crs)) lidR::st_crs(las) <- crs
  las
}
