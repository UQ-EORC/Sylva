# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

as_cloud <- function(x) structure(x, class = "sylva_cloud")

#' A point cloud
#'
#' Points with optional per-point attributes, as in the Python package's
#' `PointCloud`. Coordinates are metres in whatever frame the data came in.
#'
#' @param xyz An `n x 3` numeric matrix (or anything `as.matrix` turns into one).
#' @param ... Per-point attributes as named vectors of length `n`.
#' @return A `sylva_cloud`: a list with `xyz` and `attrs`.
#' @export
point_cloud <- function(xyz, ...) {
  xyz <- as.matrix(xyz)
  storage.mode(xyz) <- "double"
  as_cloud(core_cloud(list(xyz = xyz, attrs = list(...))))
}

#' Read a point cloud
#'
#' LAS/LAZ, PLY, ASCII (`.xyz`, `.txt`, `.csv`, `.asc`, `.pts`) or a RIEGL
#' `.rxp` scan (which needs RiVLib), chosen by the extension.
#'
#' @param path File to read.
#' @return A `sylva_cloud`.
#' @export
read_cloud <- function(path) as_cloud(core_read(path.expand(path)))

#' Write a point cloud
#'
#' @param cloud A `sylva_cloud`.
#' @param path Output file; the format follows the extension.
#' @export
write_cloud <- function(cloud, path) core_write(unclass(cloud), path.expand(path))

#' One point per voxel
#'
#' @param cloud A `sylva_cloud`.
#' @param voxel_size Voxel edge (m).
#' @param method `"first"` keeps the first point of each voxel with its
#'   attributes; `"centroid"` returns voxel centroids.
#' @return A `sylva_cloud`.
#' @export
voxel_downsample <- function(cloud, voxel_size, method = c("first", "centroid")) {
  method <- match.arg(method)
  as_cloud(core_voxel_downsample(unclass(cloud), as.double(voxel_size), method == "centroid"))
}

#' @export
length.sylva_cloud <- function(x) nrow(unclass(x)$xyz)

#' @export
print.sylva_cloud <- function(x, ...) {
  a <- names(unclass(x)$attrs)
  cat(sprintf("<sylva_cloud> %s points%s\n", format(length(x), big.mark = ","),
              if (length(a)) paste0(", attrs: ", paste(a, collapse = ", ")) else ""))
  invisible(x)
}

#' @export
`[.sylva_cloud` <- function(x, i) {
  x <- unclass(x)
  as_cloud(list(xyz = x$xyz[i, , drop = FALSE], attrs = lapply(x$attrs, `[`, i)))
}
