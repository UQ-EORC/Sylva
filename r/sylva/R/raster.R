# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

#' A raster grid
#'
#' A regular 2-D grid in map coordinates, used for DTMs and CHMs, as the
#' Python package's `Raster`. `data[row, col]` covers x from
#' `xmin + (col - 1) * resolution` and y from `ymin + (row - 1) *
#' resolution`, so row 1 is the *southern* edge (the matrix is upside down
#' compared with an image); `to_geotiff()` and `to_ascii_grid()` flip it to
#' the usual north-up order on export.
#'
#' @param data Numeric matrix; `NA` for cells without data.
#' @param xmin,ymin South-west corner of the grid (not of the first cell centre).
#' @param resolution Cell size in the units of x and y (m).
#' @param crs Optional CRS string (`"EPSG:28355"`) used by `to_geotiff()`.
#'   Sylva never reprojects.
#' @return A `sylva_raster`.
#' @export
raster <- function(data, xmin, ymin, resolution, crs = NULL) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  dimnames(data) <- NULL
  structure(list(data = data, xmin = as.double(xmin), ymin = as.double(ymin),
                 resolution = as.double(resolution), crs = if (is.null(crs)) NULL else as.character(crs)),
            class = "sylva_raster")
}

as_raster <- function(r, crs = NULL) raster(r$data, r$xmin, r$ymin, r$resolution, crs)

raster_args <- function(r) {
  r <- unclass(r)
  list(r$data, r$xmin, r$ymin, r$resolution)
}

#' @export
dim.sylva_raster <- function(x) dim(unclass(x)$data)

#' @export
print.sylva_raster <- function(x, ...) {
  r <- unclass(x)
  cat(sprintf("<sylva_raster> %d x %d cells of %g, x %g to %g, y %g to %g%s\n", nrow(r$data), ncol(r$data),
              r$resolution, r$xmin, xmax(x), r$ymin, ymax(x), if (is.null(r$crs)) "" else paste0(", ", r$crs)))
  invisible(x)
}

#' Raster extent and cell geometry
#'
#' `xmax()` and `ymax()`: the eastern and northern edges. `cell_centers()`:
#' cell-centre coordinates as two matrices aligned with `data`.
#' `cell_index()`: the cell containing each coordinate, as 1-based row and
#' column numbers that are *not* clipped, so points outside the grid get
#' numbers below 1 or beyond `dim(r)`; `NA` for a missing coordinate.
#'
#' @param r A `sylva_raster`.
#' @param x,y Coordinates.
#' @return `xmax()`, `ymax()`: a number. `cell_centers()`: `list(X, Y)`.
#'   `cell_index()`: `list(row, col)`, shaped as `y` and `x`.
#' @export
xmax <- function(r) unclass(r)$xmin + ncol(unclass(r)$data) * unclass(r)$resolution

#' @rdname xmax
#' @export
ymax <- function(r) unclass(r)$ymin + nrow(unclass(r)$data) * unclass(r)$resolution

#' @rdname xmax
#' @export
cell_centers <- function(r) {
  d <- dim(r)
  u <- unclass(r)
  core_raster_cell_centers(d[1], d[2], u$xmin, u$ymin, u$resolution)
}

#' @rdname xmax
#' @export
cell_index <- function(r, x, y) {
  u <- unclass(r)
  i <- core_raster_cell_index(u$xmin, u$ymin, u$resolution, as.double(x), as.double(y))
  shape <- function(v, like) {
    v[is.nan(v)] <- NA
    if (!is.null(dim(like))) dim(v) <- dim(like)
    v
  }
  list(row = shape(i$row, y), col = shape(i$col, x))
}

#' Interpolate a raster at arbitrary coordinates
#'
#' Bilinear interpolation between cell centres (the Python package's
#' `Raster.sample`). Coordinates beyond the outer cell centres take the edge
#' value; a missing neighbour makes the result `NaN`, so call
#' `fill_nearest()` first if the grid has holes.
#'
#' @param r A `sylva_raster`.
#' @param x,y Coordinates (flattened to vectors).
#' @return One value per coordinate.
#' @export
raster_sample <- function(r, x, y) do.call(core_raster_sample, c(raster_args(r), list(as.double(x), as.double(y))))

#' Fill missing cells from the nearest valid cell
#'
#' @param r A `sylva_raster`.
#' @return A copy with every missing cell filled.
#' @export
fill_nearest <- function(r) {
  u <- unclass(r)
  raster(core_raster_fill_nearest(u$data), u$xmin, u$ymin, u$resolution, u$crs)
}

#' ESRI ASCII grids
#'
#' `to_ascii_grid()` writes a raster north-up, with missing cells as
#' `nodata`; `from_ascii_grid()` reads one written by it or by GIS software,
#' `NODATA_value` cells becoming `NA` (ASCII grids carry no CRS).
#'
#' @param r A `sylva_raster`.
#' @param path The `.asc` file.
#' @param nodata Value written for missing cells.
#' @return `from_ascii_grid()`: a `sylva_raster` with row 1 at the south.
#' @export
to_ascii_grid <- function(r, path, nodata = -9999) {
  invisible(do.call(core_write_ascii_grid, c(list(path.expand(path)), raster_args(r), list(as.double(nodata)))))
}

#' @rdname to_ascii_grid
#' @export
from_ascii_grid <- function(path) as_raster(core_read_ascii_grid(path.expand(path)))

#' Write a GeoTIFF
#'
#' A single-band float32 GeoTIFF, north-up, missing cells as NaN. Needs the
#' terra package (GeoTIFF export is left to each language's GIS library).
#'
#' @param r A `sylva_raster`.
#' @param path Output file.
#' @param crs CRS to write; defaults to the raster's. With neither, the file
#'   has a geotransform but no CRS.
#' @export
to_geotiff <- function(r, path, crs = NULL) {
  if (!requireNamespace("terra", quietly = TRUE)) {
    stop("to_geotiff() needs the terra package: install.packages(\"terra\")", call. = FALSE)
  }
  u <- unclass(r)
  crs <- if (is.null(crs)) u$crs else crs
  d <- dim(u$data)
  g <- terra::rast(nrows = d[1], ncols = d[2], xmin = u$xmin, xmax = xmax(r), ymin = u$ymin, ymax = ymax(r), crs = "")
  if (!is.null(crs)) terra::crs(g) <- crs
  terra::values(g) <- as.vector(t(u$data[rev(seq_len(d[1])), , drop = FALSE]))
  terra::writeRaster(g, path.expand(path), datatype = "FLT4S", NAflag = NaN, overwrite = TRUE,
                     filetype = "GTiff")
  invisible(path)
}
