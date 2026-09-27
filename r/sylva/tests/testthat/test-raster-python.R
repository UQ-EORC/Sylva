# Rasters against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_raster.py).

expected <- load_expected("raster")
tol <- 1e-9

grid <- function() {
  data <- as.matrix(read.csv(test_path("fixtures", "raster", "grid.csv"), header = FALSE))
  o <- expected$origin
  raster(data, o[1], o[2], o[3])
}

test_that("raster geometry matches Python", {
  r <- grid()
  i <- cell_index(r, expected$x, expected$y)
  expect_equal(i$row, expected$row)
  expect_equal(i$col, expected$col)
  cc <- cell_centers(r)
  expect_equal(cc$X, expected$X, tolerance = tol)
  expect_equal(cc$Y, expected$Y, tolerance = tol)
  s <- cell_index(r, 101, -39)
  expect_equal(c(s$row, s$col), expected$scalar_rc)
  expect_equal(c(xmax(r), ymax(r), dim(r)), expected$extent, tolerance = tol)
  m <- cell_index(r, matrix(c(100.3, NA, 101, 102), 2), 0)
  expect_equal(dim(m$col), c(2L, 2L))
  expect_true(is.na(m$col[2, 1]))
})

test_that("sampling and hole filling match Python", {
  r <- grid()
  f <- fill_nearest(r)
  expect_equal(unclass(f)$data, expected$filled, tolerance = tol)
  expect_equal(raster_sample(r, expected$sx, expected$sy), expected$sample_holes, tolerance = tol)
  expect_equal(raster_sample(f, expected$sx, expected$sy), expected$sample_filled, tolerance = tol)
  expect_equal(raster_sample(f, 101.3, -38.2), expected$sample_scalar, tolerance = tol)
})

test_that("ASCII grids round-trip as in Python", {
  r <- grid()
  p <- tempfile(fileext = ".asc")
  to_ascii_grid(r, p)
  back <- from_ascii_grid(p)
  expect_equal(unclass(back)$data, expected$ascii_data, tolerance = tol)
  expect_equal(c(unclass(back)$xmin, unclass(back)$ymin, unclass(back)$resolution), expected$ascii_origin, tolerance = tol)
  to_ascii_grid(r, p, nodata = -1)
  expect_equal(unclass(from_ascii_grid(p))$data, expected$ascii_data_nodata, tolerance = tol)
  expect_equal(readLines(p)[1:6], expected$ascii_header)
})

test_that("a GeoTIFF is written north-up through terra", {
  skip_if_not_installed("terra")
  r <- raster(matrix(c(1, 2, 3, NA, 5, 6), 2, byrow = TRUE), 10, 20, 0.5)
  p <- tempfile(fileext = ".tif")
  to_geotiff(r, p)
  t <- terra::rast(p)
  expect_equal(as.vector(terra::ext(t)), c(xmin = 10, xmax = 11.5, ymin = 20, ymax = 21))
  expect_equal(terra::values(t)[, 1], c(NA, 5, 6, 1, 2, 3))
})
