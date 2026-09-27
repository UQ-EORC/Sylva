# Ground classification and terrain against the Python package on the same
# inputs (fixtures written by tests/parity/export_r_ground.py).

expected <- load_expected("ground")
tol <- 1e-9

expect_raster <- function(r, prefix) {
  expect_s3_class(r, "sylva_raster")
  expect_equal(unclass(r)$data, expected[[paste0(prefix, "_data")]], tolerance = tol)
  expect_equal(c(unclass(r)$xmin, unclass(r)$ymin, unclass(r)$resolution), expected[[paste0(prefix, "_origin")]], tolerance = tol)
}

test_that("CSF and PMF classify as in Python", {
  c <- fixture_cloud("ground", "plot")
  expect_equal(as.numeric(classify_ground_csf(c, return_mask = TRUE)), expected$csf_mask)
  csf <- classify_ground_csf(c, cloth_resolution = 0.5, rigidness = 2, class_threshold = 0.3)
  expect_equal(csf$classification, expected$csf_class)
  expect_type(csf$classification, "integer")
  expect_equal(as.numeric(classify_ground_csf(c, cloth_resolution = 0.8, rigidness = 3, class_threshold = 0.2,
                                              iterations = 300, time_step = 0.5, return_mask = TRUE)),
               expected$csf_r3_mask)
  expect_equal(as.numeric(classify_ground_pmf(c, return_mask = TRUE)), expected$pmf_mask)
  pmf <- classify_ground_pmf(c, cell_size = 0.4, max_window = 6, slope = 0.2, initial_distance = 0.1, max_distance = 1.5)
  expect_equal(pmf$classification, expected$pmf_class)
  expect_equal(as.numeric(ground_mask(csf)), expected$ground_mask)
  expect_error(ground_mask(c), "classification")
})

test_that("DTM, heights and CHM match Python", {
  c <- classify_ground_csf(fixture_cloud("ground", "plot"))
  dtm <- make_dtm(c, resolution = 0.5)
  expect_raster(dtm, "dtm")
  dtm_b <- make_dtm(c, resolution = 0.75, bounds = c(-1, -0.5, 13, 12.5))
  expect_raster(dtm_b, "dtm_b")
  n <- normalize_height(c, dtm)
  expect_equal(n$height, expected$height, tolerance = tol)
  expect_equal(normalize_height(c, dtm_b, attr = "hag")$hag, expected$hag, tolerance = tol)
  f <- flatten(c, dtm)
  expect_equal(f$xyz, expected$flat_xyz, tolerance = tol)
  expect_equal(f$classification, expected$flat_class)
  expect_raster(make_chm(n, resolution = 1), "chm")
  expect_raster(make_chm(f, resolution = 0.8, height_attr = "missing", bounds = c(0, 0, 12, 12), min_height = 2), "chm_b")
})

test_that("heights above a given raster match Python", {
  grid <- as.matrix(read.csv(test_path("fixtures", "ground", "grid.csv"), header = FALSE))
  pts <- point_cloud(as.matrix(read.csv(test_path("fixtures", "ground", "points.csv"), header = FALSE)))
  r <- raster(grid, 1, -2, 0.5)
  expect_equal(normalize_height(pts, r)$height, expected$given_height, tolerance = tol)
  expect_equal(flatten(pts, r)$xyz, expected$given_flat, tolerance = tol)
})
