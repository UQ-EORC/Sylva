# The point cloud container against the Python package's PointCloud
# (fixtures written by tests/parity/export_r_pointcloud.py).

expected <- load_expected("pointcloud")
tol <- 1e-9

test_that("methods match Python", {
  c <- fixture_cloud("pointcloud", "cloud21")
  t <- transform(c, expected$matrix)
  expect_equal(t$xyz, expected$transform, tolerance = tol)
  expect_equal(t$k, expected$transform_k)
  expect_equal(heights(c), expected$heights, tolerance = tol)
  expect_equal(heights(c, "absent"), expected$heights_z, tolerance = tol)
  b <- bounds(c)
  expect_equal(unname(b$min), expected$lo, tolerance = tol)
  expect_equal(unname(b$max), expected$hi, tolerance = tol)
  sub <- c[c$z > 0.5]
  expect_equal(sub$xyz, expected$sub_xyz, tolerance = tol)
  expect_equal(sub$k, expected$sub_k)
  expect_equal(c[c(6, 4, 4, 1)]$xyz, expected$idx_xyz, tolerance = tol)
  expect_equal(c[seq(11, 20, by = 3)]$xyz, expected$slice_xyz, tolerance = tol)
  expect_equal(with_attrs(c, k = rep(0, length(c)), new = seq_len(length(c)) - 1L)$new, expected$with)
  expect_equal(sort(names(without(c, "k", "none")$attrs)), expected$without)
  expect_error(transform(c, diag(3)), "4x4")
})

test_that("concatenation keeps the common attributes, as in Python", {
  a <- fixture_cloud("pointcloud", "cloud22")
  b <- with_attrs(fixture_cloud("pointcloud", "cloud23"), extra = rep(1, 25))
  c <- point_cloud(matrix(0, 3, 3), k = c(1.5, 2.5, 3.5), height = c(0, 0, 0))
  ab <- concatenate(a, b)
  expect_equal(ab$xyz, expected$ab_xyz, tolerance = tol)
  expect_equal(ab$k, expected$ab_k)
  expect_equal(names(ab$attrs), expected$ab_names)
  abc <- concatenate(list(a, b, c))
  expect_equal(abc$k, expected$abc_k)
  expect_equal(abc$height, expected$abc_height, tolerance = tol)
  expect_error(concatenate(list()), "no clouds")
})

test_that("clouds build from arrays as in Python", {
  a <- as.matrix(read.csv(test_path("fixtures", "pointcloud", "array.csv"), header = FALSE))
  colnames(a) <- NULL
  c1 <- from_array(a)
  expect_equal(c1$xyz, expected$c1_xyz, tolerance = tol)
  expect_equal(sort(names(c1$attrs)), expected$c1_names)
  expect_equal(c1$col5, expected$c1_col5, tolerance = tol)
  c2 <- from_array(a, c(NA, "intensity"))
  expect_equal(sort(names(c2$attrs)), expected$c2_names)
  expect_equal(c2$intensity, expected$c2_intensity, tolerance = tol)
})

test_that("clouds convert to and from data frames", {
  c <- fixture_cloud("pointcloud", "cloud21")
  d <- as.data.frame(c)
  expect_named(d, c("x", "y", "z", "height", "k"))
  back <- as_point_cloud(d)
  expect_equal(back$xyz, c$xyz)
  expect_equal(back$k, c$k)
  expect_equal(c$x, c$xyz[, 1])
  expect_equal(c[["k"]], c$k)
})

test_that("clouds convert to and from lidR", {
  skip_if_not_installed("lidR")
  c <- fixture_cloud("pointcloud", "cloud21")
  c <- with_attrs(c, intensity = seq_len(length(c)), classification = rep(2L, length(c)), gps_time = seq_len(length(c)) / 10)
  las <- suppressMessages(as_las(c))
  expect_s4_class(las, "LAS")
  expect_equal(las@data$Intensity, seq_len(length(c)))
  expect_type(las@data$Classification, "integer")
  back <- as_point_cloud(las)
  expect_equal(back$intensity, c$intensity)
  expect_equal(back$gps_time, c$gps_time)
  expect_equal(back$k, c$k)
  expect_equal(back$xyz, c$xyz, tolerance = 1e-3)
})
