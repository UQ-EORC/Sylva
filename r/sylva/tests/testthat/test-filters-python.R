# The filters against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_filters.py).

expected <- load_expected("filters")
tol <- 1e-9

test_that("subsampling selects the points Python selects", {
  c <- fixture_cloud("filters", "cloud1")
  v <- voxel_downsample(c, 0.7)
  expect_equal(v$id, expected$voxel_first_id)
  expect_equal(v$xyz, expected$voxel_first_xyz, tolerance = tol)
  expect_equal(voxel_downsample(c, 0.7, method = "centroid")$xyz, expected$voxel_centroid, tolerance = tol)
  expect_equal(random_subsample(c, n = 500, seed = 3)$id, expected$random_n_id)
  expect_equal(random_subsample(c, fraction = 0.125, seed = 4)$id, expected$random_fraction_id)
  expect_equal(min_distance_subsample(c, 0.6)$id, expected$min_distance_id)
  expect_error(random_subsample(c), "exactly one")
})

test_that("crops keep the points Python keeps", {
  c <- fixture_cloud("filters", "cloud2")
  expect_equal(crop_box(c, c(1, 2, 3), c(6, 7.5, 9))$id, expected$box_id)
  expect_equal(crop_box(c, list(NULL, 2, NaN), list(5, NULL, 4))$id, expected$box_open_id)
  expect_equal(crop_box(c, c(NA, 2, NA), c(5, NA, 4))$id, expected$box_open_id)
  expect_equal(crop_cylinder(c, c(5, 4), 3.2)$id, expected$cylinder_id)
  expect_equal(crop_cylinder(c, c(5, 4), 3.2, zmin = 1, zmax = 6.5)$id, expected$cylinder_z_id)
  expect_equal(range_filter(c, c(2, 3, 1), min_range = 2, max_range = 7)$id, expected$range_id)
  expect_equal(range_filter(c, max_range = 9)$id, expected$range_default_id)
})

test_that("outlier removal agrees with Python", {
  c <- fixture_cloud("filters", "cloud3")
  expect_equal(as.numeric(statistical_outlier_removal(c, k = 8, std_ratio = 1.5, return_mask = TRUE)), expected$sor_mask)
  expect_equal(statistical_outlier_removal(c)$id, expected$sor_id)
  expect_equal(as.numeric(radius_outlier_removal(c, 0.8, min_neighbors = 3, return_mask = TRUE)), expected$ror_mask)
  expect_equal(radius_outlier_removal(c, 0.8)$id, expected$ror_id)
})

test_that("local geometry, clusters and neighbours agree with Python", {
  c <- fixture_cloud("filters", "cloud4")
  expect_equal(abs(estimate_normals(c, k = 10)), expected$normals, tolerance = tol)
  expect_equal(abs(estimate_normals(c)), expected$normals_registration, tolerance = tol)
  pl <- planarity_linearity(c, k = 15)
  expect_equal(pl$planarity, expected$planarity, tolerance = tol)
  expect_equal(pl$linearity, expected$linearity, tolerance = tol)
  expect_equal(euclidean_clusters(c, 0.9, min_points = 4), expected$clusters)
  q <- c$xyz[seq(1, length(c), by = 50), ] + 0.01
  nn <- knn(c$xyz, q, 6)
  expect_equal(nn$distances, expected$knn_d, tolerance = tol)
  expect_equal(nn$indices, expected$knn_i)
  expect_true(all(is.na(knn(c$xyz[1:2, ], q[1:3, ], 4)$indices[, 3:4])))
})
