# Registration against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_registration.py).

expected <- load_expected("registration")
tol <- 1e-9

test_that("transform matrices match Python", {
  expect_equal(rotation_z(33), expected$rz, tolerance = tol)
  expect_equal(rotation_z(-181.5), expected$rz_neg, tolerance = tol)
  expect_equal(translation(1.5, -2, 0.25), expected$t, tolerance = tol)
})

test_that("kabsch matches Python", {
  k <- as.matrix(read.csv(test_path("fixtures", "registration", "kabsch.csv"), header = FALSE))
  expect_equal(kabsch(k[, 1:3], k[, 4:6]), expected$kabsch, tolerance = tol)
})

test_that("ICP matches Python", {
  xyz <- as.matrix(read.csv(test_path("fixtures", "registration", "scene.csv"), header = FALSE))
  target <- point_cloud(xyz)
  source <- transform(point_cloud(xyz[seq(1, nrow(xyz), by = 2), ]), expected$icp_source_transform)
  for (method in c("point", "plane")) {
    r <- icp(source, target, method = method, max_correspondence_distance = 0.6, max_iterations = 40)
    expect_equal(r$transform, expected[[paste0(method, "_t")]], tolerance = tol)
    expect_equal(r$info$rmse, expected[[paste0(method, "_rmse")]], tolerance = tol)
    expect_equal(r$info$iterations, expected[[paste0(method, "_iter")]])
    expect_equal(r$info$n_correspondences, expected[[paste0(method, "_n")]])
  }
  r <- icp(source, target, init = translation(0.1, -0.05, 0), method = "plane", trim = 0.8, normal_k = 8, tolerance = 1e-8)
  expect_equal(r$transform, expected$trim_t, tolerance = tol)
  expect_equal(r$info$rmse, expected$trim_rmse, tolerance = tol)
  expect_equal(r$info$n_correspondences, expected$trim_n)
})

test_that("merged scans match Python", {
  a <- fixture_cloud("registration", "merge_a")
  b <- fixture_cloud("registration", "merge_b")
  m1 <- merge_scans(list(a, b), list(rotation_z(10), translation(5, 0, 1)))
  expect_equal(m1$xyz, expected$m1_xyz, tolerance = tol)
  expect_equal(m1$scan_id, expected$m1_scan)
  expect_equal(m1$i, expected$m1_i)
  expect_equal(sort(names(m1$attrs)), expected$m1_names)
  m2 <- merge_scans(list(a, b), scan_ids = FALSE)
  expect_equal(m2$xyz, expected$m2_xyz, tolerance = tol)
  expect_equal(sort(names(m2$attrs)), expected$m2_names)
  xyz <- as.matrix(read.csv(test_path("fixtures", "registration", "scene.csv"), header = FALSE))
  expect_equal(abs(estimate_normals(point_cloud(xyz[1:1500, ]), k = 9)), expected$normals, tolerance = tol)
})
