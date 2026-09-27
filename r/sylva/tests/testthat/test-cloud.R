test_that("a cloud survives a LAZ round trip with its attributes", {
  xyz <- cbind(x = c(0, 1, 2.5), y = c(0, 0, 1), z = c(10, 11, 12))
  cloud <- point_cloud(xyz, intensity = c(1L, 2L, 3L))
  path <- tempfile(fileext = ".laz")
  write_cloud(cloud, path)
  back <- read_cloud(path)
  expect_equal(length(back), 3L)
  expect_equal(unclass(back)$xyz, unclass(cloud)$xyz, ignore_attr = TRUE, tolerance = 1e-3)
  expect_equal(as.integer(unclass(back)$attrs$intensity), c(1L, 2L, 3L))
})

test_that("voxel thinning keeps one point per voxel", {
  xyz <- rbind(c(0.01, 0.01, 0.01), c(0.02, 0.02, 0.02), c(1.5, 0, 0))
  thinned <- voxel_downsample(point_cloud(xyz), 1)
  expect_equal(length(thinned), 2L)
})

test_that("subsetting keeps attributes aligned", {
  cloud <- point_cloud(matrix(1:12, ncol = 3), id = 1:4)
  sub <- cloud[c(2, 4)]
  expect_equal(unclass(sub)$attrs$id, c(2L, 4L))
  expect_equal(unclass(sub)$xyz[, 1], c(2, 4))
})
