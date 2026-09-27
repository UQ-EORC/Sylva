# The R tree API against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_trees.py).

fixture <- function(...) test_path("fixtures", "trees", ...)
source(fixture("expected.R"), local = TRUE)

read_matrix <- function(name) unname(as.matrix(read.csv(fixture(paste0(name, ".csv")), header = FALSE)))
read_vector <- function(name) scan(fixture(paste0(name, ".csv")), quiet = TRUE)

expect_table <- function(got, want, info = NULL) {
  for (k in names(want)) {
    expect_equal(as.double(got[[k]]), want[[k]], tolerance = 1e-9, info = paste(info, k))
  }
}

xyz <- read_matrix("plot_xyz")
cloud <- point_cloud(xyz, height = xyz[, 3])

test_that("circle fits and hulls match Python", {
  xy <- read_matrix("circle_xy")
  f <- fit_circle(xy)
  expect_equal(c(f$cx, f$cy, f$r, f$rmse), expected$fit_circle, tolerance = 1e-9)
  r <- fit_circle_ransac(xy, threshold = 0.01, seed = 3)
  expect_equal(c(r$cx, r$cy, r$r), expected$ransac, tolerance = 1e-9)
  expect_equal(as.double(r$inliers), expected$ransac_inliers)
  expect_equal(convex_hull_area(xy), expected$hull, tolerance = 1e-9)
})

test_that("stem detection, merging, segmentation and heights match Python", {
  found <- detect_stems(cloud)
  expect_s3_class(found, "data.frame")
  expect_table(found, expected$stems, "stems")
  m <- merge_branches(cloud, found)
  expect_table(m$trees, expected$merged, "merged")
  expect_equal(as.double(m$merged_into), expected$merged_into)
  labels <- segment_trees(cloud, m$trees)
  expect_equal(labels, as.integer(read_vector("labels")))
  kept <- tree_heights(cloud, labels, m$trees)
  expect_table(kept, expected$heights, "heights")
  prof <- dbh_profile(cloud, c(kept$x[1], kept$y[1]), heights = c(1, 2, 3.5))
  expect_equal(unname(prof), expected$dbh_profile, tolerance = 1e-9)
})

test_that("crown metrics and shape match Python", {
  labels <- as.integer(read_vector("labels"))
  expect_table(crown_metrics_all(cloud, labels), expected$crowns, "crowns")
  one <- crown_metrics(cloud, labels, 1)
  for (k in names(expected$crown_1)) expect_equal(one[[k]], expected$crown_1[[k]], tolerance = 1e-9, info = k)
  expect_length(crown_metrics(cloud, labels, 99), 0)
  s <- crown_shape(cloud[labels == 1], base_xy = c(expected$heights$x[1], expected$heights$y[1]), crown_base = 5)
  for (k in names(expected$crown_shape)) {
    expect_equal(s[[k]], expected$crown_shape[[k]], tolerance = 1e-9, info = k)
  }
})

test_that("pruning, relabelling and basal area match Python", {
  m <- read_matrix("candidates")
  cands <- tree(m[, 1], m[, 2], m[, 3], dbh = m[, 4], height = m[, 5], n_points = m[, 6],
                inlier_fraction = m[, 7], n_slices = m[, 8], rmse = m[, 9], lean_deg = m[, 10], quality = m[, 11],
                plot = m[, 12])
  labels <- read_vector("candidate_labels")
  p0 <- prune_trees(cands, labels)
  expect_table(p0$trees, expected$prune_0, "prune_0")
  expect_equal(as.double(p0$labels), expected$prune_0_labels)
  p1 <- prune_trees(cands, labels, merge_radius = 2, max_dbh = 0.9, min_quality_short = 0.4)
  expect_table(p1$trees, expected$prune_1, "prune_1")
  expect_equal(as.double(p1$labels), expected$prune_1_labels)
  expect_equal(basal_area(cands, 2500, min_dbh = 0.1), expected$basal_area, tolerance = 1e-9)
  expect_equal(basal_area(cands$dbh, 2500, min_dbh = 0.1), expected$basal_area, tolerance = 1e-9)
  expect_error(basal_area(cands, 0))
})

test_that("buttress detection matches Python", {
  b <- read_matrix("buttress_xyz")
  base <- point_cloud(b, height = read_vector("buttress_h"))
  check <- function(got, want) {
    for (k in names(want)) expect_equal(as.double(got[[k]]), want[[k]], tolerance = 1e-9, info = k)
  }
  check(detect_buttress(base, base_xy = c(3, -2)), expected$buttress)
  check(detect_buttress(base), expected$buttress_auto)
  check(detect_buttress(base, bark_only = FALSE, voxel = 0, bins = 24), expected$buttress_raw)
})
