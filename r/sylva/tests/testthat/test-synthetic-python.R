# Synthetic scenes in R against the Python package with the same seeds
# (fixtures written by tests/parity/export_r_synthetic.py).

source(test_path("fixtures", "synthetic", "expected.R"), local = TRUE)

expect_cloud <- function(got, want) {
  xyz <- unclass(got)$xyz
  expect_equal(nrow(xyz), want$n)
  expect_equal(colSums(xyz), want$sum, tolerance = 1e-9)
  expect_equal(unname(xyz[seq(1, nrow(xyz), by = 100), , drop = FALSE]), want$rows, tolerance = 1e-9)
  for (k in setdiff(names(want), c("n", "sum", "rows"))) {
    expect_equal(tabulate(unclass(got)$attrs[[k]] + 1L, length(want[[k]])), want[[k]], info = k)
  }
}

trees <- data.frame(x = c(2, 4), y = c(2.5, 1), dbh = c(0.2, 0.1), height = c(1.2, 0.8))

test_that("terrain heights match Python", {
  x <- seq(-5, 25, length.out = 13)
  y <- seq(30, -3, length.out = 13)
  expect_equal(synthetic_terrain_height(x, y), expected$terrain, tolerance = 1e-12)
  expect_equal(synthetic_terrain_height(x, 2, slope = 0.2), expected$terrain_slope, tolerance = 1e-12)
  m <- matrix(x[1:12], 3)
  expect_equal(dim(synthetic_terrain_height(m, 1)), c(3L, 4L))
})

test_that("a tree and its leaf area match Python", {
  t <- synthetic_tree(1, -1, dbh = 0.25, height = 1.4, z0 = 0.2, n_branches = 3, leaf_points = 120, seed = 6)
  expect_cloud(t, expected$tree)
  expect_equal(synthetic_leaf_area(t), expected$tree_leaf_area, tolerance = 1e-12)
})

test_that("a forest and its scan match Python", {
  f <- synthetic_forest(trees, size = 5, ground_points = 500, margin = 1, seed = 3)
  expect_cloud(f, expected$forest)
  expect_equal(synthetic_leaf_area(f), expected$forest_leaf_area, tolerance = 1e-12)
  s <- unclass(synthetic_scan(f, origin = c(3, 3, 1), resolution_deg = 4, max_echoes = 3, echo_separation = 0.3))
  w <- expected$scan
  expect_equal(nrow(s$origin), w$n_shots)
  expect_equal(s$echo_count, w$echo_count)
  expect_equal(s$echo_range, w$echo_range, tolerance = 1e-9)
  expect_equal(unname(s$direction[seq(1, nrow(s$direction), by = 50), ]), w$direction, tolerance = 1e-9)
  expect_equal(colSums(s$direction), w$direction_sum, tolerance = 1e-9)
  expect_equal(as.double(s$echo_attrs$classification), w$classification)
  expect_equal(as.double(s$echo_attrs$tree_id), w$tree_id)
})

test_that("the default forest has the default trees", {
  expect_equal(nrow(SYNTHETIC_DEFAULT_TREES), 4)
  g <- synthetic_forest(trees[0, ], size = 5, ground_points = 50, seed = 1)
  expect_equal(length(g), 50)
})
