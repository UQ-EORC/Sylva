test_that("se3_exp and se3_log are inverse", {
  xi <- c(0.3, -0.2, 1.1, 4, -2, 0.5)
  expect_equal(se3_log(se3_exp(xi)), xi, tolerance = 1e-12)
  expect_equal(invert(se3_exp(xi)) %*% se3_exp(xi), diag(4), tolerance = 1e-12)
})

test_that("a pose graph recovers a loop and refuses bad edges", {
  truth <- list(diag(4), se3_exp(c(0, 0, 0.3, 8, 1, 0)), se3_exp(c(0, 0, -0.2, 3, 9, 0.1)))
  g <- pose_graph(3)
  for (e in list(c(1, 2), c(2, 3), c(1, 3))) {
    g <- add_edge(g, e[1], e[2], invert(truth[[e[2]]]) %*% truth[[e[1]]], n_correspondences = 500)
  }
  r <- optimise(g)
  expect_s3_class(r, "sylva_optimisation_result")
  for (k in 1:3) expect_equal(r$poses[[k]], truth[[k]], tolerance = 1e-9)
  expect_error(add_edge(g, 2, 2, diag(4)), "self-edges")
  expect_error(add_edge(g, 1, 4, diag(4)), "out of range")
  empty <- optimise(pose_graph(2))
  expect_equal(empty$iterations, 0L)
  expect_true(empty$converged)
})

test_that("optimise() still minimises functions", {
  expect_equal(optimise(function(x) (x - 2)^2, c(0, 5))$minimum, 2, tolerance = 1e-4)
})

test_that("a stem map survives a round trip through its file", {
  m <- stem_map(coreg_stem(c(1, 2.5), c(-1, 3), c(0, 0.2), c(0.3, 0.45), n_slices = 4L, rmse = 0.004, coverage = 0.4),
                name = "plot")
  path <- tempfile(fileext = ".json")
  save_stem_map(m, path)
  back <- load_stem_map(path)
  expect_equal(unclass(back)$stems, unclass(m)$stems)
  expect_equal(unclass(back)$name, "plot")
  expect_equal(length(top(m, 1)), 1L)
})
