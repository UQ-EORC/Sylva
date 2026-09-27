# The R API against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_fixtures.py).

fixture <- function(...) test_path("fixtures", "canopy", ...)
source(fixture("expected.R"), local = TRUE)

read_scan <- function(name) {
  rays <- as.matrix(read.csv(fixture(paste0(name, "_rays.csv")), header = FALSE))
  shots(origin = rays[, 1:3], direction = rays[, 4:6], echo_start = rays[, 7], echo_count = rays[, 8],
        echo_range = scan(fixture(paste0(name, "_range.csv")), quiet = TRUE))
}

heights <- function(s) echo_xyz(s)[, 3]

test_that("a gap profile reports what Python reports", {
  prof <- gap_profile()
  for (k in 0:1) {
    s <- read_scan(paste0("scan", k))
    prof <- add_scan(prof, s, heights(s))
  }
  r <- report(prof)
  for (k in names(expected$report)) {
    expect_equal(r[[k]], expected$report[[k]], tolerance = 1e-9, info = k)
  }
  expect_equal(pgap(prof), expected$pgap, tolerance = 1e-9)
  expect_equal(pai_profile(prof, "weighted"), expected$pai_weighted, tolerance = 1e-9)
  expect_equal(pavd_profile(prof, "linear"), expected$pavd_linear, tolerance = 1e-9)
  expect_equal(clumping(prof, 47.5), expected$clumping_47, tolerance = 1e-9)
})

test_that("fired pulses and pattern gap fractions match Python", {
  s <- read_scan("scan0")
  hit <- subset(s, unclass(s)$echo_count > 0)
  edges <- seq(5, 70, by = 5)
  expect_equal(fired_pulses_per_ring(hit, expected$pattern, edges), expected$fired_pattern, tolerance = 1e-9)
  expect_equal(fired_pulses_from_points(hit, edges), expected$fired_points, tolerance = 1e-9)
  g <- gap_fraction_pattern(hit, heights(hit), expected$pattern, min_height = 2)
  expect_equal(g$gap, expected$gap_pattern, tolerance = 1e-9)
})

test_that("the ground plane matches Python", {
  pts <- as.matrix(read.csv(fixture("ground_points.csv"), header = FALSE))
  expect_equal(fit_ground_plane(pts), expected$ground_plane, tolerance = 1e-9)
})
