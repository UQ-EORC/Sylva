# The R quality API against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_trees.py).

fixture <- function(...) test_path("fixtures", "quality", ...)
source(fixture("expected.R"), local = TRUE)

read_matrix <- function(name) unname(as.matrix(read.csv(fixture(paste0(name, ".csv")), header = FALSE)))
read_vector <- function(name) scan(fixture(paste0(name, ".csv")), quiet = TRUE)

test_that("stem noise and its summary match Python", {
  xyz <- read_matrix("stems_xyz")
  cloud <- point_cloud(xyz, height = xyz[, 3])
  q <- stem_noise(cloud, scan_id = read_vector("scan_ids"), stems = read_matrix("stems"), thickness = 0.15,
                  min_scan_points = 5)
  expect_s3_class(q, "sylva_stem_noise")
  expect_gt(nrow(q$slices), 10)
  for (tab in c("slices", "scan_slices", "scans")) {
    for (k in names(expected[[tab]])) {
      expect_equal(as.double(q[[tab]][[k]]), expected[[tab]][[k]], tolerance = 1e-9, info = paste(tab, k))
    }
  }
  expect_equal(q$residual, expected$residual, tolerance = 1e-9)
  s <- summary(q)
  expect_equal(names(s), names(expected$summary))
  for (k in names(expected$summary)) expect_equal(s[[k]], expected$summary[[k]], tolerance = 1e-9, info = k)
  s4 <- summary(q, min_scan_slices = 4)
  for (k in names(expected$summary_4)) expect_equal(s4[[k]], expected$summary_4[[k]], tolerance = 1e-9, info = k)
})

test_that("a summary of hand-made tables skips unsupported scans", {
  q <- structure(list(
    slices = data.frame(stem = c(0, 0), height = c(1.5, 2), n_points = c(100, 100), sigma = c(0.005, 0.005),
                        sigma_first = c(0.006, 0.006), tail_fraction = c(0, 0)),
    scan_slices = data.frame(scan = c(0, 1), slice = c(0, 1), n_points = c(50, 50), sigma_within = c(0.004, 0.004),
                             sigma_local = c(0.003, 0.003)),
    scans = data.frame(scan = 0:2, n_points = c(500, 500, 5), n_slices = c(4, 4, 0), tx = c(0.002, -0.002, 0.6),
                       ty = c(0, 0, 0)),
    residual = numeric(0)), class = "sylva_stem_noise")
  s <- summary(q)
  expect_equal(s$n_scans_registered, 2)
  expect_equal(s$registration_max, 0.002)
  expect_equal(summary(q, min_scan_slices = 0)$registration_max, 0.6)
})

test_that("scan ids from origins match Python", {
  o <- read_matrix("origins")
  expect_equal(as.double(scan_ids_from_origins(o)), expected$scan_ids)
  expect_equal(as.double(scan_ids_from_origins(o, tolerance = 5)), expected$scan_ids_coarse)
})
