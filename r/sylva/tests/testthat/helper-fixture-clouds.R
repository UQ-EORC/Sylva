# Clouds and rasters written by the tests/parity/export_r_*.py scripts.

fixture_cloud <- function(module, name) {
  d <- read.csv(test_path("fixtures", module, paste0(name, ".csv")))
  as_point_cloud(d)
}

load_expected <- function(module) {
  env <- new.env()
  source(test_path("fixtures", module, "expected.R"), local = env)
  env$expected
}
