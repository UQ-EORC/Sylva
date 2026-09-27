# Pulses in R against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_shots.py).

fixture <- function(...) test_path("fixtures", "shots", ...)
source(fixture("expected.R"), local = TRUE)

read_shots_fixture <- function(name) {
  rays <- unname(as.matrix(read.csv(fixture(paste0(name, "_rays.csv")), header = FALSE)))
  attrs <- as.list(read.csv(fixture(paste0(name, "_attrs.csv"))))
  shots(origin = rays[, 1:3], direction = rays[, 4:6], echo_start = rays[, 7], echo_count = rays[, 8],
        echo_range = scan(fixture(paste0(name, "_range.csv")), quiet = TRUE), echo_attrs = attrs)
}

expect_shots <- function(got, want) {
  g <- unclass(got)
  for (k in c("origin", "direction", "echo_start", "echo_count", "echo_range")) {
    expect_equal(unname(g[[k]]), want[[k]], tolerance = 1e-9, info = k)
  }
  if (!is.null(want$attrs$none)) {
    expect_length(g$echo_attrs, 0)
  } else {
    expect_equal(sort(names(g$echo_attrs)), names(want$attrs))
    for (k in names(want$attrs)) expect_equal(as.double(g$echo_attrs[[k]]), want$attrs[[k]], tolerance = 1e-9, info = k)
  }
}

a <- read_shots_fixture("a")
b <- read_shots_fixture("b")

test_that("echo bookkeeping and angles match Python", {
  expect_equal(shot_of_echo(a), expected$shot_of_echo)
  expect_equal(echo_rank(a), expected$echo_rank)
  expect_equal(echo_xyz(a), expected$echo_xyz, tolerance = 1e-9)
  za <- zenith_azimuth(a)
  expect_equal(za$zenith, expected$zenith, tolerance = 1e-9)
  expect_equal(za$azimuth, expected$azimuth, tolerance = 1e-9)
})

test_that("subsets and stacks match Python", {
  expect_shots(subset(a, seq_len(length(a)) %% 3 != 2), expected$subset)
  expect_shots(concatenate(a, b, a), expected$concatenate)
  expect_shots(concatenate(list(a, b, a)), expected$concatenate)
})

test_that("the misses added from the scan pattern match Python", {
  p <- read_shots_fixture("pattern")
  runs <- list(estimated = list(), given = list(pulses_per_line = 80, seed = 4), stride = list(shot_stride = 3, seed = 9))
  for (name in names(runs)) {
    f <- do.call(fill_missing, c(list(p, expected$pattern), runs[[name]]))
    want <- expected$fill[[name]]
    expect_equal(length(f), want$n_shots, info = name)
    added <- unclass(f)$direction[-seq_len(length(p)), , drop = FALSE]
    expect_equal(added, want$added, tolerance = 1e-9, info = name)
    expect_equal(unclass(f)$origin[length(f), ], want$origin, tolerance = 1e-9, info = name)
    expect_equal(sum(unclass(f)$echo_count), sum(unclass(p)$echo_count))
  }
  expect_identical(fill_missing(p, expected$pattern, pulses_per_line = 0), p)
})

test_that("shots files round trip and read as in Python", {
  info <- shots_file_info(fixture("python.parquet"))
  expect_equal(info$n_shots, expected$info$n_shots)
  expect_equal(info$n_echoes, expected$info$n_echoes)
  expect_equal(info$n_groups, expected$info$n_groups)
  expect_equal(info$bounds$min, expected$info$min, tolerance = 1e-9)
  expect_equal(info$bounds$max, expected$info$max, tolerance = 1e-9)
  expect_equal(unname(info$scans), matrix(expected$info$scans, ncol = 3), tolerance = 1e-9)
  expect_shots(load_shots(fixture("python.parquet"), groups = c(1, 3)), expected$groups)
  path <- tempfile(fileext = ".parquet")
  on.exit(unlink(path))
  save_shots(a, path, double = TRUE, row_group_size = 25)
  back <- load_shots(path)
  expect_equal(echo_xyz(back), echo_xyz(a), tolerance = 1e-9)
  expect_equal(unclass(back)$echo_count, unclass(a)$echo_count)
  expect_equal(sort(shots_file_info(path)$echo_attrs), c("amplitude", "deviation"))
})

test_that("a scan position's shots gain their misses before the SOP", {
  p <- read_shots_fixture("pattern")
  sop <- diag(4)
  sop[1:3, 4] <- c(10, -5, 2)
  sop[1:2, 1:2] <- matrix(c(0, 1, -1, 0), 2)
  position <- structure(list(name = "ScanPos001", rxp = "ScanPos001.rxp", sop = sop, pattern = expected$pattern),
                        class = "sylva_scan_position")
  local_mocked_bindings(read_rxp_shots = function(path, ...) p, .package = "sylva")
  got <- read_shots(position, fill_missing = TRUE, shot_stride = 3)
  want <- transform(fill_missing(p, expected$pattern, shot_stride = 3), sop)
  expect_equal(unclass(got), unclass(want))
  expect_equal(length(got), expected$fill$stride$n_shots)
  expect_equal(length(read_shots(position)), length(p))
  position$pattern <- NULL
  expect_error(read_shots(position, fill_missing = TRUE), "no scan pattern")
})
