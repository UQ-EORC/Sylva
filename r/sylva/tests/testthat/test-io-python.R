# Reading and writing against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_io.py).

expected <- load_expected("io")
tol <- 1e-9
fixture <- function(...) test_path("fixtures", "io", ...)

expect_cloud <- function(c, prefix, skip = character()) {
  names_ <- expected[[paste0(prefix, "_names")]]
  expect_equal(sort(names(c$attrs)), names_, info = prefix)
  expect_equal(c$xyz, expected[[paste0(prefix, "_xyz")]], tolerance = tol, info = prefix)
  for (k in setdiff(names_, skip)) {
    expect_equal(as.numeric(c[[k]]), as.numeric(expected[[paste0(prefix, "_", k)]]), tolerance = tol, info = paste(prefix, k))
  }
}

test_that("files written from R read back as Python's do", {
  c <- fixture_cloud("io", "cloud")
  d <- tempfile()
  dir.create(d)
  w <- function(name, ...) {
    p <- file.path(d, name)
    write_cloud(c, p, ...)
    read_cloud(p)
  }
  expect_cloud(w("las6.las"), "las6")
  expect_cloud(w("las7.laz", point_format = 7, scale = 0.01), "las7")
  expect_cloud(w("ply_bin.ply"), "ply_bin")
  expect_cloud(w("ply_ascii.ply", binary = FALSE), "ply_ascii")
  # Python holds `height` as float32, whose text form has fewer digits.
  expect_cloud(w("txt.txt"), "txt", skip = "height")
  expect_cloud(w("csv.csv"), "csv", skip = "height")
})

test_that("text clouds and matrix files read as in Python", {
  expect_cloud(read_ascii(fixture("plain.xyz")), "plain")
  expect_cloud(read_ascii(fixture("plain.xyz"), columns = c("r", "s")), "named")
  expect_cloud(read_ascii(fixture("plain.xyz"), columns = c("x", "y", "z", "p", "q")), "all")
  expect_cloud(read_ascii(fixture("semi.txt")), "header")
  expect_cloud(read_cloud(fixture("pts.pts")), "pts")
  expect_equal(read_matrix_file(fixture("sop.dat")), expected$matrix, tolerance = tol)
})

test_that("RiVLib is found, or its absence explained", {
  found <- tryCatch(find_rivlib(), error = conditionMessage)
  expect_true(file.exists(found) || grepl("RIVLIB_PATH", found))
})
