# RiSCAN projects and filters in R against the Python package on the same
# inputs (fixtures written by tests/parity/export_r_riscan.py).

fixture <- function(...) test_path("fixtures", "riscan", ...)
source(fixture("expected.R"), local = TRUE)
source(fixture("projects.R"), local = TRUE)

# The synthetic projects, written as the Python exporter found them.
projects <- tempfile("sylva-riscan-")
for (f in names(files)) {
  path <- file.path(projects, f)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeBin(charToRaw(files[[f]]), path)
}
for (d in dirs) dir.create(file.path(projects, d), recursive = TRUE, showWarnings = FALSE)

relative <- function(path, root) if (is.null(path)) "" else substring(path, nchar(root) + 2)
stack <- function(ms, dims) {
  if (!length(ms)) return(numeric(0))
  a <- array(NA_real_, c(length(ms), dims))
  for (k in seq_along(ms)) if (!is.null(ms[[k]])) a[k, , ] <- ms[[k]]
  a
}

for (key in c("rsp", "legacy", "proj", "empty")) {
  test_that(paste("the", key, "project reads as in Python"), {
    e <- expected[[key]]
    root <- file.path(projects, e$root)
    p <- read_riscan_project(root)
    ps <- unname(p$positions)
    expect_equal(p$name, e$name)
    expect_equal(length(p), e$len)
    expect_equal(!is.null(p$pop), e$has_pop)
    if (e$has_pop) expect_equal(p$pop, e$pop, tolerance = 1e-9)
    if (!e$len) return()
    expect_equal(position_names(p), e$names)
    expect_equal(vapply(ps, function(q) relative(q$rxp, root), ""), e$rxp)
    expect_equal(vapply(ps, function(q) paste(vapply(q$scans, relative, "", root), collapse = "|"), ""), e$scans)
    expect_equal(as.numeric(vapply(ps, function(q) !is.null(sop(q)), TRUE)), e$has_sop)
    expect_equal(stack(lapply(ps, sop), c(4, 4)), e$sop, tolerance = 1e-9)
    expect_equal(vapply(ps, function(q) if (is.null(q$instrument)) "<None>" else q$instrument, ""), e$instrument)
    keys <- c("theta_start", "theta_delta", "theta_count", "phi_start", "phi_delta", "phi_count")
    pat <- t(vapply(ps, function(q) if (is.null(pattern(q))) rep(NA_real_, 6) else unlist(pattern(q)[keys]), numeric(6)))
    expect_equal(unname(pat), e$pattern, tolerance = 1e-9)
    expect_equal(vapply(ps, function(q) relative(q$tiepoints, root), ""), e$tiepoints)
    gnss <- t(vapply(ps, function(q) if (is.null(q$gnss)) rep(NA_real_, 3) else q$gnss, numeric(3)))
    expect_equal(gnss, e$gnss, tolerance = 1e-9)
    expect_equal(stack(lapply(ps, function(q) q$attitude), c(3, 3)), e$attitude, tolerance = 1e-9)
    expect_equal(stack(lapply(ps, levelling), c(4, 4)), e$levelling, tolerance = 1e-9)
    expect_equal(stack(lapply(ps, transform), c(4, 4)), e$transform, tolerance = 1e-9)
    expect_equal(stack(lapply(ps, transform, p$pop), c(4, 4)), e$transform_pop, tolerance = 1e-9)
    expect_equal(vapply(with_scans(p), function(q) q$name, ""), as.character(e$with_scans))
    expect_equal(vapply(with_scans(p, require_sop = FALSE), function(q) q$name, ""), as.character(e$with_scans_any))
    expect_equal(unname(origins(p)), e$origins, tolerance = 1e-9)
    expect_equal(gnss_positions(p), e$gnss_positions, tolerance = 1e-9)
    expect_identical(p$positions[[e$names[2]]], ps[[2]])
    expect_equal(summary(p)$name, e$names)
  })
}

test_that("reflectors come from the position's target list", {
  p <- read_riscan_project(file.path(projects, expected$proj$root))
  expect_equal(nrow(reflectors(p$positions[[1]])), 0)
  tpl <- tempfile(fileext = ".tpl")
  writeLines('[{"positionCartesian": {"x": 1, "y": 2, "z": 3.5}, "reflectance": -2, "name": "T1", "pointcount": 12}, {"x": 1}]', tpl)
  pos <- p$positions[[2]]
  pos$tiepoints <- tpl
  r <- reflectors(pos)
  expect_equal(nrow(r), 1)
  expect_equal(c(r$x, r$y, r$z, r$n_points), c(1, 2, 3.5, 12))
  expect_equal(r$name, "T1")
  expect_true(is.na(r$diameter))
})

test_that("a missing project is an error", {
  expect_error(read_riscan_project(file.path(projects, "nothing")), "not a project directory")
  expect_error(read_points(read_riscan_project(file.path(projects, expected$legacy$root))$positions[["ScanPos004"]]), "no .rxp")
})

test_that("angular steps and RiSCAN's import filter match Python", {
  s <- as.matrix(read.csv(fixture("stream.csv"), header = FALSE))
  xyz <- s[, 1:3]
  amplitude <- s[, 4]
  expect_equal(angular_steps(xyz), expected$steps, tolerance = 1e-9)
  expect_equal(as.numeric(riscan_like_mask(xyz, amplitude, "current")), expected$mask_current)
  expect_equal(as.numeric(riscan_like_mask(xyz, amplitude, "legacy")), expected$mask_legacy)
  expect_equal(as.numeric(riscan_like_mask(xyz, amplitude, "legacy", steps = c(0.05, 0.05), min_neighbours = 3,
                                           weak_db = 20)), expected$mask_legacy_steps)
  expect_true(all(riscan_like_mask(xyz, amplitude, "none")))
  expect_error(riscan_like_mask(xyz, amplitude, "strict"), "mode must be one of")
})

test_that("export settings read and filter as in Python", {
  settings <- read_export_settings(fixture("settings.txt"))
  expect_equal(settings, lapply(expected$settings, as.numeric))
  s <- as.matrix(read.csv(fixture("stream.csv"), header = FALSE))
  a <- as.matrix(read.csv(fixture("attributes.csv"), header = FALSE))
  keep <- export_settings_mask(settings, s[, 1:3], list(deviation = a[, 1], reflectance = a[, 2]))
  expect_equal(as.numeric(keep), expected$settings_mask)
  expect_error(export_settings_mask(settings, s[, 1:3], list(deviation = a[, 1])), "reflectance")
})

test_that("GNSS fixes and rotations match Python", {
  expect_equal(gnss_to_local(expected$gnss_in), expected$gnss_local, tolerance = 1e-9)
  fixes <- lapply(seq_len(nrow(expected$gnss_in)), function(k) if (anyNA(expected$gnss_in[k, ])) NULL else expected$gnss_in[k, ])
  expect_equal(gnss_to_local(fixes), expected$gnss_local, tolerance = 1e-9)
  for (k in seq_len(nrow(expected$angles))) {
    a <- expected$angles[k, ]
    expect_equal(sylva:::core_riscan_rotation_zyx(a[1], a[2], a[3]), expected$rotations[k, , ], tolerance = 1e-9)
  }
})
