# The coregistration pipeline in R against the Python package on the same
# scans (fixtures written by tests/parity/export_r_coreg_pipeline.py; Python
# indices are 0-based, R's 1-based).

fixture <- function(...) test_path("fixtures", "coreg_pipeline", ...)
source(fixture("expected.R"), local = TRUE)

read_matrix <- function(name) unname(as.matrix(read.csv(fixture(name), header = FALSE)))
tol <- 1e-9
untimed <- function(x) gsub("\\([0-9]+\\.[0-9] s\\)|in [0-9]+\\.[0-9] s", "", x)

clouds <- lapply(0:2, function(k) read_matrix(sprintf("scan%d.csv", k)))
targets <- lapply(0:2, function(k) {
  t <- read_matrix(sprintf("targets%d.csv", k))
  reflector(t[, 1], t[, 2], t[, 3])
})
config <- function(...) {
  args <- list(verbose = FALSE, workers = 2, min_points_per_scan = 500)
  args[names(list(...))] <- list(...)
  do.call(coreg_config, args)
}
scans <- lapply(1:3, function(k) prepare_scan(clouds[[k]], config(), name = sprintf("s%d", k - 1),
                                                reflectors = targets[[k]]))
stems_only <- config(use_reflectors = FALSE)

expect_scan <- function(s, e) {
  expect_equal(s$name, e$name)
  expect_equal(s$n_points, as.integer(e$n_points))
  expect_equal(s$error, e$error)
  st <- unclass(s$stem_map)$stems
  expect_equal(unname(cbind(st$x, st$y, st$z, st$dbh, qualities(s$stem_map))), e$stems, tolerance = tol)
  expect_equal(nrow(s$icp_points), as.integer(e$n_icp))
  expect_equal(s$icp_points[1:20, ], e$icp_head, tolerance = tol)
  expect_equal(colSums(s$icp_points), e$icp_sum, tolerance = tol)
  expect_equal(s$icp_heights[1:20], e$heights_head, tolerance = tol)
  expect_equal(sum(s$icp_heights), e$heights_sum, tolerance = tol)
  expect_equal(unclass(s$ground)$elevation, e$elevation, tolerance = tol)
  expect_equal(unclass(s$ground)$origin, e$ground_origin, tolerance = tol)
  expect_equal(unclass(s$ground)$observed * 1, e$observed)
  expect_equal(matrix(c(s$reflectors$x, s$reflectors$y, s$reflectors$z), ncol = 3), e$reflectors, tolerance = tol)
  expect_equal(s$levelling, e$levelling, tolerance = tol)
  expect_equal(s$origin, e$origin, tolerance = tol)
}

expect_pair <- function(p, e) {
  expect_equal(c(p$i, p$j), ifelse(e$ij < 0, NA_integer_, as.integer(e$ij + 1)))
  expect_equal(p$success, e$success)
  expect_equal(p$reason, e$reason)
  expect_equal(summary(p), e$summary)
  expect_equal(p$transform, e$transform, tolerance = tol)
  expect_equal(p$coarse_transform, e$coarse, tolerance = tol)
  expect_equal(fitness(p), e$fitness, tolerance = tol)
  expect_equal(rmse(p), e$rmse, tolerance = tol)
  expect_equal(p$fitness_above, e$fitness_above, tolerance = tol)
  expect_equal(p$ground_offset, e$ground_offset, tolerance = tol)
  expect_equal(p$used_icp, e$used_icp)
  expect_equal(p$trusted, e$trusted)
  expect_equal(n_stem_matches(p), as.integer(e$n_stem_matches))
  expect_equal(c(p$coarse_stem_rmse, p$fine_stem_rmse), e$stem_rmse, tolerance = tol)
  expect_equal(unname(p$matched_source), e$matched_source, tolerance = tol)
  expect_equal(if (is.null(p$icp)) 0L else p$icp$n_correspondences, as.integer(e$icp_n))
}

expect_survey <- function(r, e, messages) {
  expect_equal(array(unlist(r$poses), c(4, 4, length(r$poses))), aperm(e$poses, c(2, 3, 1)), tolerance = tol)
  expect_equal(r$registered * 1, e$registered * 1)
  expect_equal(r$reference, as.integer(e$reference) + 1L)
  expect_equal(r$edge_to_pair, as.integer(e$edge_to_pair) + 1L)
  expect_equal(vapply(r$pairs, summary, ""), e$summaries)
  expect_equal(untimed(report(r)), e$report)
  expect_equal(untimed(messages), e$log)
  c <- consistency(r)
  expect_equal(nrow(c), nrow(e$consistency))
  if (nrow(c)) {
    expect_equal(unname(as.matrix(c)), unname(cbind(e$consistency[, 1:2] + 1, e$consistency[, 3])), tolerance = tol)
  }
  if (length(e$optimisation)) {
    o <- r$optimisation
    expect_equal(c(o$iterations, o$initial_error, o$final_error), e$optimisation, tolerance = tol)
  }
  path <- tempfile(fileext = ".json")
  save_survey(r, path)
  text <- paste(readLines(path, warn = FALSE), collapse = "\n")
  expect_equal(sub('"seconds": [^,]*,', '"seconds": ?,', text), e$saved)
  loaded <- load_transforms(path)
  registered <- which(r$registered)
  expect_equal(names(loaded), vapply(r$scans[registered], function(s) s$name, ""))
  for (k in seq_along(registered)) {
    expect_equal(loaded[[k]], transform_for(r, registered[k]), tolerance = tol)
  }
}

test_that("prepared scans match Python", {
  for (k in 1:3) expect_scan(scans[[k]], expected$scans[[k]])
  expect_true(all(vapply(scans, usable, TRUE)))
  level <- read_matrix("levelling.csv")
  expect_scan(prepare_scan(clouds[[2]], config(), name = "lev", levelling = level, origin = c(0, 0, 0)),
              expected$levelled)
  few <- prepare_scan(clouds[[1]][1:300, ], config())
  expect_equal(few$error, expected$few)
  expect_false(usable(few))
  expect_equal(location(scans[[1]], diag(4)), c(0, 0, 0))
})

test_that("pair registration matches Python", {
  e <- expected$pairs
  expect_pair(register_pair(scans[[1]], scans[[2]], stems_only), e$default)
  expect_pair(register_pair(scans[[3]], scans[[1]], stems_only, i = 3, j = 1), e$reverse)
  expect_pair(register_pair(scans[[1]], scans[[2]], config()), e$targets)
  expect_pair(register_pair(scans[[1]], scans[[2]], config(min_icp_fitness = 0.99, trusted_reflector_matches = 3)),
              e$trusted)
  expect_pair(register_pair(scans[[1]], scans[[3]], config(max_match_ambiguity = -1, use_reflectors = FALSE),
                            j = 3), e$rival)
  target <- icp_target(scans[[2]]$icp_points, unclass(stems_only)$icp)
  expect_pair(register_pair(scans[[1]], scans[[2]], stems_only, target_icp = target), e$default)
  expect_true(nzchar(capture.output(print(register_pair(scans[[1]], scans[[2]], stems_only)))))
  expect_pair(register_pair(scans[[1]], scans[[2]], stems_only, initial = expected$truth01), e$initial)
})

test_that("whole surveys match Python", {
  e <- expected$surveys
  run <- function(s, cfg, ...) {
    messages <- character()
    r <- coregister_prepared(s, cfg, progress = function(m) messages <<- c(messages, m), ...)
    list(r = r, messages = messages)
  }
  x <- run(scans, stems_only)
  expect_survey(x$r, e$default, x$messages)
  x <- run(scans, config())
  expect_survey(x$r, e$targets, x$messages)
  p <- expected$priors$true
  x <- run(scans, stems_only, fixed = list("1" = p[[1]], "2" = p[[2]]))
  expect_survey(x$r, e$fixed, x$messages)
  x <- run(scans, config(use_reflectors = FALSE, workers = 1), pairs = list(c(1, 2)))
  expect_survey(x$r, e$pairs, x$messages)
  stemless <- scans
  stemless[[3]]$stem_map <- stem_map(name = "s2")
  x <- run(stemless, stems_only, priors = list(p[[1]], p[[2]], expected$priors$prior))
  expect_survey(x$r, e$priors, x$messages)
  expect_length(rejected_pairs(x$r), 0)
  expect_equal(length(successful_pairs(x$r)), sum(vapply(x$r$pairs, function(q) q$success, TRUE)))
})

test_that("placement from a prior matches Python", {
  e <- expected$placed
  p <- expected$priors$true
  r <- place_from_prior(scans[[3]], scans[1:2], p[1:2], expected$priors$prior, stems_only)
  expect_pair(r$result, e$pair)
  expect_equal(r$used, as.integer(e$used) + 1L)
})

test_that("coregister and merge_clouds match Python", {
  messages <- character()
  r <- coregister(clouds, config(use_reflectors = FALSE), names = c("a", "b", ""),
                  progress = function(m) messages <<- c(messages, m))
  expect_survey(r, expected$coregister, messages)
  m <- merge_clouds(clouds, r, voxel = 0.1)
  e <- expected$merged
  expect_equal(length(m), as.integer(e$n))
  expect_equal(colSums(unclass(m)$xyz), e$sum, tolerance = tol)
  expect_equal(unclass(m)$xyz[1:20, ], e$head, tolerance = tol)
  expect_equal(as.double(tabulate(m$scan_id + 1L)), e$ids)
})

test_that("reading options merge a settings file and explicit bounds", {
  path <- tempfile(fileext = ".txt")
  writeLines(c("# export filter", "riegl.deviation, 0, 20", "range; 0.2; 25"), path)
  o <- reading_options(path, max_range = 18, min_amplitude = 3, min_range = NULL)
  expect_equal(o[order(names(o))], list(max_deviation = 20, max_range = 18, min_amplitude = 3, min_deviation = 0,
                                        min_range = 0.2))
  expect_equal(reading_options(max_deviation = 4), list(max_deviation = 4))
})
