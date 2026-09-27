# The coregistration pipeline in R on its own: files, progress, workers and
# the result objects.

scan_files <- function() {
  lapply(0:1, function(k) unname(as.matrix(read.csv(test_path("fixtures", "coreg_pipeline", sprintf("scan%d.csv", k)),
                                                     header = FALSE))))
}

test_that("results do not depend on the number of workers", {
  clouds <- scan_files()
  scans <- lapply(clouds, prepare_scan, config = coreg_config(verbose = FALSE, min_points_per_scan = 500))
  one <- coregister_prepared(scans, coreg_config(verbose = FALSE, workers = 1, use_reflectors = FALSE))
  four <- coregister_prepared(scans, coreg_config(verbose = FALSE, workers = 4, use_reflectors = FALSE))
  expect_identical(one$poses, four$poses)
  expect_identical(vapply(one$pairs, summary, ""), vapply(four$pairs, summary, ""))
  expect_s3_class(one, "sylva_survey_result")
  expect_output(print(one), "2 scans")
  expect_output(print(scans[[1]]), "sylva_scan_features")
})

test_that("progress is printed when verbose", {
  clouds <- scan_files()
  expect_output(coregister(clouds, coreg_config(min_points_per_scan = 500, use_reflectors = FALSE)),
                "Preparing 2 scans")
  expect_silent(coregister(clouds, coreg_config(verbose = FALSE, min_points_per_scan = 500)))
  seen <- character()
  coregister(clouds, coreg_config(min_points_per_scan = 500), progress = function(m) seen <<- c(seen, m))
  expect_true(any(grepl("^Refining", seen)))
  expect_error(coregister(clouds, coreg_config(min_points_per_scan = 500), progress = function(m) stop("halt")),
               "halt")
})

test_that("scans are read from files, and unreadable ones are set aside", {
  dir <- tempfile()
  dir.create(dir)
  clouds <- scan_files()
  paths <- file.path(dir, c("a.laz", "b.laz"))
  for (k in 1:2) write_cloud(point_cloud(clouds[[k]]), paths[k])
  broken <- file.path(dir, "broken.laz")
  writeBin(charToRaw("LASF garbage"), broken)
  cfg <- coreg_config(verbose = FALSE, min_points_per_scan = 500, workers = 2)
  s <- prepare_scan(paths[1], cfg)
  expect_equal(s$name, "a")
  expect_equal(s$source, paths[1])
  expect_true(usable(s))
  r <- coregister(c(paths, broken), cfg)
  expect_match(r$scans[[3]]$error, "^OSError: ")
  expect_false(r$registered[3])
  expect_match(report(r), "SET ASIDE")
  merged <- merge_clouds(paths, r, voxel = 0.2, only_registered = FALSE)
  expect_s3_class(merged, "sylva_cloud")
  expect_setequal(unique(merged$scan_id), c(0L, 1L))
  expect_error(prepare_scan(paths[1], coreg_config(riscan_filter = "current")), "amplitude")
})
