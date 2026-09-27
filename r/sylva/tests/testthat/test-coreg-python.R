# The coregistration R API against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_coreg.py; Python indices are
# 0-based, R's 1-based).

fixture <- function(...) test_path("fixtures", "coreg", ...)
source(fixture("expected.R"), local = TRUE)

read_matrix <- function(name) unname(as.matrix(read.csv(fixture(name), header = FALSE)))
tol <- 1e-9

test_that("transforms match Python", {
  e <- expected$transforms
  for (k in seq_len(nrow(e$xi))) {
    T <- se3_exp(e$xi[k, ])
    expect_equal(T, e$se3_exp[k, , ], tolerance = tol)
    expect_equal(se3_log(T), e$se3_log[k, ], tolerance = tol)
    expect_equal(so3_log(so3_exp(e$xi[k, 1:3])), e$so3_log[k, ], tolerance = tol)
    expect_equal(invert(T), e$inverse[k, , ], tolerance = tol)
  }
  expect_equal(coreg_kabsch(e$src, e$dst), e$kabsch, tolerance = tol)
  expect_equal(coreg_kabsch(e$src, e$dst, e$w), e$kabsch_w, tolerance = tol)
  expect_equal(kabsch_2d_yaw(e$src, e$dst), e$yaw, tolerance = tol)
  expect_equal(kabsch_2d_yaw(e$src, e$dst, e$w), e$yaw_w, tolerance = tol)
  expect_equal(transform_points(e$T, e$src), e$moved, tolerance = tol)
  expect_equal(transform_vectors(e$T, e$src), e$rotated, tolerance = tol)
  expect_equal(unname(transform_difference(e$T, se3_exp(e$xi[1, ]))), e$difference, tolerance = tol)
  expect_equal(rotation_angle(e$T), e$angle, tolerance = tol)
  expect_equal(yaw_transform(0.7, 1, -2, 0.5), e$yaw_transform, tolerance = tol)
  expect_equal(skew(c(1, 2, 3)), e$skew, tolerance = tol)
  expect_error(coreg_kabsch(e$src[1:2, ], e$dst[1:2, ]), "at least 3")
})

test_that("reflectors match Python", {
  e <- expected$reflectors
  tpl <- read_tiepoint_list(fixture("scan.tpl"))
  expect_equal(unname(as.matrix(tpl[, c("x", "y", "z")])), e$tpl$xyz, tolerance = tol)
  expect_equal(tpl$reflectance, e$tpl$reflectance)
  expect_equal(tpl$n_points, as.integer(e$tpl$n_points))
  expect_equal(tpl$name, e$tpl$name)
  rfl <- read_reflector_list(fixture("ScanPos002.rfl"))
  expect_equal(unname(as.matrix(rfl[, c("x", "y", "z")])), e$rfl$xyz, tolerance = tol)
  expect_equal(rfl$reflectance, e$rfl$reflectance, tolerance = tol)
  expect_equal(rfl$name, e$rfl$name)
  expect_equal(nrow(read_reflector_list(fixture("missing.rfl"))), 0)
  cloud <- read_matrix("reflectance_cloud.csv")
  found <- detect_reflectors(cloud[, 1:3], cloud[, 4])
  expect_equal(unname(as.matrix(found[, c("x", "y", "z")])), e$detected$xyz, tolerance = tol)
  expect_equal(found$reflectance, e$detected$reflectance, tolerance = tol)
  expect_equal(found$diameter, e$detected$diameter, tolerance = tol)
  expect_equal(found$n_points, as.integer(e$detected$n_points))
  m <- match_reflectors(e$source, reflector(e$target[, 1], e$target[, 2], e$target[, 3]))
  expect_true(m$success)
  expect_equal(m$transform, e$match$transform, tolerance = tol)
  expect_equal(m$n_inliers, e$match$n_inliers)
  expect_equal(m$rmse, e$match$rmse, tolerance = tol)
  expect_equal(unname(m$correspondences), e$match$correspondences + 1)
})

test_that("the pose graph matches Python", {
  e <- expected$posegraph
  g <- pose_graph(6, reference = 1, fixed = list("5" = e$fixed_pose))
  for (d in e$edges) {
    g <- add_edge(g, d$i + 1, d$j + 1, d$transform, fitness = d$fitness, rmse = 0.01, n_correspondences = d$n)
  }
  expect_equal(default_information(0.02, 0.6, 1500), e$default_information, tolerance = tol)
  expect_equal(plane_edge_information(e$H, 0.003, 300, e$pose_2), e$plane_information, tolerance = tol)
  expect_equal(adjoint(e$pose_3), e$adjoint, tolerance = tol)
  expect_equal(lapply(components(g), as.double), lapply(e$components, function(c) c + 1))
  g <- initialise(g)
  for (k in 1:6) expect_equal(unclass(g)$poses[[k]], e$initialise[k, , ], tolerance = tol)
  expect_equal(total_error(g), e$total_initial, tolerance = tol)
  expect_equal(residual(g, 3), e$residual_2, tolerance = tol)
  r <- optimise(g)
  expect_equal(r$iterations, e$optimise$iterations)
  expect_equal(r$converged, e$optimise$converged)
  expect_equal(r$initial_error, e$optimise$initial_error, tolerance = tol)
  expect_equal(r$final_error, e$optimise$final_error, tolerance = tol)
  expect_equal(r$rejected_edges, as.integer(e$optimise$rejected + 1))
  expect_equal(r$edge_errors, e$optimise$edge_errors, tolerance = tol)
  for (k in 1:6) expect_equal(r$poses[[k]], e$optimise$poses[k, , ], tolerance = tol)
  expect_equal(relative(r$graph, 2, 4), e$relative, tolerance = tol)
})

scans <- lapply(0:2, function(k) read_matrix(sprintf("scan%d.csv", k)))

test_that("joint refinement matches Python", {
  e <- expected$survey
  start <- lapply(1:3, function(k) e$start[k, , ])
  messages <- character()
  r <- refine_joint(scans, start, rbind(c(1, 2), c(2, 3), c(1, 3)), e$stems, 2, points_per_scan = 2500,
                    correspondences_per_pair = 900, voxel_sizes = c(0.2, 0.1), max_distances = c(0.4, 0.2), seed = 3,
                    log = function(m) messages <<- c(messages, m))
  for (k in 1:3) expect_equal(r$poses[[k]], e$refine$poses[k, , ], tolerance = tol)
  expect_equal(r$shifts, e$refine$shifts, tolerance = tol)
  expect_equal(r$rotations, e$refine$rotations, tolerance = tol)
  expect_equal(r$residual_before, e$refine$residual_before, tolerance = tol)
  expect_equal(r$residual_after, e$refine$residual_after, tolerance = tol)
  expect_equal(r$correspondences, as.integer(e$refine$correspondences))
  expect_equal(messages, e$refine$log)
})

test_that("the terrain model and stem detection match Python", {
  e <- expected$survey
  g <- fit_ground(scans[[1]], cell_size = 0.5)
  expect_equal(unclass(g)$elevation, e$ground$elevation, tolerance = tol)
  expect_equal(unclass(g)$origin, e$ground$origin, tolerance = tol)
  expect_equal(unclass(g)$observed * 1, e$ground$observed)
  expect_equal(slope_deg(g), e$ground$slope, tolerance = tol)
  expect_equal(height_at(g, e$ground_query), e$height_at, tolerance = tol)
  expect_equal(support(g, e$ground_query) * 1, e$support)
  expect_equal(normalise(g, scans[[1]][1:50, ]), e$normalised, tolerance = tol)
  m <- coreg_detect_stems(scans[[1]], g, name = "scan0")
  expect_gt(length(m), 0)
  expect_equal(positions(m), e$detected$xyz, tolerance = tol)
  expect_equal(diameters(m), e$detected$dbh, tolerance = tol)
  expect_equal(qualities(m), e$detected$quality, tolerance = tol)
  expect_equal(axes(m), e$detected$axes, tolerance = tol)
})

test_that("ICP and local geometry match Python", {
  e <- expected$survey
  planar <- lapply(scans[1:2], planar_filter)
  expect_equal(vapply(planar, nrow, 0), e$planar$n)
  for (k in 1:2) expect_equal(planar[[k]][1:20, ], e$planar$head[[k]], tolerance = tol)
  cfg <- icp_config(voxel_sizes = c(0.2, 0.1), max_distances = c(0.5, 0.25), max_points = 3000)
  initial <- invert(e$start[2, , ]) %*% e$start[1, , ]
  r <- coreg_icp(planar[[1]], planar[[2]], initial, cfg)
  expect_equal(r$transform, e$icp$transform, tolerance = tol)
  expect_equal(r$fitness, e$icp$fitness, tolerance = tol)
  expect_equal(r$inlier_rmse, e$icp$inlier_rmse, tolerance = tol)
  expect_equal(r$n_correspondences, as.integer(e$icp$n_correspondences))
  expect_equal(r$iterations, as.integer(e$icp$iterations))
  expect_equal(r$history, e$icp$history, tolerance = tol)
  expect_equal(r$information$hessian, e$icp$hessian, tolerance = tol)
  expect_equal(r$information$sigma, e$icp$sigma, tolerance = tol)
  target <- icp_target(planar[[2]], cfg)
  expect_gt(length(target), 0)
  expect_equal(coreg_icp(planar[[1]], target, initial, cfg)$transform, e$icp$prepared, tolerance = tol)
  info <- plane_information(planar[[1]], planar[[2]], initial, cfg)
  expect_equal(info$hessian, e$plane_information$hessian, tolerance = tol)
  expect_equal(info$n, e$plane_information$n)
  ev <- evaluate_registration(scans[[1]], scans[[2]], initial, voxel = 0.1, max_points = 2000)
  expect_equal(c(ev$fitness, ev$inlier_rmse, ev$n_inliers), e$evaluate, tolerance = tol)
  nrm <- coreg_estimate_normals(scans[[1]][1:400, ], k = 12, radius = 0.5)
  expect_equal(nrm$normals, e$geometry$normals, tolerance = tol)
  expect_equal(nrm$planarity, e$geometry$planarity, tolerance = tol)
  v <- coreg_voxel_downsample(scans[[1]], 0.3, return_counts = TRUE)
  expect_equal(v$points, e$geometry$voxel, tolerance = tol)
  expect_equal(v$counts, as.integer(e$geometry$counts))
  q <- query(kd_tree(scans[[2]][, 1:2]), scans[[1]][1:40, 1:2], distance_upper_bound = 0.1)
  n2 <- nrow(scans[[2]])
  expect_equal(q$distance, e$geometry$distance, tolerance = tol)
  expect_equal(ifelse(is.na(q$index), n2, q$index - 1), e$geometry$index)
})

test_that("stem maps and stem matching match Python", {
  e <- expected$stems
  a <- load_stem_map(fixture("stems_python.json"))
  expect_equal(xy(a), e$xy, tolerance = tol)
  expect_equal(diameters(a), e$dbh, tolerance = tol)
  expect_equal(qualities(a), e$quality, tolerance = tol)
  expect_equal(positions(sorted_by_quality(a))[, 1], e$sorted_x, tolerance = tol)
  expect_equal(positions(top(a, 5))[, 1], e$top5_x, tolerance = tol)
  T <- yaw_transform(0.8, 3, -2, 0.1)
  expect_equal(positions(transformed(a, T)), e$moved, tolerance = tol)
  expect_equal(axes(transformed(a, T)), e$moved_axes, tolerance = tol)
  # Written back, the file is the one Python wrote.
  path <- tempfile(fileext = ".json")
  save_stem_map(a, path)
  expect_identical(readLines(path, warn = FALSE), readLines(fixture("stems_python.json"), warn = FALSE))
  b <- stem_map_from_arrays(e$b_xy, e$b_dbh, name = "b")
  r <- match_stem_maps(a, b)
  expect_true(r$success)
  expect_equal(r$transform, e$match$transform, tolerance = tol)
  expect_equal(r$n_inliers, as.integer(e$match$n_inliers))
  expect_equal(r$inlier_rmse, e$match$inlier_rmse, tolerance = tol)
  expect_equal(r$score, e$match$score, tolerance = tol)
  expect_equal(unname(r$correspondences), e$match$correspondences + 1)
  expect_equal(r$ambiguity, e$match$ambiguity, tolerance = tol)
})
