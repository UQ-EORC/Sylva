# The R QSM API against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_qsm.py).

fixture <- function(...) test_path("fixtures", "qsm", ...)
source(fixture("expected.R"), local = TRUE)

read_matrix <- function(name, ncol = 12) {
  path <- fixture(paste0(name, ".csv"))
  if (file.size(path) == 0) return(matrix(0, 0, ncol))
  unname(as.matrix(read.csv(path, header = FALSE)))
}
read_vector <- function(name) scan(fixture(paste0(name, ".csv")), quiet = TRUE)
md5 <- function(path) unname(tools::md5sum(path))
cylinders <- function(q) {
  m <- unname(as.matrix(q$cylinders))
  storage.mode(m) <- "double"
  m
}

expect_values <- function(got, want, info) {
  for (k in names(want)) expect_equal(as.double(got[[k]]), as.double(want[[k]]), tolerance = 1e-9, info = paste(info, k))
}

rows <- read_matrix("rows")
model <- qsm(rows)

test_that("a model's values, cuts and summaries match Python", {
  expect_s3_class(model$cylinders, "data.frame")
  expect_equal(names(model$cylinders), QSM_COLUMNS)
  expect_equal(length(model), nrow(rows))
  e <- expected$model
  expect_equal(model$end, e$end, tolerance = 1e-9)
  for (k in c("volumes", "total_volume", "stem_volume", "branch_volume", "total_length", "max_branch_order", "dbh")) {
    expect_equal(as.double(model[[k]]), e[[k]], tolerance = 1e-9, info = k)
  }
  expect_equal(model$start, rows[, 1:3])
  expect_equal(column(model, "radius"), rows[, 8])
  expect_error(column(model, "nope"), "nope")
  expect_equal(total_volume(model), e$total_volume, tolerance = 1e-9)
  expect_equal(sapply(expected$zs, function(z) volume_above(model, z)), expected$volume_above, tolerance = 1e-9)
  for (k in seq_along(expected$zs)) {
    cut <- above(model, expected$zs[k])
    expect_s3_class(cut, "sylva_qsm")
    expect_equal(cylinders(cut), read_matrix(paste0("above_", k - 1)), tolerance = 1e-9, info = paste("above", k))
  }
  m <- metrics(model)
  for (k in setdiff(names(expected$metrics), grep("^crown_", names(expected$metrics), value = TRUE))) {
    expect_equal(as.double(m[[k]]), as.double(expected$metrics[[k]]), tolerance = 1e-9, info = k)
  }
  for (k in sub("^crown_", "", grep("^crown_", names(expected$metrics), value = TRUE))) {
    expect_equal(m$crown[[k]], expected$metrics[[paste0("crown_", k)]], tolerance = 1e-9, info = k)
  }
  b <- branches(model)
  expect_s3_class(b, "data.frame")
  expect_values(b, expected$branches, "branches")
  expect_values(summary(model), expected$summary, "summary")
  for (name in c("mesh", "mesh_cont")) {
    ms <- if (name == "mesh") mesh(model) else mesh(model, sides = 8, contiguous = TRUE)
    expect_equal(ms$vertices, read_matrix(paste0(name, "_vertices")), tolerance = 1e-9, info = name)
    expect_equal(ms$faces, matrix(as.integer(read_matrix(paste0(name, "_faces"), 3)), ncol = 3), info = name)
    expect_equal(ms$owner, expected[[paste0(name, "_owner")]], info = name)
  }
  empty <- qsm(matrix(0, 0, 12))
  expect_equal(length(empty), 0)
  expect_equal(empty$total_volume, 0)
  expect_equal(empty$max_branch_order, 0L)
})

test_that("files match Python byte for byte", {
  f <- expected$files
  out <- withr::local_tempdir()
  to_csv(model, file.path(out, "m.csv"))
  expect_equal(md5(file.path(out, "m.csv")), f$csv)
  back <- qsm_from_csv(file.path(out, "m.csv"))
  expect_equal(cylinders(back), expected$from_csv, tolerance = 1e-9)
  to_treefile(model, file.path(out, "m.txt"))
  expect_equal(md5(file.path(out, "m.txt")), f$treefile)
  to_obj(model, file.path(out, "m.obj"), sides = 8)
  expect_equal(md5(file.path(out, "m.obj")), f$obj)
  to_ply(model, file.path(out, "m.ply"), contiguous = TRUE)
  expect_equal(md5(file.path(out, "m.ply")), f$ply)
  to_ply(model, file.path(out, "c.ply"), color = c(10, 200, 30))
  expect_equal(md5(file.path(out, "c.ply")), f$ply_color)
  ms <- mesh(model, sides = 6)
  write_obj(file.path(out, "w.obj"), list(ms, list(vertices = ms$vertices[1:3, ], faces = matrix(0:2, 1))),
            names = c("a", "b"))
  expect_equal(md5(file.path(out, "w.obj")), f$write_obj)
  write_ply_mesh(file.path(out, "w.ply"), ms$vertices, ms$faces, c(5, 6, 7))
  expect_equal(md5(file.path(out, "w.ply")), f$write_ply)
})

test_that("cylinder fits, skeletons, models and wood match Python", {
  section <- read_matrix("section", 3)
  fit <- fit_cylinder(section[1:300, ])
  expect_equal(c(fit$point, fit$axis, fit$radius, fit$rmse), expected$fit, tolerance = 1e-9)
  fit <- fit_cylinder(section[1:300, ], axis_init = c(0.1, 0, 1))
  expect_equal(c(fit$point, fit$axis, fit$radius, fit$rmse), expected$fit_axis, tolerance = 1e-9)
  r <- fit_cylinder_ransac(section, threshold = 0.01, iterations = 50, seed = 4)
  expect_equal(c(r$point, r$axis, r$radius, r$rmse), expected$ransac, tolerance = 1e-9)
  expect_equal(as.double(r$inliers), expected$ransac_inliers)
  tree <- point_cloud(read_matrix("tree", 3))
  s <- skeletonize(tree, bin_length = 0.2)
  for (k in names(expected$skeleton)) {
    expect_equal(unname(s[[k]]), expected$skeleton[[k]], tolerance = 1e-9, info = k)
  }
  q <- build_qsm(tree)
  expect_equal(cylinders(q), read_matrix("build_qsm"), tolerance = 1e-9)
  q <- build_qsm(tree, base_xy = c(0.01, 0), bin_length = 0.15, radius_power = 0.25, buttress_equivalent_area = FALSE)
  expect_equal(cylinders(q), read_matrix("build_qsm_set"), tolerance = 1e-9)
  expect_equal(wood_points(tree)$xyz, read_matrix("wood", 3), tolerance = 1e-9)
  expect_equal(wood_points(tree, voxel_size = 0.03, threshold = 0.8, passage = FALSE)$xyz, read_matrix("wood_set", 3),
               tolerance = 1e-9)
})

test_that("buttress meshes and their join to a model match Python", {
  base <- read_matrix("base", 3)
  cloud <- point_cloud(base, height = base[, 3])
  b <- buttress_mesh(cloud, c(0, 0), ground_z = 0, top = 1.2, resolution = 0.04)
  auto <- buttress_mesh(cloud, c(0.01, 0.02), resolution = 0.05, slice_height = 0.1)
  for (name in c("buttress", "buttress_auto")) {
    x <- unclass(if (name == "buttress") b else auto)
    expect_equal(x$vertices, read_matrix(paste0(name, "_vertices"), 3), tolerance = 1e-9, info = name)
    expect_equal(x$faces, read_matrix(paste0(name, "_faces"), 3), info = name)
    expect_values(x, expected[[name]], name)
  }
  stem <- qsm(read_matrix("stem"))
  expect_equal(total_volume(b, stem), expected$buttress_total_volume, tolerance = 1e-9)
  nb <- nrow(unclass(b)$vertices)
  nf <- nrow(unclass(b)$faces)
  for (name in c("fused", "fused_flat")) {
    t <- if (name == "fused") fuse(b, stem) else fuse(b, stem, sides = 8, contiguous = FALSE, overlap = 0)
    t <- unclass(t)
    expect_equal(t$vertices[seq_len(nb), ], unclass(b)$vertices)
    expect_equal(t$vertices[-seq_len(nb), ], read_matrix(paste0(name, "_vertices"), 3), tolerance = 1e-9, info = name)
    expect_equal(t$faces[-seq_len(nf), ], read_matrix(paste0(name, "_faces"), 3), info = name)
    expect_values(t, expected[[name]], name)
  }
  f <- expected$files
  out <- withr::local_tempdir()
  to_obj(b, file.path(out, "b.obj"))
  expect_equal(md5(file.path(out, "b.obj")), f$buttress_obj)
  to_ply(b, file.path(out, "b.ply"))
  expect_equal(md5(file.path(out, "b.ply")), f$buttress_ply)
  fused <- fuse(b, stem)
  to_obj(fused, file.path(out, "f.obj"))
  expect_equal(md5(file.path(out, "f.obj")), f$fused_obj)
  to_ply(fused, file.path(out, "f.ply"))
  expect_equal(md5(file.path(out, "f.ply")), f$fused_ply)
  to_ply(fused, file.path(out, "g.ply"), color = c(1, 2, 3))
  expect_equal(md5(file.path(out, "g.ply")), f$fused_ply_color)
})

test_that("a plot of QSMs matches Python", {
  xyz <- read_matrix("plot_xyz", 3)
  labels <- as.integer(read_vector("plot_labels"))
  cloud <- point_cloud(xyz, height = xyz[, 3])
  stems <- tree(4, 9.0, -3.0, dbh = 0.5)
  plots <- suppressWarnings(list(
    plain = build_plot(cloud, labels, wood = FALSE, min_points = 1000),
    full = build_plot(cloud, labels, stems, wood = TRUE, buttress = TRUE, min_points = 1000, bin_length = 0.15)
  ))
  f <- expected$files
  for (name in names(plots)) {
    p <- plots[[name]]
    e <- expected[[paste0("plot_", name)]]
    expect_s3_class(p, "sylva_plot_qsms")
    ids <- names(unclass(p)$models)
    expect_equal(as.double(ids), e$ids, info = name)
    expect_equal(length(p), length(e$ids))
    expect_equal(as.double(names(unclass(p)$skipped)), e$skipped_ids, info = name)
    expect_equal(unclass(p)$skipped, e$skipped, info = name)
    expect_equal(as.double(names(unclass(p)$buttresses)), as.double(e$buttress_ids), info = name)
    for (t in ids) {
      expect_equal(cylinders(unclass(p)$models[[t]]), read_matrix(paste0(name, "_model_", t)),
                   tolerance = 1e-9, info = paste(name, t))
    }
    expect_equal(total_volume(p), e$total_volume, tolerance = 1e-9, info = name)
    expect_equal(sapply(ids, function(t) volume(p, t), USE.NAMES = FALSE), e$volumes, tolerance = 1e-9, info = name)
    tab <- as.data.frame(p)
    for (k in names(tab)) {
      want <- e[[paste0("table_", k)]]
      expect_equal(as.double(ifelse(is.na(tab[[k]]), NaN, tab[[k]])), want, tolerance = 1e-9, info = paste(name, k))
    }
    out <- withr::local_tempdir()
    to_csv(p, file.path(out, "trees.csv"))
    expect_equal(md5(file.path(out, "trees.csv")), f[[paste0(name, "_csv")]], info = name)
    for (fmt in c("ply", "obj")) {
      written <- write_meshes(p, file.path(out, fmt), fmt = fmt, sides = 8)
      expect_equal(basename(written), paste0("tree", ids, ".", fmt))
      for (w in written) expect_equal(md5(w), f[[paste0(name, "_", fmt, "_", basename(w))]], info = w)
    }
    write_cylinders(p, file.path(out, "cyl"), prefix = "c")
    for (w in list.files(file.path(out, "cyl"), full.names = TRUE)) {
      expect_equal(md5(w), f[[paste0(name, "_cyl_", basename(w))]], info = w)
    }
  }
  expect_error(write_meshes(plots$plain, tempfile(), fmt = "stl"), "fmt")
  expect_error(build_plot(cloud, labels[-1]), "one value per point")
  expect_error(build_plot(cloud, labels, nope = 1), "nope")
})

test_that("a thinned plot warns that its models were hardly fitted", {
  xyz <- read_matrix("plot_xyz", 3)
  labels <- as.integer(read_vector("plot_labels"))
  keep <- labels %in% c(1, 2)
  cloud <- point_cloud(xyz[keep, ])
  expect_warning(build_plot(cloud, labels[keep], voxel_size = 0.06, wood = FALSE, min_points = 100,
                            spacing_scale = 0), "fitted to points")
})
