# The R foliage API against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_leaves.py).

fixture <- function(...) test_path("fixtures", "leaves", ...)
source(fixture("expected.R"), local = TRUE)

read_matrix <- function(name) unname(as.matrix(read.csv(fixture(paste0(name, ".csv")), header = FALSE)))
read_vector <- function(name) scan(fixture(paste0(name, ".csv")), quiet = TRUE)

expect_lad <- function(got, want, info) {
  for (k in names(want)) expect_equal(got[[k]], want[[k]], tolerance = 1e-9, info = paste(info, k))
}

expect_grid <- function(g, want, info) {
  expect_equal(unclass(g)$origin, want$origin, tolerance = 1e-9, info = info)
  expect_equal(unclass(g)$density, want$density, tolerance = 1e-9, info = info)
  expect_equal(total_area(g), want$total_area, tolerance = 1e-9, info = info)
  p <- profile(g)
  expect_equal(p$z, want$profile_z, tolerance = 1e-9, info = info)
  expect_equal(p$area, want$profile_area, tolerance = 1e-9, info = info)
  c <- cells(g)
  expect_equal(c$centres, matrix(want$cells, ncol = 3), tolerance = 1e-9, info = info)
  expect_equal(c$area, want$cell_area, tolerance = 1e-9, info = info)
}

expect_shape <- function(s, want, info) {
  s <- unclass(s)
  expect_equal(s$vertices, want$vertices, tolerance = 1e-9, info = info)
  expect_equal(s$faces, matrix(as.integer(want$faces), ncol = 3), info = info)
  expect_equal(c(s$length, s$width), c(want$length, want$width), tolerance = 1e-9, info = info)
  expect_equal(area(structure(s, class = "sylva_leaf_shape")), want$area, tolerance = 1e-9, info = info)
}

expect_mesh <- function(m, name, want) {
  mm <- unclass(m)
  expect_equal(mm$vertices, read_matrix(paste0(name, "_vertices")), tolerance = 1e-9, info = name)
  expect_equal(mm$faces, matrix(as.integer(read_matrix(paste0(name, "_faces"))), ncol = 3), info = name)
  expect_equal(mm$centres, read_matrix(paste0(name, "_centres")), tolerance = 1e-9, info = name)
  expect_equal(mm$normals, read_matrix(paste0(name, "_normals")), tolerance = 1e-9, info = name)
  expect_equal(mm$inclination, want$inclination, tolerance = 1e-9, info = name)
  expect_equal(mm$cylinder, want$cylinder, info = name)
  expect_equal(mm$leaf_area, want$leaf_area, tolerance = 1e-9, info = name)
  expect_equal(length(m), want$n, info = name)
  expect_equal(total_area(m), want$total_area, tolerance = 1e-9, info = name)
}

test_that("leaf and wood labels match Python", {
  tree <- read_matrix("tree_xyz")
  expect_equal(classify_leaf_wood(point_cloud(tree)), read_vector("gbs") == 1)
  expect_equal(classify_leaf_wood(tree, intervals = c(0.2, 0.4, 0.8), max_angle = 0.6), read_vector("gbs_intervals") == 1)
  expect_equal(classify_leaf_wood(tree, method = "passage", threshold = 0.9, voxel_size = 0.03),
               read_vector("passage") == 1)
  expect_error(classify_leaf_wood(tree, method = "nope"), "passage")
  expect_error(classify_leaf_wood(tree, nope = 1), "nope")
})

test_that("angle distributions match Python", {
  pts <- read_matrix("leaf_xyz")
  expect_lad(leaf_angle_distribution(pts), expected$lad_res0, "res0")
  expect_lad(leaf_angle_distribution(pts, k = 8, res = NULL, n_bins = 9), expected$lad_none, "none")
  incl <- read_vector("incl")
  expect_lad(leaf_angle_distribution(incl, inclinations = TRUE, weights = incl^2, n_bins = 12), expected$lad_incl, "incl")
  for (name in c("spherical", "planophile", "extremophile")) {
    expect_lad(leaf_angle_distribution_from_type(name, n_bins = 15), expected[[paste0("type_", name)]], name)
  }
  expect_equal(g(leaf_angle_distribution_from_type("planophile"), c(0, 0.3, 0.9, 1.4)), expected$g_planophile,
               tolerance = 1e-9)
  expect_error(leaf_angle_distribution_from_type("round"), "round")
})

test_that("leaf area grids match Python", {
  dense <- read_matrix("dense_xyz")
  g <- leaf_area_density(dense)
  expect_grid(g, expected$grid, "grid")
  expect_grid(leaf_area_density(point_cloud(dense), voxel_size = 0.1, res = 0.015, k = 9), expected$grid_fine, "fine")
  expect_grid(scaled_to(g, 7.5), expected$grid_scaled, "scaled")
  ng <- leaf_area_grid(c(-1, 2, 0.5), 0.4, expected$nan_density)
  expect_grid(ng, expected$nan, "nan")
  expect_equal(area(ng), expected$nan$area, tolerance = 1e-9)
  expect_grid(scaled_to(ng, 4), expected$nan_scaled, "nan scaled")
})

test_that("leaf shapes match Python", {
  v <- expected$mesh_v
  f <- matrix(c(0, 1, 2, 0, 2, 3), ncol = 3, byrow = TRUE)
  expect_shape(leaf_shape(), expected$shape_default, "default")
  expect_shape(default_leaf(), expected$shape_default, "default leaf")
  expect_shape(leaf_shape_from_mesh(v, f), expected$shape_mesh, "mesh")
  expect_shape(leaf_shape_from_mesh(v, f, length = 0.3, normalise = FALSE), expected$shape_mesh_raw, "raw")
  expect_shape(leaf_shape_from_mesh(v[, 1:2], f, width = 0.05), expected$shape_mesh_2d, "2d")
  expect_shape(scaled_to(leaf_shape(length = 0.1, width = 0.03), 0.004), expected$shape_scaled, "scaled")
  expect_shape(resized(leaf_shape(), 0.15), expected$shape_resized, "resized")
  expect_shape(leaf_shape_from_obj(fixture("leaf.obj"), width = 0.05), expected$shape_obj, "obj")
  expect_equal(single_leaf_area(0.06, 0.03), expected$single_default, tolerance = 1e-9)
  expect_equal(single_leaf_area(0.06, 0.03, leaf_shape_from_mesh(v, f)), expected$single_custom, tolerance = 1e-9)
  expect_error(leaf_shape(length = 0), "positive")
  expect_error(leaf_shape(faces = matrix(c(0, 1, 9), 1)), "index")
  expect_error(scaled_to(leaf_shape(), 0), "positive")
  was <- set_default_leaf(length = 0.12)
  expect_equal(unclass(default_leaf())$length, 0.12)
  set_default_leaf(was)
  expect_equal(unclass(default_leaf())$length, 0.08)
})

test_that("leaves on a model and the OBJ files match Python", {
  model <- expected$cylinders
  seeds <- read_matrix("seeds")
  angles <- leaf_angle_distribution(seeds)
  sg <- leaf_area_density(seeds, voxel_size = 0.25)
  expect_mesh(add_leaves(model, sg, angles, leaf_points = seeds, leaf_length = 0.06, leaf_width = 0.03,
                         max_branch_distance = 1.5), "mesh_grid", expected$mesh_grid)
  expect_mesh(add_leaves(list(cylinders = model), 1.5, "planophile", leaf_points = seeds, seed = 4), "mesh_total",
              expected$mesh_total)
  v <- expected$mesh_v
  custom <- leaf_shape_from_mesh(v, matrix(c(0, 1, 2, 0, 2, 3), ncol = 3, byrow = TRUE))
  expect_mesh(add_leaves(NULL, scaled_to(sg, 3), angles, leaf_points = seeds, shape = custom, leaf_width = 0.05,
                         seed = 11), "mesh_shape", expected$mesh_shape)
  few <- add_leaves(model, 0.05, "spherical", leaf_points = seeds, seed = 2)
  expect_mesh(few, "mesh_few", expected$mesh_few)
  expect_error(add_leaves(model, 2), "leaf_points")
  out <- tempfile(fileext = ".obj")
  to_obj(few, out)
  expect_identical(readLines(out), readLines(fixture("few_leaves.obj")))
  write_tree_obj(out, model, few, sides = 8, contiguous = TRUE)
  expect_identical(readLines(out), readLines(fixture("few_tree.obj")))
  unlink(out)
})
