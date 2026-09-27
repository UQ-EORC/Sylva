# The R voxel API against the Python package on the same inputs
# (fixtures written by tests/parity/export_r_leaves.py).

fixture <- function(...) test_path("fixtures", "voxels", ...)
source(fixture("expected.R"), local = TRUE)

read_matrix <- function(name) unname(as.matrix(read.csv(fixture(paste0(name, ".csv")), header = FALSE)))
read_vector <- function(name) scan(fixture(paste0(name, ".csv")), quiet = TRUE)

rays <- read_matrix("canopy_rays")
attrs <- read_matrix("canopy_attrs")
canopy <- shots(origin = rays[, 1:3], direction = rays[, 4:6], echo_start = rays[, 7], echo_count = rays[, 8],
                echo_range = read_vector("canopy_range"),
                echo_attrs = list(classification = as.integer(attrs[, 1]), tree_id = as.integer(attrs[, 2]),
                                  intensity = attrs[, 3]))
dtm <- raster(expected$dtm_data, -0.5, -0.5, 1)
bounds <- list(c(0, 0, 0), c(3, 3, 7))

grids <- list(
  classes = ray_voxelize(canopy, 0.5, bounds, ground_class = 2, leaf_classes = 4, wood_classes = 6, occlusion = TRUE,
                         attenuation = c("fpl", "transmittance"), laser = "VZ-400"),
  dtm = ray_voxelize(canopy, 0.5, bounds, dtm = dtm, ground_distance = 0.15, wood_classes = 6, occlusion = TRUE,
                     attenuation = "transmittance", beam = c(0.005, 0.0003)),
  arrays = ray_voxelize(canopy, 0.75, ground = attrs[, 1] == 2, foliage = read_vector("foliage"),
                        weighting = "relative", tree_attr = NULL),
  iad = ray_voxelize(canopy, 0.5, bounds, ground_class = 2, weighting = "strongest", inclination = TRUE,
                     leaf_classes = 4, wood_classes = 6)
)

expect_list <- function(got, want, info) {
  for (k in names(want)) {
    if (is.list(want[[k]])) expect_list(got[[k]], want[[k]], paste(info, k))
    else if (length(want[[k]]) == 0) expect_length(got[[k]], 0)  # empty arrays are written as c()
    else expect_equal(as.vector(got[[k]]), as.vector(want[[k]]), tolerance = 1e-9, info = paste(info, k))
  }
}

test_that("ray-traced grids and their summaries match Python", {
  for (name in names(grids)) {
    g <- grids[[name]]
    e <- expected[[name]]
    expect_equal(g$origin, e$origin, tolerance = 1e-9, info = name)
    expect_equal(g$voxel_size, e$voxel_size, info = name)
    expect_equal(as.double(g$shape), e$shape, info = name)
    expect_equal(paste(g$fields, collapse = ","), e$fields, info = name)
    expect_equal(z_levels(g), e$z_levels, tolerance = 1e-9, info = name)
    for (k in c("num_hits", "num_beams", "path_length", "state", "pad_fpl", "transmittance", "distance_from_ground")) {
      if (!is.null(e[[k]])) expect_equal(as.double(g[[k]]), as.double(e[[k]]), tolerance = 1e-9, info = paste(name, k))
    }
    expect_equal(dim(g$num_hits), dim(e$num_hits), info = name)
    c <- centers(g)
    expect_equal(c$X, e$X, tolerance = 1e-9, info = name)
    expect_equal(c$Z, e$Z, tolerance = 1e-9, info = name)
    expect_equal(profile(g), e$profile_pad, tolerance = 1e-9, info = name)
    expect_equal(profile(g, "free_path_length", min_beams = 3), e$profile_fpl, tolerance = 1e-9, info = name)
    expect_list(occlusion_profile(g, 0, 6.5), e$occ, paste(name, "occ"))
    expect_list(occlusion_profile(g, 0.5), e$occ_default, paste(name, "occ default"))
    expect_equal(observed_map(g, 0, 6.5), e$map, tolerance = 1e-9, info = name)
    expect_equal(g$observed, g$state >= STATES[["empty"]])
  }
  expect_error(ray_voxelize(canopy, 1, bounds, ground_class = 2, class_attr = "nope"), "nope")
  expect_error(ray_voxelize(canopy, 1, bounds, ground = TRUE), "values for")
  expect_error(ray_voxelize(canopy, 1, bounds, laser = "VZ-400", beam = c(0.1, 0.1)), "not both")
})

test_that("inclination distributions, wood, files and sampling match Python", {
  g <- grids$iad
  iad <- tree_iad(g)
  expect_equal(sort(names(iad)), sort(names(expected$tree_iad)))
  expect_list(iad, expected$tree_iad, "iad")
  lg <- leaf_area_grid_from_voxels(grids$classes, "lad_fpl")
  expect_equal(unclass(lg)$density, expected$lad_grid$density, tolerance = 1e-9)
  xyz <- read_matrix("sample_xyz")
  s <- tree_sampling(grids$classes, xyz, read_vector("sample_labels"), min_beams = 3, above = 1)
  expect_list(s, expected$sampling, "sampling")
  add_wood_volume(g, expected$wood_cyl)
  expect_equal(as.double(g$wood_volume), as.double(expected$wood_volume), tolerance = 1e-9)
  expect_equal(g$wood_volume_density, expected$wood_density, tolerance = 1e-9)
  out <- tempfile(fileext = ".vox")
  expect_equal(write(g, out), expected$n_vox)
  expect_identical(readLines(out), readLines(fixture("iad.vox")))
  txt <- tempfile(fileext = ".txt")
  expect_equal(write(g, txt, filled_only = TRUE), expected$n_txt)
  expect_identical(readLines(txt), readLines(fixture("iad.txt")))
  write_iad_csv(g, out)
  expect_identical(readLines(out), readLines(fixture("iad.csv")))
  unlink(c(out, txt))
  expect_equal(names(to_dict(g, c("num_hits", "pad_fpl"))), c("num_hits", "pad_fpl"))
})

test_that("a shots file, scanner beams and leaf projection match Python", {
  f <- ray_voxelize(fixture("plot.shots"), 0.5, bounds, ground_class = 2, leaf_classes = 4, wood_classes = 6,
                    occlusion = TRUE)
  expect_equal(as.double(f$num_hits), as.double(expected$file$num_hits))
  expect_equal(f$pad_fpl, expected$file$pad_fpl, tolerance = 1e-9)
  expect_list(occlusion_profile(f), expected$file$occ, "file occ")
  expect_equal(laser_spec("LMS-Q780"), expected$laser, tolerance = 1e-12)
  expect_error(laser_spec("nope"), "unknown")
  theta <- seq(0, pi / 2, length.out = 11)
  expect_equal(leaf_projection(theta, "ellipsoidal", 1.7), expected$g_ellipsoidal, tolerance = 1e-9)
  expect_equal(leaf_projection(theta, "erectophile"), expected$g_erectophile, tolerance = 1e-9)
})
