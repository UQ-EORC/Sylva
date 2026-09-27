# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# Foliage for a structure model, as the Python package's sylva.leaves. Grids
# keep Python's (nz, ny, nx) order; mesh faces index vertices from 0, as in
# the Python package.

leaf_xyz <- function(points) {
  if (inherits(points, "sylva_cloud")) unclass(points)$xyz else as_xyz(points)
}

cylinder_rows <- function(model) {
  if (is.null(model)) return(NULL)
  if (is.list(model) && !is.null(model$cylinders)) model <- model$cylinders
  m <- matrix(as.double(model), ncol = 12)
  if (nrow(m)) m else NULL
}

as_faces <- function(f) {
  f <- as.matrix(f)
  storage.mode(f) <- "double"
  f
}

#' Generics for the foliage objects
#'
#' `area()`: leaf area per voxel of a leaf area grid (an array), or the area
#' of one leaf of a shape. `total_area()`: the total one-sided leaf area of
#' a grid or a leaf mesh. `scaled_to()`: a grid rescaled to a total area, or
#' a shape to an area per leaf. `profile()`: the vertical profile of a leaf
#' area grid or of a ray-traced grid. `cells()`: the voxels of a grid that
#' hold leaf area.
#'
#' @param x A `sylva_leaf_area_grid`, `sylva_leaf_shape`, `sylva_leaf_mesh`
#'   or `sylva_ray_voxel_grid`.
#' @param area Target area (m²).
#' @param ... Method arguments.
#' @export
area <- function(x, ...) UseMethod("area")

#' @rdname area
#' @export
total_area <- function(x, ...) UseMethod("total_area")

#' @rdname area
#' @export
scaled_to <- function(x, area, ...) UseMethod("scaled_to")

#' @rdname area
#' @export
profile <- function(x, ...) UseMethod("profile")

#' @export
profile.default <- function(x, ...) stats::profile(x, ...)

#' @rdname area
#' @export
cells <- function(x, ...) UseMethod("cells")

# ------------------------------------------------------------ leaf / wood

#' Wood and leaf labels for one tree
#'
#' `method = "gbs"` (default) is the graph-based separation of Tian and Li
#' (2022), with shells of 0.1-1 m and a 45 degree direction limit for trees
#' under 15 m and 0.5-3 m with 27 degrees for taller ones. `method =
#' "passage"` is the path-passage and anisotropy filter of the QSM input,
#' run on the cloud thinned to `voxel_size` with a second anisotropy scale
#' of 10 cm.
#'
#' @param cloud A `sylva_cloud` or `n x 3` matrix of one tree.
#' @param voxel_size Work on the cloud thinned to this spacing (m); every
#'   point takes the label of its nearest thinned point. 0 uses every point.
#' @param method `"gbs"` or `"passage"`.
#' @param ... Options under the Python names: for `"gbs"`, `intervals`,
#'   `max_angle`, `linearity`, `circle_error`, `graph_k`, `max_edge`,
#'   `base_height`, `min_points`; for `"passage"`, the options of the wood
#'   filter (`threshold` is its `high_threshold`).
#' @return Logical per point, `TRUE` for wood.
#' @export
classify_leaf_wood <- function(cloud, voxel_size = 0.02, method = "gbs", ...) {
  opts <- list(...)
  if (!method %in% c("gbs", "passage")) stop("method must be 'passage' or 'gbs'")
  if (method == "passage" && !is.null(opts$threshold)) {
    opts$high_threshold <- opts$threshold
    opts$threshold <- NULL
  }
  core_classify_leaf_wood(leaf_xyz(cloud), as.double(voxel_size), method, opts)
}

# ------------------------------------------------------------ angles

as_lad <- function(d) {
  d$mean_deg <- d$mean * 180 / pi
  structure(d, class = "sylva_leaf_angle_distribution")
}

#' Leaf inclination angle distribution
#'
#' From leaf points, with normals from a PCA over `k` neighbours (Vicari et
#' al. 2019): by default the points are thinned to `res` (0: 3.5 times the
#' median spacing) and each is weighted by the leaf area it stands for.
#' With `inclinations = TRUE` the first argument holds inclinations (rad).
#' `leaf_angle_distribution_from_type()` gives a textbook de Wit (1965)
#' distribution. `g()` is the leaf projection function of a distribution
#' (Wilson 1960).
#'
#' @param leaf_points A `sylva_cloud` or `n x 3` matrix of leaf points, or
#'   inclinations with `inclinations = TRUE`.
#' @param k Neighbours for the normals.
#' @param n_bins Bins over 0-90 degrees.
#' @param weights Weights for inclinations or unthinned points.
#' @param inclinations Treat the first argument as inclinations.
#' @param res Thinning cube (m); 0 picks it, `NULL` disables thinning and
#'   area weighting.
#' @param name `"spherical"`, `"uniform"`, `"planophile"`, `"erectophile"`,
#'   `"plagiophile"` or `"extremophile"`.
#' @param lad A `sylva_leaf_angle_distribution`.
#' @param beam_zenith Beam zenith angle(s) (rad).
#' @return A `sylva_leaf_angle_distribution`: `bin_centres` (rad),
#'   `density`, `mean` and `std` (rad), `mean_deg`, the beta parameters
#'   `beta_a`, `beta_b`, Campbell's `chi` and the nearest `de_wit` type.
#' @export
leaf_angle_distribution <- function(leaf_points, k = 12, n_bins = 18, weights = NULL, inclinations = FALSE,
                                    res = 0) {
  w <- if (is.null(weights)) NULL else as.double(weights)
  if (isTRUE(inclinations)) {
    incl <- as.double(leaf_points)
  } else if (!is.null(res)) {
    a <- core_point_leaf_area(leaf_xyz(leaf_points), as.double(res), as.integer(k))
    incl <- a$inclination
    w <- a$area
  } else {
    incl <- core_leaf_inclinations(leaf_xyz(leaf_points), as.integer(k))$inclination
  }
  as_lad(core_leaf_angle_distribution(incl, w, as.integer(n_bins)))
}

#' @rdname leaf_angle_distribution
#' @export
leaf_angle_distribution_from_type <- function(name = "spherical", n_bins = 18) {
  as_lad(core_leaf_de_wit(name, as.integer(n_bins)))
}

#' @rdname leaf_angle_distribution
#' @export
g <- function(lad, beam_zenith) {
  core_leaf_projection_histogram(lad$bin_centres, lad$density, as.double(beam_zenith))
}

#' @export
print.sylva_leaf_angle_distribution <- function(x, ...) {
  cat(sprintf("<sylva_leaf_angle_distribution> mean %.1f deg, chi %.2f, %s\n", x$mean_deg, x$chi,
              if (is.null(x$de_wit)) "no de Wit type" else x$de_wit))
  invisible(x)
}

# ------------------------------------------------------------ leaf area grids

#' Leaf area density on a regular grid
#'
#' As the Python package's `LeafAreaGrid`. Build one with
#' `leaf_area_density()` (from leaf points, a lower bound where foliage was
#' occluded) or `leaf_area_grid_from_voxels()` (from a ray-traced grid).
#' `area()`, `total_area()`, `scaled_to()`, `profile()` and `cells()` read
#' it.
#'
#' @param origin Minimum corner.
#' @param voxel_size Voxel edge (m).
#' @param density Leaf area density (m² m⁻³), an array `[nz, ny, nx]`.
#' @return A `sylva_leaf_area_grid`.
#' @export
leaf_area_grid <- function(origin, voxel_size, density) {
  density <- as.array(density)
  if (length(dim(density)) != 3) stop("density must be a 3-d array [nz, ny, nx]")
  storage.mode(density) <- "double"
  structure(list(origin = as.double(origin), voxel_size = as.double(voxel_size), density = density),
            class = "sylva_leaf_area_grid")
}

grid_from_core <- function(g) leaf_area_grid(g$origin, g$voxel_size, from_row_major(g$density, g$dim))

leaf_grid_args <- function(x) {
  x <- unclass(x)
  list(as.double(x$origin), as.double(x$voxel_size), as.double(dim(x$density)), row_major(x$density))
}

#' @rdname leaf_area_grid
#' @param leaf_points A `sylva_cloud` or `n x 3` matrix of leaf points.
#' @param res Thinning cube (m); `NULL` picks 3.5 times the median spacing.
#' @param k Neighbours for the normals.
#' @export
leaf_area_density <- function(leaf_points, voxel_size = 0.25, res = NULL, k = 12) {
  grid_from_core(core_leaf_area_density(leaf_xyz(leaf_points), as.double(voxel_size),
                                        if (is.null(res)) 0 else as.double(res), as.integer(k)))
}

#' @rdname leaf_area_grid
#' @param grid A `sylva_ray_voxel_grid`.
#' @param field Density field: `"lad_fpl"` for leaf area (grids built with
#'   `leaf_classes`), `"pad_fpl"` for plant area. Unobserved voxels become 0.
#' @export
leaf_area_grid_from_voxels <- function(grid, field = "pad_fpl") {
  grid_from_core(core_leaf_grid_from_voxels(ray_ptr(grid), field))
}

#' @export
area.sylva_leaf_area_grid <- function(x, ...) {
  from_row_major(do.call(core_leaf_grid_area, leaf_grid_args(x)), dim(unclass(x)$density))
}

#' @export
total_area.sylva_leaf_area_grid <- function(x, ...) do.call(core_leaf_grid_total_area, leaf_grid_args(x))

#' @export
scaled_to.sylva_leaf_area_grid <- function(x, area, ...) {
  d <- do.call(core_leaf_grid_scaled, c(leaf_grid_args(x), list(as.double(area))))
  leaf_area_grid(unclass(x)$origin, unclass(x)$voxel_size, from_row_major(d, dim(unclass(x)$density)))
}

#' @export
profile.sylva_leaf_area_grid <- function(x, ...) do.call(core_leaf_grid_profile, leaf_grid_args(x))

#' @export
cells.sylva_leaf_area_grid <- function(x, ...) do.call(core_leaf_grid_cells, leaf_grid_args(x))

#' @export
print.sylva_leaf_area_grid <- function(x, ...) {
  d <- dim(unclass(x)$density)
  cat(sprintf("<sylva_leaf_area_grid> %dx%dx%d @ %g m, %.3g m2 of leaves\n", d[3], d[2], d[1],
              unclass(x)$voxel_size, total_area(x)))
  invisible(x)
}

# ------------------------------------------------------------ leaf shapes

#' Leaf shapes
#'
#' The blade one leaf is cut from, as the Python package's `LeafShape`: in
#' unit leaf space (along, across, up; base at the origin, tip at along = 1,
#' greatest width 1) and scaled by `length` along and `width` across and
#' up. The default is the built-in six-sided blade at 8 x 4 cm.
#' `leaf_shape_from_mesh()` and `leaf_shape_from_obj()` take a mesh of one
#' leaf (base at the smallest x, tip along +x) and by default map it into
#' unit leaf space, its own extent becoming its size. `resized()` changes
#' the size, `scaled_to()` sets the area per leaf, `area()` gives it, and
#' `single_leaf_area()` is the area of one leaf of a given size.
#' `default_leaf()` is the blade `add_leaves()` uses when none is given, and
#' `set_default_leaf()` sets it for the session, returning the previous one.
#'
#' @param vertices `n x 3` matrix in unit leaf space (or, for
#'   `leaf_shape_from_mesh()`, the mesh; `n x 2` for a flat blade).
#' @param faces `m x 3` matrix of 0-based vertex indices.
#' @param length,width Leaf length and greatest width (m); `NULL` keeps the
#'   current or the mesh's own.
#' @param normalise Map the mesh into unit leaf space.
#' @param path OBJ file of one leaf; only `v` and `f` lines are read.
#' @param shape A `sylva_leaf_shape`.
#' @return A `sylva_leaf_shape`: `vertices`, `faces`, `length`, `width`.
#' @export
leaf_shape <- function(vertices = NULL, faces = NULL, length = 0.08, width = 0.04) {
  if (!is.null(vertices)) vertices <- as_xyz(vertices)
  if (!is.null(faces)) faces <- as_faces(faces)
  structure(core_leaf_shape(vertices, faces, as.double(length), as.double(width)), class = "sylva_leaf_shape")
}

as_shape <- function(s) structure(s, class = "sylva_leaf_shape")

unit_blade <- function() {
  s <- core_leaf_shape(NULL, NULL, 0.08, 0.04)
  list(vertices = s$vertices, faces = s$faces)
}

#' @rdname leaf_shape
#' @export
resized <- function(shape, length = NULL, width = NULL) {
  s <- unclass(shape)
  leaf_shape(s$vertices, s$faces, if (is.null(length)) s$length else length, if (is.null(width)) s$width else width)
}

#' @export
area.sylva_leaf_shape <- function(x, ...) {
  s <- unclass(x)
  core_leaf_shape_area(s$vertices, as_faces(s$faces), s$length, s$width)
}

#' @export
scaled_to.sylva_leaf_shape <- function(x, area, ...) {
  s <- unclass(x)
  as_shape(core_leaf_shape_scaled(s$vertices, as_faces(s$faces), s$length, s$width, as.double(area)))
}

#' @rdname leaf_shape
#' @export
leaf_shape_from_mesh <- function(vertices, faces, length = NULL, width = NULL, normalise = TRUE) {
  v <- as.matrix(vertices)
  storage.mode(v) <- "double"
  if (ncol(v) == 2) v <- cbind(v, 0)
  if (ncol(v) != 3 || nrow(v) == 0) stop("vertices must be (V, 3) or (V, 2)")
  f <- as_faces(faces)
  if (ncol(f) != 3) stop("vertices must be (V, 3) and faces (F, 3)")
  as_shape(core_leaf_shape_from_mesh(v, f, length, width, isTRUE(normalise)))
}

#' @rdname leaf_shape
#' @export
leaf_shape_from_obj <- function(path, length = NULL, width = NULL, normalise = TRUE) {
  as_shape(core_leaf_shape_from_obj(path.expand(path), length, width, isTRUE(normalise)))
}

#' @rdname leaf_shape
#' @export
single_leaf_area <- function(length, width, shape = NULL) {
  if (is.null(shape)) return(core_single_leaf_area(as.double(length), as.double(width), NULL, NULL))
  s <- unclass(shape)
  core_single_leaf_area(as.double(length), as.double(width), s$vertices, as_faces(s$faces))
}

leaf_state <- new.env(parent = emptyenv())

#' @rdname leaf_shape
#' @export
default_leaf <- function() {
  if (is.null(leaf_state$default)) leaf_state$default <- leaf_shape()
  leaf_state$default
}

#' @rdname leaf_shape
#' @export
set_default_leaf <- function(shape = NULL, length = NULL, width = NULL) {
  was <- default_leaf()
  leaf_state$default <- resized(if (is.null(shape)) was else shape, length, width)
  invisible(was)
}

#' @export
print.sylva_leaf_shape <- function(x, ...) {
  s <- unclass(x)
  cat(sprintf("<sylva_leaf_shape> %g x %g m, %d triangles, %.3g m2 per leaf\n", s$length, s$width,
              nrow(s$faces), area(x)))
  invisible(x)
}

# ------------------------------------------------------------ leaf insertion

#' Leaf polygons for a QSM
#'
#' Each voxel of the leaf area grid receives leaves until its area is met,
#' centred on leaf points in it (uniformly inside it otherwise), with
#' normals drawn from `angles` and a uniform azimuth; leaves within
#' `max_branch_distance` of a cylinder point away from it. A total area is
#' first spread over `leaf_points` by `leaf_area_density()`. Leaves may
#' intersect (no collision test is made). `to_obj()` writes the leaves and
#' `write_tree_obj()` the wood and the leaves to one OBJ file.
#'
#' @param model The tree's QSM (its `n x 12` cylinder matrix, or a list
#'   with `cylinders`), or `NULL` for leaves without wood.
#' @param leaf_area A `sylva_leaf_area_grid`, or a total area (m²).
#' @param angles A `sylva_leaf_angle_distribution` or a de Wit type name.
#' @param leaf_points Leaf points to centre leaves on; needed with a total.
#' @param leaf_length,leaf_width Leaf size (m); the shape's own by default.
#' @param shape Blade (`leaf_shape()`); `default_leaf()` if `NULL`.
#' @param max_branch_distance Leaves this close (m) to a cylinder attach to it.
#' @param jitter Random offset (m) of leaf centres.
#' @param seed Random seed.
#' @return A `sylva_leaf_mesh`: `vertices`, `faces` (0-based), and per leaf
#'   `centres`, `normals`, `inclination` (rad) and `cylinder` (row of the
#'   nearest cylinder, 0-based, -1 if none in reach); `leaf_area` is the
#'   area of one leaf. `length()` is the number of leaves.
#' @export
add_leaves <- function(model, leaf_area, angles = "spherical", leaf_points = NULL, leaf_length = NULL,
                       leaf_width = NULL, shape = NULL, max_branch_distance = 0.5, jitter = 0.01, seed = 1) {
  if (is.character(angles)) angles <- leaf_angle_distribution_from_type(angles)
  seeds <- if (is.null(leaf_points)) matrix(0, 0, 3) else leaf_xyz(leaf_points)
  grid <- NULL
  total <- 0
  if (inherits(leaf_area, "sylva_leaf_area_grid")) {
    a <- leaf_grid_args(leaf_area)
    grid <- list(origin = a[[1]], voxel_size = a[[2]], dim = a[[3]], density = a[[4]])
  } else {
    total <- as.double(leaf_area)
  }
  s <- unclass(resized(if (is.null(shape)) default_leaf() else shape, leaf_length, leaf_width))
  m <- core_add_leaves(grid, total, seeds, as.double(angles$bin_centres), as.double(angles$density),
                       cylinder_rows(model), s$vertices, as_faces(s$faces), s$length, s$width,
                       as.double(max_branch_distance), as.double(jitter), as.double(seed))
  structure(m, class = "sylva_leaf_mesh")
}

#' @export
length.sylva_leaf_mesh <- function(x) nrow(unclass(x)$centres)

#' @export
total_area.sylva_leaf_mesh <- function(x, ...) unclass(x)$leaf_area * length(x)

#' @export
print.sylva_leaf_mesh <- function(x, ...) {
  cat(sprintf("<sylva_leaf_mesh> %d leaves, %.3g m2\n", length(x), total_area(x)))
  invisible(x)
}

#' @rdname add_leaves
#' @param leaf_mesh,mesh A `sylva_leaf_mesh`.
#' @param path Output OBJ file.
#' @export
to_obj <- function(mesh, path) {
  m <- unclass(mesh)
  invisible(core_write_obj(path.expand(path), list(list(vertices = m$vertices, faces = as_faces(m$faces))), "leaves"))
}

#' @rdname add_leaves
#' @param sides Facets around each cylinder.
#' @param contiguous One continuous tube per branch.
#' @export
write_tree_obj <- function(path, model, leaf_mesh, sides = 12, contiguous = FALSE) {
  m <- unclass(leaf_mesh)
  invisible(core_write_tree_obj(path.expand(path), cylinder_rows(model), m$vertices, as_faces(m$faces),
                                as.integer(sides), isTRUE(contiguous)))
}
