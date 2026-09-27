# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# Ray-traced voxel grids, as the Python package's sylva.voxels. The grid
# lives in the Rust core (a RayVoxelGrid external pointer); arrays are
# copied out as [nz, ny, nx], the Python package's order.

#' Voxel states and echo foliage codes
#'
#' `STATES`: the values of a grid's `state` (unobserved, occluded, empty,
#' filled). `EXCLUDED`, `PLANT`, `LEAF`, `WOOD`: per-echo foliage codes for
#' `ray_voxelize(foliage = )`.
#' @export
STATES <- c(unobserved = 0L, occluded = 1L, empty = 2L, filled = 3L)

#' @rdname STATES
#' @export
EXCLUDED <- 0L

#' @rdname STATES
#' @export
PLANT <- 1L

#' @rdname STATES
#' @export
LEAF <- 2L

#' @rdname STATES
#' @export
WOOD <- 3L

#' Beam geometry of a scanner known to AMAPVox
#'
#' @param name Scanner name, e.g. `"VZ-400"`, `"LMS-Q780"`, `"FARO-FOCUS-X330"`.
#' @return `c(diameter, divergence)`: beam diameter at exit (m) and full
#'   divergence (rad). An error for an unknown scanner.
#' @export
laser_spec <- function(name) {
  s <- core_laser_spec(name)
  if (is.null(s)) stop(sprintf("unknown laser '%s'", name))
  s
}

#' Leaf projection function of an analytic leaf angle distribution
#'
#' @param theta Beam zenith angle(s) (rad).
#' @param lad `"spherical"`, `"uniform"`, `"planophile"`, `"erectophile"`,
#'   `"plagiophile"`, `"extremophile"`, `"ellipsoidal"` (`lad_params =
#'   chi`) or `"twoParamBeta"` (`c(mu, nu)`).
#' @param lad_params Parameters of the ellipsoidal or beta distribution.
#' @return G for each angle.
#' @export
leaf_projection <- function(theta, lad = "spherical", lad_params = numeric()) {
  core_leaf_projection(as.double(theta), lad, as.double(lad_params))
}

ray_ptr <- function(grid) {
  if (!inherits(grid, "sylva_ray_voxel_grid")) stop("not a ray-traced voxel grid (see ray_voxelize())")
  .subset2(grid, "ptr")
}

as_ray_grid <- function(ptr) structure(list(ptr = ptr), class = "sylva_ray_voxel_grid")

grid_dims <- function(p) rev(p$shape())

has_echo_attr <- function(shots, attr) attr %in% names(unclass(shots)$echo_attrs)

#' Trace pulses through a voxel grid
#'
#' AMAPVox-style Beer-Lambert statistics, as the Python package's
#' `ray_voxelize()`: every pulse is traced through the grid and each voxel
#' accumulates beam counts, path lengths, beam sections and beam angles,
#' from which attenuation and plant, leaf and wood area density follow.
#' `shots` may be the path of a shots file, streamed a few row groups at a
#' time; echo labels then come from attributes rather than from `ground` or
#' `foliage` vectors.
#'
#' @param shots A `sylva_shots` (every scan of a plot in one frame, with the
#'   misses) or the path of a shots file.
#' @param voxel_size Voxel edge (m).
#' @param bounds `list(min_xyz, max_xyz)` (or a 2 x 3 matrix, rows min and
#'   max); by default the extent of the echoes.
#' @param dtm,ground,ground_class,ground_distance Ground echoes: a logical
#'   per echo, a `class_attr` code, or echoes at most `ground_distance`
#'   above a DTM (`raster()`), which also gives `distance_from_ground`.
#' @param foliage,leaf_classes,wood_classes,class_attr Foliage code per echo
#'   (`EXCLUDED`, `PLANT`, `LEAF`, `WOOD`), or the `class_attr` codes that
#'   mean leaf and wood.
#' @param tree_attr Echo attribute grouping echoes into trees (`NULL` pools
#'   them).
#' @param intensity_attr Echo attribute for the `relative` and `strongest`
#'   weightings.
#' @param weighting `"equal"`, `"full"`, `"first"`, `"relative"` or `"strongest"`.
#' @param attenuation One or more of `"fpl"`, `"ppl"`, `"transmittance"`, `"bailey"`.
#' @param laser,beam A scanner name for `laser_spec()`, or `c(diameter, divergence)`.
#' @param lad,lad_params Analytic leaf angle distribution (`leaf_projection()`).
#' @param inclination,n_iad_bins,knn_normal,triangle_lmax Estimate
#'   inclination distributions from echo normals.
#' @param occlusion Also trace beyond each pulse's last echo.
#' @param flat_top Start the path in each column's top voxel at the highest echo.
#' @param neighbour_prior_min_rays Top up voxels crossed by fewer weighted beams.
#' @param subvoxel_split,subvoxel_min_beams Sub-voxel grid for `exploration_rate`.
#' @param average_leaf_area Mean leaf area (m²) of the free path correction; 0 disables.
#' @param unbounded_range How far pulses without an echo are traced.
#' @return A `sylva_ray_voxel_grid`. Raw fields and derived metrics are read
#'   by name, `grid$num_hits` or `grid[["pad_fpl"]]`, as `[nz, ny, nx]`
#'   arrays; `grid$fields` and `grid$metrics` list them, and `grid$origin`,
#'   `grid$voxel_size`, `grid$shape` (`c(nx, ny, nz)`), `grid$observed` and
#'   `grid$tree_iad` describe the grid.
#' @export
ray_voxelize <- function(shots, voxel_size = 0.1, bounds = NULL, dtm = NULL, ground = NULL, ground_class = NULL,
                         ground_distance = 0.2, foliage = NULL, leaf_classes = integer(), wood_classes = integer(),
                         class_attr = "classification", tree_attr = "tree_id", intensity_attr = "intensity",
                         weighting = "equal", attenuation = "fpl", laser = NULL, beam = NULL, lad = "spherical",
                         lad_params = numeric(), inclination = FALSE, n_iad_bins = 18, knn_normal = 10,
                         triangle_lmax = 0.05, occlusion = FALSE, flat_top = FALSE, neighbour_prior_min_rays = 0,
                         subvoxel_split = 0, subvoxel_min_beams = 10, average_leaf_area = 0.005,
                         unbounded_range = Inf) {
  if (!is.null(laser)) {
    if (!is.null(beam)) stop("give laser or beam, not both")
    beam <- laser_spec(laser)
  }
  if (!is.null(bounds)) bounds <- if (is.list(bounds)) c(bounds[[1]], bounds[[2]]) else c(t(as.matrix(bounds)))
  opts <- list(voxel_size = as.double(voxel_size), bounds = if (is.null(bounds)) NULL else as.double(bounds),
               class_attr = class_attr, ground_class = if (is.null(ground_class)) NULL else as.double(ground_class),
               ground_distance = as.double(ground_distance), leaf_classes = as.double(leaf_classes),
               wood_classes = as.double(wood_classes), tree_attr = if (is.null(tree_attr)) "" else tree_attr,
               intensity_attr = intensity_attr, weighting = weighting, occlusion = isTRUE(occlusion),
               flat_top = isTRUE(flat_top), neighbour_prior_min_rays = as.double(neighbour_prior_min_rays),
               beam = if (is.null(beam)) NULL else as.double(beam), subvoxel_split = as.double(subvoxel_split),
               subvoxel_min_beams = as.double(subvoxel_min_beams), average_leaf_area = as.double(average_leaf_area),
               lad = lad, lad_params = as.double(lad_params), attenuation = as.character(attenuation),
               inclination = isTRUE(inclination), n_iad_bins = as.double(n_iad_bins),
               knn_normal = as.double(knn_normal), triangle_lmax = as.double(triangle_lmax),
               unbounded_range = as.double(unbounded_range))
  d <- if (is.null(dtm)) NULL else unclass(dtm)
  if (is.character(shots)) {
    if (!is.null(ground) || !is.null(foliage)) {
      stop("ground / foliage arrays need in-memory shots; label a shots file through its echo attributes")
    }
    return(as_ray_grid(core_ray_voxelize_file(path.expand(shots), d, opts)))
  }
  need <- function(attr) if (!has_echo_attr(shots, attr)) stop(sprintf("shots have no '%s' echo attribute", attr))
  if (is.null(ground) && !is.null(ground_class)) need(class_attr)
  if (is.null(foliage) && (length(leaf_classes) || length(wood_classes))) need(class_attr)
  if (weighting %in% c("relative", "strongest")) need(intensity_attr)
  as_ray_grid(core_ray_voxelize(unclass(shots), if (is.null(ground)) NULL else as.logical(ground),
                                if (is.null(foliage)) NULL else as.double(foliage), d, opts))
}

ray_values <- function(grid, name) {
  p <- ray_ptr(grid)
  v <- if (name %in% p$field_names()) p$field(name) else p$metric(name)
  dims <- grid_dims(p)
  if (name == "ground_height") dims <- dims[2:3]
  if (name == "subvoxel_counts") dims <- c(dims, length(v) / prod(dims))
  from_row_major(v, dims)
}

#' @export
`$.sylva_ray_voxel_grid` <- function(x, name) x[[name]]

#' @export
`[[.sylva_ray_voxel_grid` <- function(x, i, ...) {
  p <- ray_ptr(x)
  switch(i,
         ptr = p,
         origin = p$origin(),
         voxel_size = p$voxel_size(),
         shape = p$shape(),
         fields = p$field_names(),
         metrics = p$metric_names(),
         has_leaf = p$has_leaf(),
         has_wood = p$has_wood(),
         observed = ray_values(x, "state") >= STATES[["empty"]],
         tree_iad = tree_iad(x),
         ray_values(x, i))
}

#' @export
print.sylva_ray_voxel_grid <- function(x, ...) {
  s <- x$shape
  cat(sprintf("RayVoxelGrid(%dx%dx%d @ %s m)\n", s[1], s[2], s[3], format(x$voxel_size, digits = 6)))
  invisible(x)
}

#' Layer summaries of a ray-traced grid
#'
#' `occlusion_profile()`: what the scan saw of the canopy space (every
#' voxel from `min_height` above the ground, or the grid floor without a
#' DTM, up to `max_height`, by default the highest filled voxel), per layer
#' of one voxel. `observed_map()`: the share of each column's canopy space
#' observed, to find the parts of a plot to rescan. `profile()`: the mean
#' of a field or metric per layer over voxels entered by at least
#' `min_beams` pulses. `z_levels()`: the bottom of each layer; `centers()`
#' the voxel centres.
#'
#' @param grid,x A `sylva_ray_voxel_grid`.
#' @param min_height,max_height Canopy space (m above ground).
#' @param name Field or metric, e.g. `"pad_fpl"`.
#' @param min_beams Pulses for a voxel to count.
#' @param ... Unused.
#' @return `occlusion_profile()`: a list with `height`, `n_voxels`,
#'   `observed`, `occluded`, `unobserved`, `mean_beams` per layer and
#'   `total` (`observed`, `occluded`, `unobserved`, `top`).
#'   `observed_map()`: a `[ny, nx]` matrix (`NaN` without canopy space).
#'   `profile()`: one value per layer, bottom first (`NaN` where no voxel
#'   qualifies). `centers()`: `list(X, Y, Z)` of `[nz, ny, nx]` arrays.
#' @export
occlusion_profile <- function(grid, min_height = 0, max_height = NULL) {
  ray_ptr(grid)$occlusion_profile(as.double(min_height), if (is.null(max_height)) NULL else as.double(max_height))
}

#' @rdname occlusion_profile
#' @export
observed_map <- function(grid, min_height = 0, max_height = NULL) {
  p <- ray_ptr(grid)
  m <- p$observed_map(as.double(min_height), if (is.null(max_height)) NULL else as.double(max_height))
  s <- p$shape()
  matrix(m, s[2], s[1], byrow = TRUE)
}

#' @rdname occlusion_profile
#' @export
profile.sylva_ray_voxel_grid <- function(x, name = "pad_fpl", min_beams = 1, ...) {
  ray_ptr(x)$profile(name, as.double(min_beams))
}

#' @rdname occlusion_profile
#' @export
z_levels <- function(grid) ray_ptr(grid)$z_levels()

#' @rdname occlusion_profile
#' @export
centers <- function(grid) {
  p <- ray_ptr(grid)
  c <- p$centers()
  d <- grid_dims(p)
  list(X = from_row_major(c$x, d), Y = from_row_major(c$y, d), Z = from_row_major(c$z, d))
}

#' Inclination distributions, wood volume and files of a ray-traced grid
#'
#' `tree_iad()`: per-tree inclination angle distributions (normalised
#' `liad`, `wiad`, `piad` histograms over `bin_centres`, the G values and
#' de Wit types), named by tree id; empty unless the grid was built with
#' `inclination = TRUE`. `add_wood_volume()` rasterises QSM cylinders into
#' `wood_volume` and `wood_volume_density` (repeated calls add up; the grid
#' is changed in place). `to_dict()` copies arrays out by name. `write()`
#' writes an AMAPVox `.vox` file or a text table; `write_iad_csv()` the
#' per-tree distributions.
#'
#' @param grid,x A `sylva_ray_voxel_grid`.
#' @param qsms A QSM (an `n x 12` cylinder matrix, or a list with
#'   `cylinders`), or a list of them, in the grid's frame.
#' @param names Fields and metrics; every raw field if `NULL`.
#' @param path Output file.
#' @param format `"vox"` or `"text"`; from the extension if `NULL`.
#' @param include_unobserved Also write voxels no pulse reached.
#' @param filled_only Only write voxels holding echoes.
#' @param ... Passed to `base::write()` for other objects.
#' @return `write()`: the number of voxels written.
#' @export
tree_iad <- function(grid) {
  t <- ray_ptr(grid)$tree_iad()
  if (length(t) == 0) names(t) <- character()
  t
}

#' @rdname tree_iad
#' @export
add_wood_volume <- function(grid, qsms) {
  p <- ray_ptr(grid)
  one <- is.matrix(qsms) || (is.list(qsms) && !is.null(qsms$cylinders))
  for (q in if (one) list(qsms) else qsms) p$add_wood_volume(cylinder_rows(q) %||% matrix(0, 0, 12))
  invisible(grid)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' @rdname tree_iad
#' @export
to_dict <- function(grid, names = NULL) {
  if (is.null(names)) names <- grid$fields
  stats::setNames(lapply(names, function(n) grid[[n]]), names)
}

#' @rdname tree_iad
#' @export
write <- function(x, ...) UseMethod("write")

#' @export
write.default <- function(x, ...) base::write(x, ...)

#' @rdname tree_iad
#' @export
write.sylva_ray_voxel_grid <- function(x, path, format = NULL, include_unobserved = FALSE, filled_only = FALSE, ...) {
  if (is.null(format)) format <- if (tolower(tools::file_ext(path)) == "vox") "vox" else "text"
  ray_ptr(x)$write(path.expand(path), format, isTRUE(include_unobserved), isTRUE(filled_only))
}

#' @rdname tree_iad
#' @export
write_iad_csv <- function(grid, path) invisible(ray_ptr(grid)$write_iad_csv(path.expand(path)))

#' How well each tree was seen
#'
#' From a grid built with `occlusion = TRUE`: per tree (points labelled by
#' `labels`, negative ignored) its crown envelope (layer by layer the convex
#' hull of its points), the pulses that reached it and whether its top is
#' real, as the Python package's `tree_sampling()`.
#'
#' @param grid A `sylva_ray_voxel_grid`.
#' @param cloud A `sylva_cloud` or `n x 3` matrix in the grid's frame.
#' @param labels Tree id per point.
#' @param min_beams Pulses for a voxel to count as well sampled.
#' @param above Height (m) above each tree's top that is checked.
#' @return A list of columns, one row per tree: `tree_id`, `n_voxels`,
#'   `volume`, `observed_fraction`, `occluded_fraction`,
#'   `unobserved_fraction`, `median_beams`, `p10_beams`,
#'   `well_sampled_fraction`, `above_observed_fraction` and the matrix
#'   `beams_by_quarter`.
#' @export
tree_sampling <- function(grid, cloud, labels, min_beams = 10, above = 2) {
  ray_ptr(grid)$tree_sampling(leaf_xyz(cloud), as.double(labels), as.double(min_beams), as.double(above))
}
