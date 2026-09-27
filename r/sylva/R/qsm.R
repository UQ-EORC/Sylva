# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# Quantitative structure models, as the Python package's sylva.qsm. A model
# is a list holding its cylinders as a data frame (one row per cylinder,
# the Python columns); a model's derived values are read with `$` as the
# Python properties are. Mesh faces index vertices from 0, as in Python.

#' Cylinder columns of a QSM
#'
#' `sx, sy, sz` start (base) of the cylinder; `ax, ay, az` unit axis,
#' pointing away from the tree base; `length`, `radius` (m); `parent` row of
#' the parent cylinder (0-based, -1 for the first); `branch_order` (0 stem,
#' 1 first-order branch, ...); `branch_id`; `n_points` the fit used (0 where
#' the radius came from priors).
#' @export
QSM_COLUMNS <- c("sx", "sy", "sz", "ax", "ay", "az", "length", "radius", "parent", "branch_order", "branch_id",
                 "n_points")

# An n x 12 double matrix of cylinders from a model, data frame, matrix or
# row-major vector.
qsm_rows <- function(x) {
  if (inherits(x, "sylva_qsm")) x <- unclass(x)$cylinders
  if (is.list(x) && !is.data.frame(x) && !is.null(x$cylinders)) x <- x$cylinders
  if (is.null(x)) return(matrix(0, 0, 12))
  if (is.data.frame(x) || is.matrix(x)) {
    m <- as.matrix(x)
    storage.mode(m) <- "double"
    if (ncol(m) != 12) m <- matrix(t(m), ncol = 12, byrow = TRUE)
  } else {
    m <- matrix(as.double(x), ncol = 12, byrow = TRUE)
  }
  dimnames(m) <- NULL
  m
}

new_qsm <- function(m) {
  df <- as.data.frame(qsm_rows(m))
  names(df) <- QSM_COLUMNS
  structure(list(cylinders = df), class = "sylva_qsm")
}

#' A tree as connected cylinders
#'
#' Build one with `build_qsm()`, read one with `qsm_from_csv()`, or wrap a
#' cylinder table with `qsm()`. The cylinders are `model$cylinders`, a data
#' frame with the columns [QSM_COLUMNS]; `model$start`, `model$axis` and
#' `model$end` are `n x 3` matrices, `model$volumes` the volume of each
#' cylinder (m³), and `model$total_volume`, `model$stem_volume` (order 0),
#' `model$branch_volume`, `model$total_length` (m), `model$max_branch_order`
#' and `model$dbh` (m, at 1.3 m above the base, from the stem cylinders)
#' the Python properties. `length(model)` is the number of cylinders.
#'
#' @param cylinders A data frame or `n x 12` matrix with the columns
#'   [QSM_COLUMNS], or their values row by row.
#' @param path CSV written by `to_csv()`.
#' @return A `sylva_qsm`.
#' @export
qsm <- function(cylinders) new_qsm(cylinders)

#' @rdname qsm
#' @export
qsm_from_csv <- function(path) new_qsm(core_qsm_read_csv(path.expand(path)))

#' @export
length.sylva_qsm <- function(x) nrow(unclass(x)$cylinders)

#' @export
print.sylva_qsm <- function(x, ...) {
  cat(sprintf("<sylva_qsm> %d cylinders, %.4g m3\n", length(x), x$total_volume))
  invisible(x)
}

#' @export
`$.sylva_qsm` <- function(x, name) qsm_field(x, name)

#' @export
`[[.sylva_qsm` <- function(x, i, ...) qsm_field(x, i)

qsm_field <- function(x, name) {
  cyl <- unclass(x)$cylinders
  totals <- function() core_qsm_totals(qsm_rows(x))
  switch(name,
         cylinders = cyl,
         start = unname(as.matrix(cyl[, 1:3])),
         axis = unname(as.matrix(cyl[, 4:6])),
         end = core_qsm_ends(qsm_rows(x)),
         volumes = core_qsm_volumes(qsm_rows(x)),
         total_volume = totals()$total_volume,
         stem_volume = totals()$stem_volume,
         branch_volume = totals()$branch_volume,
         total_length = totals()$total_length,
         max_branch_order = as.integer(totals()$max_branch_order),
         dbh = core_qsm_dbh(qsm_rows(x)),
         NULL)
}

#' One column of a QSM's cylinders
#'
#' @param model A `sylva_qsm`.
#' @param name A name from [QSM_COLUMNS], e.g. `"radius"`.
#' @return A numeric vector, one value per cylinder.
#' @export
column <- function(model, name) {
  if (!name %in% QSM_COLUMNS) stop(sprintf("'%s' is not a QSM column", name), call. = FALSE)
  unclass(model)$cylinders[[name]]
}

#' Tree architecture from a QSM
#'
#' `metrics()`: height, DBH, volumes and lengths (total, stem, branches, and
#' per branch order), branch and tip counts, path fraction, crown base
#' height, stem lean and its direction, sweep, the stem taper profile, the
#' crown outlined by the branches (as `crown_shape()`), median insertion and
#' zenith angles of first-order branches (deg), and the share of volume and
#' length fitted to points rather than filled in by the priors. Heights are
#' above the stem base. `branches()`: one row per branch. `summary()`: the
#' headline numbers with units in the names.
#'
#' @param model,object A `sylva_qsm`.
#' @param crown_branch_length Shortest first-order branch (m) that marks the
#'   crown base.
#' @param crown_slice Slice height (m) of the stacked crown hulls.
#' @param ... Unused.
#' @return `metrics()`: a list as the Python `QSM.metrics()`. `branches()`: a
#'   data frame with `id`, `order`, `parent`, `n_cylinders`, `length`,
#'   `volume`, `base_radius`, `mean_radius`, `base_height`, `tip_height`,
#'   `insertion_angle`, `zenith`, `azimuth`, `tortuosity`, `n_children` and
#'   `measured_fraction`. `summary()`: a list with `n_cylinders`,
#'   `total_volume_m3`, `stem_volume_m3`, `branch_volume_m3`,
#'   `total_length_m`, `max_branch_order` and `dbh_m`.
#' @export
metrics <- function(model, crown_branch_length = 1.0, crown_slice = 0.5) {
  core_qsm_metrics(qsm_rows(model), as.double(crown_branch_length), as.double(crown_slice))
}

#' @rdname metrics
#' @export
branches <- function(model) {
  b <- core_qsm_branches(qsm_rows(model))
  for (k in c("id", "order", "parent", "n_cylinders", "n_children")) b[[k]] <- as.integer(b[[k]])
  as.data.frame(b)
}

#' @rdname metrics
#' @export
summary.sylva_qsm <- function(object, ...) {
  s <- core_qsm_summary(qsm_rows(object))
  list(n_cylinders = as.integer(s$n_cylinders), total_volume_m3 = s$total_volume, stem_volume_m3 = s$stem_volume,
       branch_volume_m3 = s$branch_volume, total_length_m = s$total_length,
       max_branch_order = as.integer(s$max_branch_order), dbh_m = s$dbh)
}

#' Cut a QSM at a height
#'
#' `volume_above()`: cylinder volume above a horizontal plane, a cylinder
#' crossing it counted by the share of its axis above it. `above()`: the
#' model above the plane; a cylinder crossing it keeps the part above (cut
#' where its axis crosses the plane, so a leaning one keeps a slanted stub),
#' one below it is dropped, and a child whose parent went becomes a branch
#' of its own. Use them to join a `buttress_mesh()` to the model.
#'
#' @param model A `sylva_qsm`.
#' @param z Absolute height of the plane (e.g. a buttress's `top_z`).
#' @return `volume_above()`: volume (m³). `above()`: a new `sylva_qsm`.
#' @export
volume_above <- function(model, z) core_qsm_volume_above(qsm_rows(model), as.double(z))

#' @rdname volume_above
#' @export
above <- function(model, z) new_qsm(core_qsm_above(qsm_rows(model), as.double(z)))

#' Triangle mesh of a QSM's cylinders
#'
#' @param model A `sylva_qsm`.
#' @param sides Facets around each cylinder.
#' @param contiguous One continuous tube per branch instead of one closed
#'   tube per cylinder: consecutive cylinders share a ring, so a branch has
#'   no caps inside it. Each branch is still its own closed surface.
#' @return A list: `vertices` (`n x 3`), `faces` (`m x 3`, 0-based vertex
#'   indices) and `owner` (0-based cylinder row of each face).
#' @export
mesh <- function(model, sides = 12, contiguous = FALSE) {
  core_qsm_mesh(qsm_rows(model), as.double(sides), isTRUE(contiguous))
}

#' Write QSMs, buttresses and meshes to files
#'
#' `to_csv()`: a model's cylinders as CSV with a header of [QSM_COLUMNS]
#' (read back with `qsm_from_csv()`), or a plot's table. `to_treefile()`:
#' the raycloudtools/treetools `_trees.txt` format. `to_obj()`: a Wavefront
#' OBJ (a model as object `tree_1`, a buttress as `buttress`, a joined mesh
#' as `buttress` and `wood`, a leaf mesh as `leaves`). `to_ply()`: a binary
#' PLY with face colours (a model by branch order, brown stem to green
#' twigs; a joined mesh with the buttress darker than the wood).
#'
#' @param x A `sylva_qsm`, `sylva_buttress`, `sylva_tree_mesh`,
#'   `sylva_leaf_mesh` or (for `to_csv()`) `sylva_plot_qsms`.
#' @param model A `sylva_qsm`.
#' @param path Output file.
#' @param sides Facets around each cylinder.
#' @param contiguous One continuous tube per branch (see `mesh()`).
#' @param color One RGB triple (0-255) for every face, or `NULL` for the
#'   default colours.
#' @param ... Method arguments.
#' @export
to_csv <- function(x, path, ...) UseMethod("to_csv")

#' @rdname to_csv
#' @export
to_csv.sylva_qsm <- function(x, path, ...) invisible(core_qsm_write_csv(qsm_rows(x), path.expand(path)))

#' @rdname to_csv
#' @export
to_treefile <- function(model, path) invisible(core_qsm_write_treefile(qsm_rows(model), path.expand(path)))

#' @rdname to_csv
#' @export
to_obj <- function(x, path, ...) UseMethod("to_obj")

#' @rdname to_csv
#' @export
to_obj.sylva_qsm <- function(x, path, sides = 12, contiguous = FALSE, ...) {
  invisible(core_qsm_write_obj(path.expand(path), qsm_rows(x), as.double(sides), isTRUE(contiguous)))
}

#' @rdname to_csv
#' @export
to_ply <- function(x, path, ...) UseMethod("to_ply")

rgb_arg <- function(color) if (is.null(color)) NULL else as.double(color)

#' @rdname to_csv
#' @export
to_ply.sylva_qsm <- function(x, path, sides = 12, color = NULL, contiguous = FALSE, ...) {
  invisible(core_qsm_write_ply(path.expand(path), qsm_rows(x), as.double(sides), rgb_arg(color), isTRUE(contiguous)))
}

#' Write meshes to OBJ or PLY
#'
#' `write_obj()`: several meshes to one OBJ file, one named object each.
#' `write_ply_mesh()`: one binary little-endian PLY triangle mesh, vertices
#' stored as float32.
#'
#' @param path Output file.
#' @param meshes A list of meshes, each a list with `vertices` (`n x 3`)
#'   and `faces` (`m x 3`, 0-based), e.g. from `mesh()` for every tree of a
#'   plot.
#' @param names Object names; `tree_1`, `tree_2`, ... if `NULL`.
#' @param vertices An `n x 3` matrix.
#' @param faces An `m x 3` matrix of 0-based vertex indices.
#' @param face_rgb Optional colour per face: an `m x 3` matrix (0-255), or
#'   one triple for every face.
#' @export
write_obj <- function(path, meshes, names = NULL) {
  parts <- lapply(meshes, function(m) {
    v <- if (!is.null(m$vertices)) m$vertices else m[[1]]
    f <- if (!is.null(m$faces)) m$faces else m[[2]]
    list(vertices = matrix(as.double(v), ncol = 3), faces = matrix(as.double(f), ncol = 3))
  })
  n <- if (length(names)) as.character(names[seq_along(parts)]) else paste0("tree_", seq_along(parts))
  invisible(core_write_obj(path.expand(path), parts, n))
}

#' @rdname write_obj
#' @export
write_ply_mesh <- function(path, vertices, faces, face_rgb = NULL) {
  faces <- matrix(as.double(faces), ncol = 3)
  if (!is.null(face_rgb)) {
    face_rgb <- if (length(face_rgb) == 3) matrix(rep(as.double(face_rgb), each = nrow(faces)), ncol = 3)
                else matrix(as.double(face_rgb), ncol = 3)
  }
  invisible(core_write_ply_mesh(path.expand(path), matrix(as.double(vertices), ncol = 3), faces, face_rgb))
}

# ------------------------------------------------------------ fits

#' Cylinder fits to 3-D points
#'
#' `fit_cylinder()`: least squares through points on a roughly cylindrical
#' surface (a stem section). `fit_cylinder_ransac()`: a fit that tolerates
#' outliers (leaves, twigs, noise).
#'
#' @param xyz An `n x 3` matrix or a `sylva_cloud`.
#' @param axis_init Starting axis direction; the principal direction of the
#'   points if `NULL`.
#' @param threshold Inlier distance from the surface (m).
#' @param iterations RANSAC trials.
#' @param sample_size Points per trial fit.
#' @param seed Random seed.
#' @return A list: `point` (on the axis), `axis` (unit vector), `radius` and
#'   `rmse` (m); `fit_cylinder_ransac()` adds `inliers` (logical per point).
#' @export
fit_cylinder <- function(xyz, axis_init = NULL) {
  core_fit_cylinder(xyz_of(xyz), if (is.null(axis_init)) NULL else as.double(axis_init))
}

#' @rdname fit_cylinder
#' @export
fit_cylinder_ransac <- function(xyz, threshold = 0.02, iterations = 100, sample_size = 12, seed = 0) {
  core_fit_cylinder_ransac(xyz_of(xyz), as.double(threshold), as.double(iterations), as.double(sample_size),
                           as.double(seed))
}

base_arg <- function(base_xy) if (is.null(base_xy)) NULL else as.double(base_xy[1:2])

#' Graph skeleton of one tree
#'
#' The first stage of `build_qsm()`: geodesic distance from the base over a
#' kNN graph, binned and split into connected segments.
#'
#' @param cloud One tree's (wood) points.
#' @param base_xy Stem position; the lowest points if `NULL`.
#' @param k Neighbours in the kNN graph.
#' @param max_edge Longest graph edge (m).
#' @param bin_length Width of the geodesic distance bins (m).
#' @return A list: `segment_id` per point (-1 if disconnected from the base),
#'   `geodesic` distance from the base per point (m), segment `centres` and
#'   `(child, parent)` `edges` (0-based).
#' @export
skeletonize <- function(cloud, base_xy = NULL, k = 15, max_edge = 1.0, bin_length = 0.1) {
  core_skeletonize(xyz_of(cloud), base_arg(base_xy), as.double(k), as.double(max_edge), as.double(bin_length))
}

#' A QSM for one segmented tree
#'
#' Skeleton nodes (geodesic bins of `bin_length`) are smoothed and a RANSAC
#' circle is fitted to each node's points in the plane perpendicular to the
#' skeleton; radii are then regularised along every root-to-tip path by an
#' allometric prior, made non-increasing, interpolated over gaps and shared
#' among children by a pipe model. See the Python `sylva.qsm.build_qsm` for
#' every setting; the defaults are the same. Run `wood_points()` first on
#' leafy trees.
#'
#' @param cloud One segmented tree, ideally wood only, in metres with z up.
#' @param base_xy Stem position (e.g. `c(tree$x, tree$y)`); the lowest points
#'   if `NULL`.
#' @param k,max_edge kNN graph: neighbours per point and longest edge (m).
#' @param bin_length Geodesic shell width (m); sets the cylinder length.
#' @param min_points Smallest segment kept.
#' @param ransac_threshold Circle inlier distance (m).
#' @param max_radius Upper limit on any radius (m).
#' @param taper_limit A child may be at most this times its parent's radius.
#' @param max_rmse Circle fits with a larger RMSE (m) are rejected.
#' @param smooth_steps Smoothing passes over skeleton node positions.
#' @param apex_radius Smallest tip radius (m).
#' @param min_arc_deg Contiguous arc a circle fit must cover (degrees).
#' @param min_inlier_fraction,branch_min_inlier_fraction Inlier share a stem
#'   / branch circle needs.
#' @param prune_points Unmeasured leaf segments with fewer points are pruned.
#' @param fit_min_points Segments with fewer points take their radius from
#'   the taper model.
#' @param crop_length Leafy tips with less subtree length (m) are not
#'   reconstructed.
#' @param butt_height Below this height (m) only the largest component per
#'   shell is kept.
#' @param relative_tolerance Refit band around the circle, as a fraction of
#'   its radius.
#' @param base_radius Breast-height radius (m) anchoring the taper prior; 0
#'   estimates it.
#' @param allometry_tolerance Weak fits further than this fraction from the
#'   prior are replaced.
#' @param buttress_equivalent_area,buttress_max_inlier_fraction Use an
#'   equivalent-area radius where a circle explains fewer than that share of
#'   a section's points.
#' @param pipe_slack Children's summed cross-section may exceed the
#'   parent's by this factor; 0 disables.
#' @param spacing_scale,power_above_spacing,radius_power,sensor_noise How the
#'   fit follows the cloud's own point spacing: band and shell scaling, the
#'   spacing past which a power mean of axis distances replaces the circle,
#'   that power (0 for circles), and the scanner's range noise (m).
#' @param cluster_eps Clustering radius within a shell (m); 0 uses graph
#'   components.
#' @param centre_fit_points Clusters with at least this many points get a
#'   circle-fitted centre.
#' @param radius_smooth_steps Smoothing passes over radii along each axis.
#' @param butt_swell Unmeasured stem nodes below the lowest fit may exceed
#'   it by this factor.
#' @param butt_vertical_run,butt_max_lean_deg Replace the stem below the
#'   first run of this many cylinders within `butt_max_lean_deg` of vertical
#'   by a vertical stump; 0 disables.
#' @param chain_max_d Skeleton chaining radius (m).
#' @param fourier_min_radius Sections at least this wide (m) and well covered
#'   use an equivalent-area Fourier contour; 0 disables.
#' @return A `sylva_qsm`.
#' @export
build_qsm <- function(cloud, base_xy = NULL, k = 15, max_edge = 1.0, bin_length = 0.1, min_points = 1,
                      ransac_threshold = 0.02, max_radius = 1.0, taper_limit = 1.1, max_rmse = 0.03,
                      smooth_steps = 10, apex_radius = 0.0025, min_arc_deg = 90.0, min_inlier_fraction = 0.05,
                      prune_points = 5, fit_min_points = 50, crop_length = 0.0, butt_height = 0.6,
                      relative_tolerance = 0.08, base_radius = 0.0, allometry_tolerance = 0.3,
                      buttress_equivalent_area = TRUE, buttress_max_inlier_fraction = 0.3, pipe_slack = 1.2,
                      branch_min_inlier_fraction = 0.3, spacing_scale = 1.5, radius_power = 0.0,
                      power_above_spacing = 0.025, sensor_noise = 0.02, cluster_eps = 0.1,
                      centre_fit_points = 100, radius_smooth_steps = 15, butt_swell = 1.1, butt_vertical_run = 4,
                      butt_max_lean_deg = 50.0, chain_max_d = 0.1, fourier_min_radius = 0.15) {
  params <- mget(qsm_setting_names())
  new_qsm(core_build_qsm(xyz_of(cloud), base_arg(base_xy), lapply(params, as.double)))
}

qsm_setting_names <- function() setdiff(names(formals(build_qsm)), c("cloud", "base_xy"))

# Every build_qsm() setting, its default unless `params` sets it.
qsm_settings <- function(params) {
  keys <- qsm_setting_names()
  unknown <- setdiff(names(params), keys)
  if (length(unknown)) stop(sprintf("build_qsm() has no setting '%s'", unknown[1]), call. = FALSE)
  out <- lapply(formals(build_qsm)[keys], eval)
  out[names(params)] <- params
  lapply(out, as.double)
}

#' Wood points of one tree
#'
#' Leaf / wood separation tuned to give a QSM what it needs (every stem and
#' branch surface), returning the wood thinned to `voxel_size`. Local
#' anisotropy marks bark and branch surfaces; paths from the base over a
#' kNN graph recover the trunk that anisotropy misses (Vicari et al. 2019).
#' See the Python `sylva.qsm.wood_points` for the settings; the defaults are
#' the same.
#'
#' @param cloud One segmented tree (a `sylva_cloud`).
#' @param k Neighbours for the anisotropy features.
#' @param threshold,medium_threshold High- and medium-likelihood anisotropy
#'   cut-offs (0-1).
#' @param voxel_size Thin the input to this spacing (m) first; `NULL` or 0
#'   keeps every point.
#' @param scale_radius Second, wider feature scale (m); 0 disables.
#' @param passage,min_passage,target_res Path-passage wood: enable, paths a
#'   point must carry, and target cell size (m).
#' @param graph_k,max_edge,base_height kNN graph neighbours, longest edge
#'   (m), and height (m) of the base region the paths start from.
#' @param assign_dist,assign_scale Reach (m) around passage points, and its
#'   growth with the share of the tree a point carries.
#' @param component_res,component_min Connected-component cell size (m) and
#'   minimum size.
#' @param sor_k,sor_std Statistical outlier removal for the medium-likelihood
#'   points.
#' @param dilate_dist Final dilation (m).
#' @param method `"passage"`, or `"gbs"` for the graph-based labeller of
#'   `classify_leaf_wood()`.
#' @return The wood points, a `sylva_cloud` with attributes.
#' @export
wood_points <- function(cloud, k = 20, threshold = 0.85, voxel_size = 0.02, medium_threshold = 0.75,
                        scale_radius = 0.0, passage = TRUE, min_passage = 3, target_res = 0.2, graph_k = 10,
                        max_edge = 1.0, base_height = 0.25, assign_dist = 0.05, assign_scale = 0.0,
                        component_res = 0.05, component_min = 200, sor_k = 50, sor_std = 1.0, dilate_dist = 0.03,
                        method = "passage") {
  if (!inherits(cloud, "sylva_cloud")) cloud <- point_cloud(cloud)
  thin <- if (length(voxel_size) && voxel_size != 0) voxel_downsample(cloud, voxel_size) else cloud
  if (identical(method, "gbs")) return(thin[classify_leaf_wood(thin, voxel_size = 0, method = "gbs")])
  opts <- list(k = k, threshold = threshold, medium_threshold = medium_threshold, scale_radius = scale_radius,
               passage = passage, min_passage = min_passage, target_res = target_res, graph_k = graph_k,
               max_edge = max_edge, base_height = base_height, assign_dist = assign_dist,
               assign_scale = assign_scale, component_res = component_res, component_min = component_min,
               sor_k = sor_k, sor_std = sor_std, dilate_dist = dilate_dist)
  thin[core_wood_mask(xyz_of(thin), lapply(opts, as.double))]
}

# ------------------------------------------------------------ buttresses

as_buttress <- function(d) {
  d$faces <- as_faces(d$faces)
  structure(d[c("vertices", "faces", "volume", "top", "top_z", "heights", "areas", "solidities", "open")],
            class = "sylva_buttress")
}

#' An irregular stem base as a closed mesh
#'
#' `buttress_mesh()` rebuilds a buttressed or otherwise irregular stem base
#' volumetrically, so any shape works: the tree's points are cut into
#' slices and rasterised, a morphological closing bridges gaps in the bark,
#' flood filling gives the solid cross-section, slices are built from the
#' top down (each containing the one above, widening no faster than
#' `max_flare`), the buttress ends where the section turns convex, and the
#' stacked sections become a watertight surface (surface nets, Gibson 1998,
#' smoothed as in Taubin 1995). `buttress()` wraps a mesh made elsewhere.
#'
#' A buttress is a list: `vertices`, `faces` (0-based), `volume` below the
#' top (m³), `top` (m above ground) and `top_z` (absolute), and per slice
#' `heights`, `areas`, `solidities` and `open`.
#'
#' @param cloud One tree's points (or the plot's) with height above ground.
#' @param base_xy Stem centre `c(x, y)`.
#' @param ground_z Terrain elevation at the stem; the median `z - height` of
#'   the points within 1 m if `NULL`.
#' @param height_attr Attribute holding height above ground.
#' @param resolution Raster cell (m).
#' @param slice_height Slice thickness (m).
#' @param close_radius Gaps in the outline up to twice this wide are bridged (m).
#' @param max_radius Horizontal reach from the stem centre (m).
#' @param max_height Highest possible top (m above ground).
#' @param top Buttress top (m above ground); found from the solidity if `NULL`.
#' @param solidity Solidity at which a section counts as a round stem.
#' @param max_flare How fast a section may widen going down (m out per m
#'   down); 0 lifts the limit.
#' @param smooth Taubin smoothing passes over the mesh.
#' @param vertices,faces,volume,top_z,heights,areas,solidities,open The fields.
#' @return A `sylva_buttress`; empty (no faces, zero volume) if too few
#'   points are near the stem.
#' @export
buttress_mesh <- function(cloud, base_xy, ground_z = NULL, height_attr = "height", resolution = 0.02,
                          slice_height = 0.05, close_radius = 0.08, max_radius = 4.0, max_height = 6.0, top = NULL,
                          solidity = 0.9, max_flare = 1.0, smooth = 10) {
  as_buttress(core_qsm_buttress_mesh(xyz_of(cloud), cloud_heights(cloud, height_attr), as.double(base_xy[1]),
                                     as.double(base_xy[2]), if (is.null(ground_z)) NULL else as.double(ground_z),
                                     as.double(resolution), as.double(slice_height), as.double(close_radius),
                                     as.double(max_radius), as.double(max_height),
                                     if (is.null(top)) NULL else as.double(top), as.double(solidity),
                                     as.double(max_flare), as.double(smooth)))
}

#' @rdname buttress_mesh
#' @export
buttress <- function(vertices, faces, volume, top, top_z, heights = numeric(), areas = numeric(),
                     solidities = numeric(), open = logical()) {
  as_buttress(list(vertices = matrix(as.double(vertices), ncol = 3), faces = matrix(as.double(faces), ncol = 3),
                   volume = as.double(volume), top = as.double(top), top_z = as.double(top_z),
                   heights = as.double(heights), areas = as.double(areas), solidities = as.double(solidities),
                   open = as.logical(open)))
}

#' @export
print.sylva_buttress <- function(x, ...) {
  cat(sprintf("<sylva_buttress> %d faces, %.4g m3 below %.3g m\n", nrow(unclass(x)$faces), unclass(x)$volume,
              unclass(x)$top))
  invisible(x)
}

#' Wood volume
#'
#' Of a model (its cylinders), of a buttress joined to a model (the
#' buttress plus the model's volume above the buttress top), or of a plot
#' (every tree, buttresses included).
#'
#' @param x A `sylva_qsm`, `sylva_buttress` or `sylva_plot_qsms`.
#' @param model For a buttress, the tree's `sylva_qsm`.
#' @param ... Unused.
#' @return Volume (m³).
#' @export
total_volume <- function(x, ...) UseMethod("total_volume")

#' @rdname total_volume
#' @export
total_volume.sylva_qsm <- function(x, ...) x$total_volume

#' @rdname total_volume
#' @export
total_volume.sylva_buttress <- function(x, model, ...) unclass(x)$volume + volume_above(model, unclass(x)$top_z)

tree_mesh_of <- function(d) {
  d$faces <- as_faces(d$faces)
  d$volume <- d$buttress_volume + d$wood_volume
  structure(d, class = "sylva_tree_mesh")
}

#' Join a buttress to a QSM as one mesh of the whole stem
#'
#' The buttress replaces the cylinders below its top: the model is cut at
#' `top_z` (`above()`) and both surfaces go into one mesh, so nothing is
#' counted twice and the volume is the one `total_volume()` reports. The
#' parts stay watertight and separately labelled rather than being welded
#' into one shell. `overlap` cuts the wood that much lower, so a leaning
#' stem meets the buttress inside it; it changes no volume.
#'
#' @param buttress A `sylva_buttress`.
#' @param model The tree's `sylva_qsm`, in the same frame.
#' @param sides Facets around each cylinder.
#' @param contiguous One continuous tube per branch (see `mesh()`).
#' @param overlap How far below `top_z` (m) the wood is cut.
#' @return A `sylva_tree_mesh`: a list with `vertices`, `faces` (0-based),
#'   `part` per face (0 buttress, 1 wood), `buttress_volume`, `wood_volume`,
#'   `volume` (their sum), `top_z`, `offset` (m between the middles of base
#'   and wood at the join) and `overhang` (share of the wood's cross-section
#'   at the join outside the base; much above zero means the two disagree
#'   about where the stem is).
#' @export
fuse <- function(buttress, model, sides = 12, contiguous = TRUE, overlap = 0.1) {
  b <- unclass(buttress)
  tree_mesh_of(core_qsm_fuse(b$vertices, as_faces(b$faces), as.double(b$volume), as.double(b$top_z),
                             qsm_rows(model), as.double(sides), isTRUE(contiguous), as.double(overlap)))
}

#' @export
print.sylva_tree_mesh <- function(x, ...) {
  cat(sprintf("<sylva_tree_mesh> %d faces, %.4g m3\n", nrow(unclass(x)$faces), unclass(x)$volume))
  invisible(x)
}

#' @rdname to_csv
#' @export
to_obj.sylva_buttress <- function(x, path, ...) {
  b <- unclass(x)
  invisible(core_write_obj(path.expand(path), list(list(vertices = b$vertices, faces = as_faces(b$faces))), "buttress"))
}

#' @rdname to_csv
#' @export
to_ply.sylva_buttress <- function(x, path, ...) {
  b <- unclass(x)
  invisible(core_write_ply_mesh(path.expand(path), b$vertices, as_faces(b$faces), NULL))
}

#' @rdname to_csv
#' @export
to_obj.sylva_tree_mesh <- function(x, path, ...) {
  m <- unclass(x)
  invisible(core_tree_mesh_write_obj(path.expand(path), m$vertices, as_faces(m$faces), as.double(m$part)))
}

#' @rdname to_csv
#' @export
to_ply.sylva_tree_mesh <- function(x, path, color = NULL, ...) {
  m <- unclass(x)
  invisible(core_tree_mesh_write_ply(path.expand(path), m$vertices, as_faces(m$faces), as.double(m$part),
                                     rgb_arg(color)))
}

# ------------------------------------------------------------ plots

#' A QSM for every tree of a segmented plot
#'
#' The loop that `build_qsm()` needs around it: each tree's points are taken
#' from `labels`, thinned, put through the wood filter and fitted, and a tree
#' that cannot be fitted is recorded rather than failing the plot.
#' `plot_qsms()` puts a plot together from models made elsewhere.
#'
#' A plot is a list: `models` and `buttresses` (lists named by tree id),
#' `skipped` (why a tree was not modelled, named by tree id), and `points`
#' and `heights` (per modelled tree). `length()` counts the models,
#' `volume(plot, tree_id)` is one tree's wood volume (buttress included
#' where there is one), `total_volume()` the plot's, and `as.data.frame()`
#' the table of the Python `PlotQSMs.table()`: `tree_id`, `points`,
#' `volume_m3`, `dbh_m`, `height_m`, `n_cylinders`, `measured_volume`,
#' `measured_length` (the share fitted to points rather than taken from the
#' priors), `buttress_m3` and `buttress_top_m` (`NA` without a buttress).
#' A model whose length was hardly fitted to points at all (median share
#' under 0.1) raises a warning: the cloud is probably too sparse for the
#' shell width, and the radii then come from the priors and run large.
#'
#' @param cloud The whole plot, height-normalised (needed for `buttress`).
#' @param labels Tree id per point, as `segment_trees()` returns; anything
#'   below 0 is not part of a tree.
#' @param stems The detected trees (a data frame with `tree_id`, `x`, `y`),
#'   used for the stem centre each model is built around. Without them the
#'   centre is the middle of the tree's own points between 0.5 and 1.5 m.
#' @param voxel_size Thin each tree to this spacing first (m); 0 keeps
#'   every point.
#' @param wood Run `wood_points()` on each tree first.
#' @param buttress Look for a buttress on each tree (`detect_buttress()`) and
#'   mesh it. Needs `height_attr` on the cloud.
#' @param min_points Trees with fewer points than this are skipped.
#' @param height_attr Attribute holding height above ground.
#' @param ... Passed to `build_qsm()`.
#' @param models,buttresses,skipped Lists named by tree id.
#' @param points,heights Named numeric vectors (tree id names).
#' @return A `sylva_plot_qsms`.
#' @export
build_plot <- function(cloud, labels, stems = NULL, voxel_size = 0.01, wood = TRUE, buttress = FALSE,
                       min_points = 2000, height_attr = "height", ...) {
  params <- list(...)
  if (length(labels) != length(cloud)) stop("labels must have one value per point", call. = FALSE)
  settings <- qsm_settings(params)
  h <- attrs_of(cloud)[[height_attr]]
  stems <- if (is.null(stems)) data.frame(tree_id = numeric(), x = numeric(), y = numeric()) else as.data.frame(stems)
  r <- core_qsm_build_plot(xyz_of(cloud), as.double(labels), if (is.null(h)) NULL else as.double(h),
                           as.double(stems$tree_id), as.double(stems$x), as.double(stems$y), as.double(voxel_size),
                           isTRUE(wood), isTRUE(buttress), as.double(min_points), settings)
  named <- function(v, ids) stats::setNames(v, as.character(ids))
  out <- plot_qsms(named(lapply(r$models, new_qsm), r$model_ids), named(lapply(r$buttresses, as_buttress), r$buttress_ids),
                   named(as.list(r$skipped), r$skipped_ids), points = named(r$points, r$point_ids),
                   heights = named(r$heights, r$height_ids))
  share <- r$median_measured_length
  if (!is.null(share) && !is.na(share) && share < 0.1) {
    bin <- if (is.null(params$bin_length)) 0.1 else params$bin_length
    warning(sprintf(paste0("only %.0f%% of the median model's length was fitted to points: the cloud may be too sparse ",
                           "for bin_length=%s m. Radii then come from the priors and run large; check ",
                           "measured_length in the table."), 100 * share, format(bin)), call. = FALSE)
  }
  out
}

#' @rdname build_plot
#' @export
plot_qsms <- function(models, buttresses = list(), skipped = list(), points = numeric(), heights = numeric()) {
  structure(list(models = models, buttresses = buttresses, skipped = skipped, points = points, heights = heights),
            class = "sylva_plot_qsms")
}

#' @export
length.sylva_plot_qsms <- function(x) length(unclass(x)$models)

#' @export
print.sylva_plot_qsms <- function(x, ...) {
  p <- unclass(x)
  cat(sprintf("<sylva_plot_qsms> %d models, %d buttresses, %d skipped\n", length(p$models), length(p$buttresses),
              length(p$skipped)))
  invisible(x)
}

# The plot as the core reads it: one list per model, in the models' order.
plot_entries <- function(x, ids = NULL) {
  p <- unclass(x)
  ids <- if (is.null(ids)) names(p$models) else as.character(ids)
  lapply(ids, function(t) {
    m <- p$models[[t]]
    if (is.null(m)) stop(sprintf("no model for tree %s", t), call. = FALSE)
    b <- p$buttresses[[t]]
    if (!is.null(b)) {
      b <- unclass(b)
      b <- list(vertices = b$vertices, faces = as_faces(b$faces), volume = b$volume, top = b$top, top_z = b$top_z)
    }
    list(tree_id = as.double(t), cylinders = qsm_rows(m),
         points = if (t %in% names(p$points)) as.double(p$points[[t]]) else NULL,
         height = if (t %in% names(p$heights)) as.double(p$heights[[t]]) else NULL, buttress = b)
  })
}

#' @rdname build_plot
#' @param plot A `sylva_plot_qsms`.
#' @param tree_id Which tree.
#' @export
volume <- function(plot, tree_id) core_plot_volumes(plot_entries(plot, tree_id))

#' @rdname total_volume
#' @export
total_volume.sylva_plot_qsms <- function(x, ...) core_plot_total_volume(plot_entries(x))

#' @export
as.data.frame.sylva_plot_qsms <- function(x, ...) {
  t <- core_plot_table(plot_entries(x))
  df <- data.frame(tree_id = as.integer(t$tree_id), points = as.integer(ifelse(t$has_points, t$points, NA)),
                   volume_m3 = t$volume_m3, dbh_m = t$dbh_m, height_m = t$height_m,
                   n_cylinders = as.integer(t$n_cylinders), measured_volume = t$measured_volume,
                   measured_length = t$measured_length, buttress_m3 = ifelse(t$has_buttress, t$buttress_m3, NA),
                   buttress_top_m = ifelse(t$has_buttress, t$buttress_top_m, NA))
  df
}

#' @rdname to_csv
#' @export
to_csv.sylva_plot_qsms <- function(x, path, ...) invisible(core_plot_write_csv(path.expand(path), plot_entries(x)))

#' Write a plot's meshes and cylinders
#'
#' `write_meshes()`: a surface mesh per tree, as `<prefix><tree_id>.<fmt>`.
#' A tree with a buttress is written joined to it (`fuse()`), so the
#' flanged base and the cylinders above it come out as one file; every
#' other tree is its cylinder mesh. `write_cylinders()`: one cylinder CSV
#' per tree, as `<prefix><tree_id>.csv`.
#'
#' @param plot A `sylva_plot_qsms`.
#' @param directory Created if it does not exist.
#' @param fmt `"ply"` (binary, with face colours) or `"obj"` (text, the
#'   buttress and the wood as named objects).
#' @param sides Facets around each cylinder.
#' @param contiguous One continuous tube per branch (see `mesh()`).
#' @param prefix File name stem.
#' @return `write_meshes()`: the files written, in tree order.
#' @export
write_meshes <- function(plot, directory, fmt = "ply", sides = 12, contiguous = TRUE, prefix = "tree") {
  core_plot_write_meshes(path.expand(directory), plot_entries(plot), fmt, as.double(sides), isTRUE(contiguous),
                         prefix)
}

#' @rdname write_meshes
#' @export
write_cylinders <- function(plot, directory, prefix = "tree") {
  invisible(core_plot_write_cylinders(path.expand(directory), plot_entries(plot), prefix))
}
