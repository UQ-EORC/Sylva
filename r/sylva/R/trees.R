# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# Trees are data frames with one row per tree and the Python package's Tree
# fields as columns; further columns are carried along like Tree.extra.

tree_columns <- c("tree_id", "x", "y", "dbh", "height", "n_points", "inlier_fraction", "n_slices", "rmse",
                  "lean_deg", "quality")

as_trees <- function(cols) {
  df <- as.data.frame(cols[tree_columns], stringsAsFactors = FALSE)
  for (k in c("tree_id", "n_points", "n_slices")) df[[k]] <- as.integer(df[[k]])
  df
}

trees_arg <- function(trees) {
  trees <- as.data.frame(trees)
  cols <- intersect(tree_columns, names(trees))
  lapply(trees[cols], as.double)
}

cloud_heights <- function(cloud, height_attr) {
  c <- unclass(cloud)
  as.double(if (!is.null(c$attrs[[height_attr]])) c$attrs[[height_attr]] else c$xyz[, 3])
}

cloud_xyz <- function(points) {
  if (inherits(points, "sylva_cloud")) unclass(points)$xyz else as_xyz(points)
}

#' Trees
#'
#' A table of detected trees, one row per tree, as the Python package's
#' `Tree`: `tree_id` (1..n, matching the labels of `segment_trees()`), stem
#' centre `x`, `y` at breast height, `dbh` (m), `height` (m; `NaN` until
#' `tree_heights()`), `n_points`, `inlier_fraction` (share of the stem
#' circumference observed), `n_slices` (layers the stem was found in), `rmse`
#' of the circle fits (m), `lean_deg` and `quality` (0-1 confidence).
#'
#' @param tree_id,x,y,dbh,height,n_points,inlier_fraction,n_slices,rmse,lean_deg,quality
#'   The columns; vectors of one length (or 1).
#' @param ... Further columns, carried along like the Python `extra`.
#' @return A data frame.
#' @export
tree <- function(tree_id, x, y, dbh = NaN, height = NaN, n_points = 0L, inlier_fraction = NaN, n_slices = 0L,
                 rmse = NaN, lean_deg = NaN, quality = NaN, ...) {
  df <- data.frame(tree_id = as.integer(tree_id), x = as.double(x), y = as.double(y), dbh = as.double(dbh),
                   height = as.double(height), n_points = as.integer(n_points),
                   inlier_fraction = as.double(inlier_fraction), n_slices = as.integer(n_slices),
                   rmse = as.double(rmse), lean_deg = as.double(lean_deg), quality = as.double(quality))
  extra <- list(...)
  if (length(extra)) df <- cbind(df, as.data.frame(extra, stringsAsFactors = FALSE))
  df
}

# -------------------------------------------------------------------- circles

#' Circle fits to 2-D points
#'
#' `fit_circle()`: least squares, an algebraic fit (Kasa 1976) refined by
#' Levenberg-Marquardt on the geometric distance; not robust to outliers.
#' `fit_circle_ransac()`: RANSAC (Fischler & Bolles 1981) on circles through
#' three points, then a least-squares refit on the inliers.
#'
#' @param xy An `n x 2` matrix of points.
#' @param threshold Inlier distance from the circle (m).
#' @param iterations RANSAC trials.
#' @param min_radius,max_radius Radius limits (m).
#' @param seed Random seed.
#' @return `fit_circle()`: a list with `cx`, `cy`, `r` and `rmse`.
#'   `fit_circle_ransac()`: `cx`, `cy`, `r` and `inliers` (logical, one per
#'   point); an error if no circle within the limits is found.
#' @export
fit_circle <- function(xy) {
  f <- core_fit_circle(as_xyz(xy))
  list(cx = f[1], cy = f[2], r = f[3], rmse = f[4])
}

#' @rdname fit_circle
#' @export
fit_circle_ransac <- function(xy, threshold = 0.01, iterations = 200, min_radius = 0.02, max_radius = 1.5,
                              seed = 0) {
  core_fit_circle_ransac(as_xyz(xy), as.double(threshold), as.double(iterations), as.double(min_radius),
                         as.double(max_radius), as.double(seed))
}

#' Area of the convex hull of 2-D points
#'
#' @param xy An `n x 2` matrix; non-finite rows are ignored.
#' @return Area in squared input units (`NaN` for fewer than 3 distinct points).
#' @export
convex_hull_area <- function(xy) core_convex_hull_area(as_xyz(xy))

# ---------------------------------------------------------------- detection

#' Detect stems in a height-normalised cloud
#'
#' The 1-5 m band is cut into 0.3 m layers every 0.25 m; each layer is
#' clustered in 2-D and circles are fitted by RANSAC with an
#' angular-coverage check; circles are linked across layers into chains
#' that must span `min_slices` (3) layers and lean under `max_lean_deg`
#' (25). DBH is read from a linear taper at `reference_height` (1.3 m).
#' Detection favours recall; follow with `merge_branches()`,
#' `segment_trees()`, `tree_heights()` and `prune_trees()`.
#'
#' @param cloud A `sylva_cloud` with height above ground.
#' @param height_attr Attribute holding heights; z is used if absent.
#' @param ... Detection settings under the Python names, e.g. `slice_min`,
#'   `slice_max`, `min_radius`, `max_radius`, `min_coverage`, `min_arc_deg`,
#'   `min_slices`, `max_lean_deg`, `prefilter`, `seed`.
#' @return A tree table (see `tree()`) sorted by quality with ids 1..n.
#' @export
detect_stems <- function(cloud, height_attr = "height", ...) {
  as_trees(core_detect_stems(unclass(cloud)$xyz, cloud_heights(cloud, height_attr), list(...)))
}

#' Stem diameter at a series of heights (a taper curve)
#'
#' @param cloud A height-normalised `sylva_cloud`.
#' @param center_xy Stem position, e.g. `c(trees$x[1], trees$y[1])`.
#' @param height_attr Attribute holding heights.
#' @param heights Heights to measure at (m); 0.5 to 10 m every 0.5 m if `NULL`.
#' @param slice_thickness Slice thickness (m).
#' @param search_radius Only points within this horizontal distance (m).
#' @return A matrix with columns `height` and `diameter` (m); `NaN` where a
#'   slice has fewer than 10 points or no circle fits.
#' @export
dbh_profile <- function(cloud, center_xy, height_attr = "height", heights = NULL, slice_thickness = 0.1,
                        search_radius = 0.75) {
  if (is.null(heights)) heights <- 0.5 * (1:20)
  heights <- as.double(heights)
  d <- core_dbh_profile(unclass(cloud)$xyz, cloud_heights(cloud, height_attr), as.double(center_xy[1]),
                        as.double(center_xy[2]), heights, as.double(slice_thickness), as.double(search_radius))
  cbind(height = heights, diameter = d)
}

# ------------------------------------------------------------- segmentation

graph_settings <- function(k, max_edge, voxel_size, seed_height, seed_radius, seed_ring, power, angle_penalty,
                           gravity = 0, cut_above_ground, height_prior = FALSE, height_prior_radius = 1.5,
                           height_prior_power = 1, low_height = 0.5, low_radius = 1, wood_costs = FALSE,
                           wood_k = 20, wood_threshold = 0.9, understorey_height = 10, understorey_band = 0.5) {
  list(k = as.double(k), max_edge = as.double(max_edge), voxel_size = as.double(voxel_size),
       seed_height = as.double(seed_height), seed_radius = as.double(seed_radius), seed_ring = isTRUE(seed_ring),
       power = as.double(power), angle_penalty = isTRUE(angle_penalty), gravity = as.double(gravity),
       cut_above_ground = as.double(cut_above_ground), height_prior = isTRUE(height_prior),
       height_prior_radius = as.double(height_prior_radius), height_prior_power = as.double(height_prior_power),
       low_height = as.double(low_height), low_radius = as.double(low_radius), wood_costs = isTRUE(wood_costs),
       wood_k = as.double(wood_k), wood_threshold = as.double(wood_threshold),
       understorey_height = as.double(understorey_height), understorey_band = as.double(understorey_band))
}

#' Assign each point to a stem
#'
#' Least-cost paths through a directed kNN graph (multi-source Dijkstra from
#' stem seeds), after raycloudtools' `rayextract trees` (Devereux et al.
#' 2026). The edge cost is `d ^ power` times an angle penalty, so paths run
#' up through a tree instead of leaking across the ground or understorey;
#' see the Python documentation of `sylva.trees.segment_trees` for every
#' term.
#'
#' @param cloud A height-normalised `sylva_cloud` of the plot.
#' @param trees Stems (a tree table), usually after `merge_branches()`.
#' @param height_attr Attribute holding heights.
#' @param k Neighbours per graph node.
#' @param max_edge Longest graph edge (m).
#' @param voxel_size Graph resolution (m); 0 uses every point.
#' @param seed_height,seed_radius,seed_ring Which graph nodes seed each tree.
#' @param power Exponent on edge length.
#' @param angle_penalty Apply the angle factor.
#' @param gravity Lateral-distance term; 0 disables it.
#' @param cut_above_ground Points below this height (m) are left unassigned.
#' @param height_prior,height_prior_radius,height_prior_power Scale costs by
#'   the inverse tree height.
#' @param low_height,low_radius Near-ground points farther than `low_radius`
#'   from their stem are unassigned.
#' @param wood_costs,wood_k,wood_threshold Wood/leaf edge factors.
#' @param understorey_height,understorey_band Let the understorey compete; 0
#'   disables.
#' @return Integer label per point: the `tree_id` of its tree, or -1.
#' @export
segment_trees <- function(cloud, trees, height_attr = "height", k = 6, max_edge = 1.0, voxel_size = 0.03,
                          seed_height = 1.5, seed_radius = 0.25, seed_ring = TRUE, power = 6.0,
                          angle_penalty = TRUE, gravity = 0.0, cut_above_ground = 0.25, height_prior = TRUE,
                          height_prior_radius = 1.5, height_prior_power = 1.0, low_height = 0.5, low_radius = 1.0,
                          wood_costs = FALSE, wood_k = 20, wood_threshold = 0.9, understorey_height = 10.0,
                          understorey_band = 0.5) {
  g <- graph_settings(k, max_edge, voxel_size, seed_height, seed_radius, seed_ring, power, angle_penalty, gravity,
                      cut_above_ground, height_prior, height_prior_radius, height_prior_power, low_height,
                      low_radius, wood_costs, wood_k, wood_threshold, understorey_height, understorey_band)
  core_segment_trees(unclass(cloud)$xyz, cloud_heights(cloud, height_attr), trees_arg(trees), g)
}

#' Drop candidates that are branches or secondary stems of another
#'
#' Every graph node's least-cost path to the ground is traced (as in
#' raycloudtools); a candidate whose seed routes to the ground through
#' another candidate's trunk (within `max(trunk_scale * radius, trunk_min)`
#' of its axis, below its seed height) is merged into it. Call before
#' `segment_trees()`.
#'
#' @inheritParams segment_trees
#' @param trees Candidates from `detect_stems()`.
#' @param ground_height Graph nodes below this height (m) count as ground.
#' @param trunk_scale,trunk_min Width of the trunk zone, in stem radii and
#'   its minimum (m).
#' @param search_radius Only candidates within this distance (m) are compared.
#' @return A list with the surviving `trees` and `merged_into`, for each
#'   input tree the id it now belongs to.
#' @export
merge_branches <- function(cloud, trees, height_attr = "height", ground_height = 0.5, trunk_scale = 1.5,
                           trunk_min = 0.15, search_radius = 6.0, k = 10, max_edge = 1.0, voxel_size = 0.1,
                           seed_height = 1.5, seed_radius = 0.5, seed_ring = TRUE, power = 3.0,
                           angle_penalty = TRUE, cut_above_ground = 0.25) {
  g <- graph_settings(k, max_edge, voxel_size, seed_height, seed_radius, seed_ring, power, angle_penalty,
                      cut_above_ground = cut_above_ground)
  r <- core_merge_branches(unclass(cloud)$xyz, cloud_heights(cloud, height_attr), trees_arg(trees), g,
                           as.double(ground_height), as.double(trunk_scale), as.double(trunk_min),
                           as.double(search_radius))
  list(trees = as_trees(r$trees), merged_into = as.integer(r$merged_into))
}

#' Set each tree's height and point count from the segmentation
#'
#' @param cloud The cloud that was segmented.
#' @param labels Point labels from `segment_trees()`.
#' @param trees Tree table to update.
#' @param height_attr Attribute holding heights.
#' @param percentile Height percentile of the tree's points; 100 is the top.
#' @return `trees` with `height` (`NaN` without points) and `n_points` set.
#' @export
tree_heights <- function(cloud, labels, trees, height_attr = "height", percentile = 100.0) {
  r <- core_tree_heights(cloud_heights(cloud, height_attr), as.double(labels), trees_arg(trees),
                         as.double(percentile))
  trees <- as.data.frame(trees)
  trees$height <- r$height
  trees$n_points <- as.integer(r$n_points)
  trees
}

#' Drop short candidates and merge duplicates after segmentation
#'
#' Trees lower than `min_height` (`NaN` counts as low) are removed; of any
#' two stems closer than `merge_radius` the one with more points survives
#' and absorbs the other's points. `min_quality_short` also drops candidates
#' supported by fewer than `short_slices` layers whose `quality` is below it.
#'
#' @param trees Tree table with `height` and `n_points` from `tree_heights()`.
#' @param labels Point labels from `segment_trees()`.
#' @param min_height Minimum tree height (m).
#' @param merge_radius Stems closer than this (m) are merged.
#' @param max_dbh Drop stems wider than this (m); `NULL` keeps all.
#' @param min_quality_short Minimum quality of short-chain candidates.
#' @param short_slices Candidates with fewer layers than this count as short.
#' @return A list with `trees` (survivors by decreasing DBH, ids 1..n,
#'   `n_points` recounted, other columns kept) and `labels` (remapped;
#'   points of dropped trees are -1).
#' @export
prune_trees <- function(trees, labels, min_height = 3.0, merge_radius = 0.2, max_dbh = NULL,
                        min_quality_short = 0.0, short_slices = 4) {
  trees <- as.data.frame(trees)
  r <- core_prune_trees(trees_arg(trees), as.double(labels), as.double(min_height), as.double(merge_radius),
                        if (is.null(max_dbh)) NULL else as.double(max_dbh), as.double(min_quality_short),
                        as.double(short_slices))
  out <- trees[r$row, , drop = FALSE]
  new <- as_trees(r$trees)
  for (k in names(new)) out[[k]] <- new[[k]]
  rownames(out) <- NULL
  list(trees = out, labels = r$labels)
}

#' Basal area of a plot (m2/ha)
#'
#' `sum(pi * (dbh / 2)^2) / area * 1e4` over stems with `dbh >= min_dbh`;
#' stems without a DBH (`NaN`) are left out. Restrict `trees` to the plot
#' first so that stems and `area` cover the same ground.
#'
#' @param trees A tree table or a numeric vector of DBHs (m).
#' @param area Plot area (m2), e.g. `pi * radius^2`.
#' @param min_dbh Smallest DBH counted (m).
#' @return Basal area in m2/ha.
#' @export
basal_area <- function(trees, area, min_dbh = 0.0) {
  if (!isTRUE(area > 0)) stop(sprintf("area must be positive, got %s", format(area)))
  dbh <- if (is.data.frame(trees) || is.list(trees)) trees$dbh else trees
  core_basal_area(as.double(dbh), as.double(area), as.double(min_dbh))
}

# ------------------------------------------------------------------- crowns

#' Crown size of segmented trees
#'
#' Crown base is found from a height histogram of the tree's points in
#' 0.5 m bins: the first bin, from a quarter of the tree height upwards,
#' holding at least `crown_base_fraction` of the fullest bin.
#'
#' @param cloud The cloud that was segmented.
#' @param labels Point labels from `segment_trees()`.
#' @param tree_id Tree to measure.
#' @param height_attr Attribute holding heights.
#' @param crown_base_fraction Density threshold for the crown base (0-1).
#' @return `crown_metrics()`: a list with `crown_area` (m2),
#'   `crown_base_height`, `crown_depth` and `crown_diameter` (m); empty if
#'   the tree has fewer than 4 points. `crown_metrics_all()`: the same for
#'   every tree as a data frame with `tree_id`.
#' @export
crown_metrics <- function(cloud, labels, tree_id, height_attr = "height", crown_base_fraction = 0.1) {
  core_crown_metrics(unclass(cloud)$xyz, cloud_heights(cloud, height_attr), as.double(labels), as.double(tree_id),
                     as.double(crown_base_fraction))
}

#' @rdname crown_metrics
#' @export
crown_metrics_all <- function(cloud, labels, height_attr = "height", crown_base_fraction = 0.1) {
  r <- core_crown_metrics_all(unclass(cloud)$xyz, cloud_heights(cloud, height_attr), as.double(labels),
                              as.double(crown_base_fraction))
  r$tree_id <- as.integer(r$tree_id)
  df <- as.data.frame(r)
  df[order(df$tree_id), , drop = FALSE]
}

#' Crown shape from a tree's points
#'
#' The vertical projection (convex hull `projected_area`, equivalent
#' `diameter`, `max_width`), `volume` and `surface` of convex hulls stacked
#' every `slice_height` metres, `base_height` and `top_height`, and the
#' horizontal `offset` of the crown's centroid from `base_xy`, its
#' `offset_direction` (deg, counter-clockwise from +x) and `asymmetry`.
#'
#' @param points A `sylva_cloud` or `n x 3` matrix of one tree.
#' @param base_xy Stem position; the lowest point if `NULL`.
#' @param crown_base Absolute z of the crown base; everything is crown if `NULL`.
#' @param slice_height Thickness (m) of the stacked hull slices.
#' @return A named list.
#' @export
crown_shape <- function(points, base_xy = NULL, crown_base = NULL, slice_height = 0.5) {
  core_crown_shape(cloud_xyz(points), if (is.null(base_xy)) NULL else as.double(base_xy[1:2]),
                   if (is.null(crown_base)) -Inf else as.double(crown_base), as.double(slice_height))
}

# ---------------------------------------------------------------- buttresses

#' Does a stem have buttresses, and how high do they reach?
#'
#' A RANSAC circle is fitted to every `slice_height` slice; on a buttressed
#' stem it explains only a small share of the bark near the ground.
#' Angular bins holding points well outside the stem radius (`1.4 r + 0.1`
#' m) in at least 60 % of the slices below `low` are ridges. A tree is
#' buttressed when the median circle fit below `low` is under
#' `max_circle_fit` and there are at least `min_ridges` ridges.
#'
#' @param cloud One tree's points (or a plot's) with height above ground.
#' @param base_xy Stem centre; estimated from the points 2.5-3.5 m up if `NULL`.
#' @param height_attr Attribute holding height above ground.
#' @param max_radius Horizontal reach from the stem centre (m).
#' @param slice_height Slice thickness (m).
#' @param max_height Highest slice (m).
#' @param low Top of the base zone the decision looks at (m).
#' @param bins Angular bins for the ridges.
#' @param max_circle_fit,min_ridges Decision thresholds.
#' @param bark_only Use only locally planar, near-vertical surface points.
#' @param voxel Thin the points to this spacing first (m); 0 keeps all.
#' @return A list: `buttressed`, `base_circle_fit`, `stem_circle_fit`,
#'   `stem_radius`, `ridges`, `ridge_share`, `spread`, `top` (`NaN` if not
#'   buttressed) and `centre`.
#' @export
detect_buttress <- function(cloud, base_xy = NULL, height_attr = "height", max_radius = 4.0, slice_height = 0.1,
                            max_height = 6.0, low = 1.0, bins = 36, max_circle_fit = 0.55, min_ridges = 2,
                            bark_only = TRUE, voxel = 0.02) {
  core_detect_buttress(unclass(cloud)$xyz, cloud_heights(cloud, height_attr),
                       if (is.null(base_xy)) NULL else as.double(base_xy[1:2]), as.double(max_radius),
                       as.double(slice_height), as.double(max_height), as.double(low), as.double(bins),
                       as.double(max_circle_fit), as.double(min_ridges), isTRUE(bark_only), as.double(voxel))
}
