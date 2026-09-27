# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

#' Thin a cloud to at most one point per voxel
#'
#' Evens out the density falloff with range from each scanner. The grid
#' starts at the cloud's minimum corner.
#'
#' @param cloud A `sylva_cloud`.
#' @param voxel_size Voxel edge length (m).
#' @param method `"first"` keeps one original point per voxel (the first in
#'   file order) with its attributes, in their original order;
#'   `"centroid"` returns the mean of the points in each voxel and drops
#'   attributes.
#' @return A `sylva_cloud`.
#' @export
voxel_downsample <- function(cloud, voxel_size, method = c("first", "centroid")) {
  method <- match.arg(method)
  if (method == "first") return(cloud[core_voxel_downsample_indices(xyz_of(cloud), as.double(voxel_size))])
  new_cloud(core_voxel_centroids(xyz_of(cloud), as.double(voxel_size)))
}

#' Random subsample without replacement
#'
#' @param cloud A `sylva_cloud`.
#' @param n Number of points to keep.
#' @param fraction Fraction of points to keep (0-1), rounded half to even.
#'   Give exactly one of `n` and `fraction`.
#' @param seed Random seed; the same seed gives the same subsample, and the
#'   same one as the Python package.
#' @return The selected points in their original order.
#' @export
random_subsample <- function(cloud, n = NULL, fraction = NULL, seed = 0) {
  if (is.null(n) == is.null(fraction)) stop("give exactly one of n or fraction", call. = FALSE)
  if (is.null(n)) n <- round(length(cloud) * fraction)
  cloud[core_random_indices(as.double(length(cloud)), as.double(n), as.double(seed))]
}

#' Thin to an even spacing
#'
#' An approximate Poisson-disk sample: no two kept points are closer than
#' `distance`. Unlike `voxel_downsample()` the result has no grid pattern,
#' which suits normals, curvature and segmentation.
#'
#' @param cloud A `sylva_cloud`.
#' @param distance Minimum spacing (m).
#' @return Original points, with attributes, in their original order.
#' @export
min_distance_subsample <- function(cloud, distance) {
  cloud[core_min_distance_indices(xyz_of(cloud), as.double(distance))]
}

open_bounds <- function(v) {
  v <- lapply(as.list(v), function(x) if (is.null(x)) NaN else as.double(x))
  vapply(v, function(x) if (is.na(x)) NaN else x, 0)
}

#' Crop a cloud
#'
#' `crop_box()`: points inside an axis-aligned box, bounds inclusive; `NA`
#' (or `NULL` in a list) leaves that side open, e.g.
#' `crop_box(c, c(NA, NA, 0.5), c(NA, NA, NA))`. `crop_cylinder()`: points
#' inside a vertical cylinder such as a circular plot, with limits on z (not
#' on height above ground). `range_filter()`: points by 3-D distance from a
#' scanner, limits inclusive.
#'
#' @param cloud A `sylva_cloud`.
#' @param min_xyz,max_xyz Three lower and three upper bounds.
#' @param center_xy Plot centre `c(x, y)`.
#' @param radius Horizontal radius (m).
#' @param zmin,zmax Vertical limits.
#' @param origin Scanner position, in the cloud's frame.
#' @param min_range,max_range Distance limits (m).
#' @return A `sylva_cloud`.
#' @export
crop_box <- function(cloud, min_xyz, max_xyz) {
  cloud[core_crop_box_mask(xyz_of(cloud), open_bounds(min_xyz), open_bounds(max_xyz))]
}

#' @rdname crop_box
#' @export
crop_cylinder <- function(cloud, center_xy, radius, zmin = -Inf, zmax = Inf) {
  c0 <- as.double(center_xy)
  cloud[core_crop_cylinder_mask(xyz_of(cloud), c0[1], c0[2], as.double(radius), as.double(zmin), as.double(zmax))]
}

#' @rdname crop_box
#' @export
range_filter <- function(cloud, origin = c(0, 0, 0), min_range = 0, max_range = Inf) {
  cloud[core_range_mask(xyz_of(cloud), as.double(origin), as.double(min_range), as.double(max_range))]
}

#' Remove outlying points
#'
#' `statistical_outlier_removal()` (Rusu et al. 2008): for each point the
#' mean distance to its `k` nearest neighbours; points where this exceeds
#' `mean + std_ratio * sd` over the whole cloud are dropped. The threshold
#' is global, so run it per scan or on a thinned cloud when density varies
#' strongly with range. `radius_outlier_removal()`: points with fewer than
#' `min_neighbors` other points within `radius` are dropped.
#'
#' @param cloud A `sylva_cloud`.
#' @param k Neighbours per point.
#' @param std_ratio Threshold in standard deviations; lower removes more.
#' @param radius Search radius (m).
#' @param min_neighbors Fewest neighbours a kept point has.
#' @param return_mask Return the logical keep-mask instead of the cloud.
#' @return The kept points, or the mask (`TRUE` = keep).
#' @export
statistical_outlier_removal <- function(cloud, k = 8, std_ratio = 2, return_mask = FALSE) {
  mask <- core_statistical_outlier_mask(xyz_of(cloud), as.integer(k), as.double(std_ratio))
  if (return_mask) mask else cloud[mask]
}

#' @rdname statistical_outlier_removal
#' @export
radius_outlier_removal <- function(cloud, radius, min_neighbors = 4, return_mask = FALSE) {
  mask <- core_radius_outlier_mask(xyz_of(cloud), as.double(radius), as.integer(min_neighbors))
  if (return_mask) mask else cloud[mask]
}

#' Local geometry
#'
#' `estimate_normals()`: unit normals from local principal components (the
#' eigenvector of the smallest eigenvalue of the `k` neighbours'
#' covariance); the sign is arbitrary. `planarity_linearity()`: with the
#' eigenvalues sorted `l1 <= l2 <= l3`, planarity `(l2 - l1) / l3` and
#' linearity `(l3 - l2) / l3` (Weinmann et al. 2015), both in 0-1; stems and
#' branches are linear, leaves and ground planar.
#'
#' @param cloud A `sylva_cloud` (or an `n x 3` matrix).
#' @param k Neighbours per point.
#' @return `estimate_normals()`: an `n x 3` matrix. `planarity_linearity()`:
#'   `list(planarity, linearity)`, one value per point each.
#' @export
estimate_normals <- function(cloud, k = 12) core_estimate_normals(xyz_of(cloud), as.integer(k))

#' @rdname estimate_normals
#' @export
planarity_linearity <- function(cloud, k = 20) core_planarity_linearity(xyz_of(cloud), as.integer(k))

#' Euclidean clusters
#'
#' Two points are connected if they are within `radius`; clusters are the
#' connected components.
#'
#' @param xyz An `n x 3` matrix or a `sylva_cloud`.
#' @param radius Linking distance (m).
#' @param min_points Clusters smaller than this are labelled -1.
#' @return A label per point: 0 is the largest cluster, 1 the next, ...; -1
#'   for points in clusters that are too small (as in the Python package).
#' @export
euclidean_clusters <- function(xyz, radius, min_points = 1) {
  core_euclidean_clusters(xyz_of(xyz), as.double(radius), as.integer(min_points))
}

#' k nearest neighbours
#'
#' @param xyz Points to search: an `n x 3` matrix or a `sylva_cloud`.
#' @param queries Query points (`m x 3`). A query that is also in `xyz` finds
#'   itself at distance 0.
#' @param k Neighbours per query.
#' @return `list(distances, indices)`, two `m x k` matrices, nearest first;
#'   `indices` are row numbers of `xyz` (1-based), `NA` (distance `Inf`)
#'   where `xyz` has fewer than `k` points.
#' @export
knn <- function(xyz, queries, k) {
  r <- core_knn(xyz_of(xyz), xyz_of(queries), as.integer(k))
  r$indices[is.nan(r$indices)] <- NA
  r
}
