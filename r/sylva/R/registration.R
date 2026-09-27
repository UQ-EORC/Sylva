# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

#' Transform matrices
#'
#' @param angle_deg Rotation about z in degrees, counter-clockwise seen from above.
#' @param dx,dy,dz Shift (m).
#' @return A `4 x 4` matrix acting on column vectors, for `transform()`.
#' @export
rotation_z <- function(angle_deg) core_rotation_z(as.double(angle_deg))

#' @rdname rotation_z
#' @export
translation <- function(dx, dy, dz) core_translation(as.double(dx), as.double(dy), as.double(dz))

#' Best rigid transform between paired points
#'
#' Kabsch (1976) / Umeyama (1991) without scale, for matched targets
#' (reflectors, tie points).
#'
#' @param source,target `n x 3` corresponding points, `n >= 3` and not collinear.
#' @return A `4 x 4` matrix mapping `source` onto `target` in the least
#'   squares sense.
#' @export
kabsch <- function(source, target) core_kabsch(xyz_of(source), xyz_of(target))

#' Iterative closest point
#'
#' `method = "point"` is point-to-point ICP (Besl and McKay 1992); `"plane"`
#' point-to-plane (Chen and Medioni 1992, linearised as in Low 2004) with
#' PCA normals of the target. `trim` keeps that fraction of closest
#' correspondences per iteration (trimmed ICP for partial overlap,
#' Chetverikov et al. 2002). Thin both clouds to 2-5 cm first.
#'
#' @param source Cloud to move (`sylva_cloud` or `n x 3` matrix).
#' @param target Fixed reference cloud.
#' @param init Starting `4 x 4` transform (e.g. the SOP, or from `kabsch()`);
#'   ICP only converges from a start within about
#'   `max_correspondence_distance`.
#' @param max_correspondence_distance Pairs further apart (m) are ignored.
#' @param max_iterations Iteration limit.
#' @param tolerance Stop when the RMSE changes less than this.
#' @param method `"point"` or `"plane"`.
#' @param trim Fraction of pairs kept (0-1).
#' @param normal_k Neighbours for the PCA normals with `"plane"`.
#' @return `list(transform, info)`: the `4 x 4` matrix mapping `source` onto
#'   `target`, and `list(rmse, iterations, n_correspondences)`.
#' @export
icp <- function(source, target, init = NULL, max_correspondence_distance = 0.5, max_iterations = 50,
                tolerance = 1e-6, method = c("point", "plane"), trim = 1, normal_k = 12) {
  method <- match.arg(method)
  if (!is.null(init)) {
    init <- as.matrix(init)
    storage.mode(init) <- "double"
  }
  core_icp(xyz_of(source), xyz_of(target), init, as.double(max_correspondence_distance), as.integer(max_iterations),
           as.double(tolerance), method, as.double(trim), as.integer(normal_k))
}

#' Put several scans in one frame and merge them
#'
#' @param clouds A list of clouds, one per scan position.
#' @param transforms A list of `4 x 4` matrices, one per cloud (SOPs or ICP
#'   results); `NULL` if the clouds are already registered.
#' @param scan_ids Add a `scan_id` attribute: the scan's 0-based index in
#'   `clouds`, as in the Python package.
#' @return A `sylva_cloud`; only attributes present in every cloud are kept.
#' @export
merge_scans <- function(clouds, transforms = NULL, scan_ids = TRUE) {
  out <- lapply(seq_along(clouds), function(i) {
    c <- clouds[[i]]
    if (!is.null(transforms)) c <- transform(c, transforms[[i]])
    if (scan_ids) c <- with_attrs(c, scan_id = rep(i - 1L, length(c)))
    c
  })
  concatenate(out)
}
