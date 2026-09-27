# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# Coregistration works on 4 x 4 homogeneous transforms (column-vector
# convention, q = T %*% c(p, 1)) and n x 3 point matrices. Twists are
# c(wx, wy, wz, tx, ty, tz), rotation first. Where a Python name is taken by
# another part of the package (or by base R) the function carries a
# `coreg_` prefix, as `sylva.coreg.<name>` does in Python.

as_transform <- function(T) {
  T <- as.matrix(T)
  storage.mode(T) <- "double"
  if (identical(dim(T), c(3L, 4L))) T <- rbind(T, c(0, 0, 0, 1))
  if (!identical(dim(T), c(4L, 4L))) stop("a transform must be a 4 x 4 matrix")
  T
}

as_points <- function(p, what = "points") {
  if (inherits(p, "sylva_cloud")) p <- unclass(p)$xyz
  if (is.data.frame(p)) p <- as.matrix(p[, c("x", "y", "z")])
  if (is.null(dim(p))) p <- matrix(as.double(p), ncol = 3, byrow = TRUE)
  p <- as.matrix(p)
  storage.mode(p) <- "double"
  if (ncol(p) != 3) stop(sprintf("%s must have 3 columns", what))
  p
}

as_rotation <- function(R) {
  R <- as.matrix(R)
  storage.mode(R) <- "double"
  R[1:3, 1:3, drop = FALSE]
}

#' Rigid transforms
#'
#' The SO(3) and SE(3) maps coregistration is built on, as the Python
#' package's `sylva.coreg.transforms`. `so3_exp()` turns a rotation vector
#' into a rotation matrix and `so3_log()` back; `se3_exp()` turns a twist
#' `c(w, t)` into a 4 x 4 transform and `se3_log()` back. `skew()` is the
#' cross-product matrix of a 3-vector, `invert()` the inverse of a rigid
#' transform, `yaw_transform()` a rotation about +z followed by a
#' translation, and `coreg_identity()` the 4 x 4 identity.
#'
#' @param v,w A 3-vector (a rotation vector for `so3_exp()`).
#' @param R A rotation matrix (3 x 3, or the rotation block of a 4 x 4).
#' @param xi A twist `c(wx, wy, wz, tx, ty, tz)`.
#' @param T,A,B 4 x 4 transforms.
#' @param yaw Rotation about +z (radians).
#' @param tx,ty,tz Translation.
#' @return A matrix or vector; `transform_difference()` gives
#'   `c(rotation = <radians>, translation = <m>)` of `A^-1 B`, and
#'   `rotation_angle()` the rotation magnitude of `T` (radians).
#' @export
se3_exp <- function(xi) core_coreg_se3_exp(as.double(xi))

#' @rdname se3_exp
#' @export
se3_log <- function(T) core_coreg_se3_log(as_transform(T))

#' @rdname se3_exp
#' @export
so3_exp <- function(w) core_coreg_so3_exp(as.double(w))

#' @rdname se3_exp
#' @export
so3_log <- function(R) core_coreg_so3_log(as_rotation(R))

#' @rdname se3_exp
#' @export
skew <- function(v) core_coreg_skew(as.double(v))

#' @rdname se3_exp
#' @export
invert <- function(T) core_coreg_invert(as_transform(T))

#' @rdname se3_exp
#' @export
coreg_identity <- function() diag(4)

#' @rdname se3_exp
#' @export
yaw_transform <- function(yaw, tx = 0, ty = 0, tz = 0) {
  core_coreg_yaw_transform(as.double(yaw), as.double(tx), as.double(ty), as.double(tz))
}

#' @rdname se3_exp
#' @export
rotation_angle <- function(T) core_coreg_rotation_angle(as_rotation(T))

#' @rdname se3_exp
#' @export
transform_difference <- function(A, B) {
  stats::setNames(core_coreg_transform_difference(as_transform(A), as_transform(B)), c("rotation", "translation"))
}

#' Move points by a rigid transform
#'
#' `transform_points()` applies `T` to n x 3 points; `transform_vectors()`
#' rotates directions (the translation is ignored).
#'
#' @param T A 4 x 4 transform.
#' @param points,vectors n x 3 matrices.
#' @return An n x 3 matrix.
#' @export
transform_points <- function(T, points) core_coreg_transform_points(as_transform(T), as_points(points))

#' @rdname transform_points
#' @export
transform_vectors <- function(T, vectors) core_coreg_transform_vectors(as_transform(T), as_points(vectors, "vectors"))

#' Least-squares rigid fits between paired points
#'
#' `coreg_kabsch()` is the weighted Kabsch (1976) fit of all six degrees of
#' freedom (reflections suppressed), the Python package's
#' `sylva.coreg.kabsch`; `kabsch_2d_yaw()` fits yaw and translation only,
#' the right estimator for levelled scans.
#'
#' @param source,target n x 3 corresponding points (n >= 3; n >= 2 for the
#'   yaw fit).
#' @param weights Optional weights, one per pair.
#' @return The 4 x 4 transform mapping `source` onto `target`.
#' @export
coreg_kabsch <- function(source, target, weights = NULL) {
  s <- as_points(source, "source")
  t <- as_points(target, "target")
  if (!identical(dim(s), dim(t))) stop("source and target must both be (N, 3) arrays of equal length")
  if (nrow(s) < 3) stop("at least 3 correspondences are required")
  core_coreg_kabsch(s, t, if (is.null(weights)) NULL else as.double(weights))
}

#' @rdname coreg_kabsch
#' @export
kabsch_2d_yaw <- function(source, target, weights = NULL) {
  s <- as_points(source, "source")
  t <- as_points(target, "target")
  if (!identical(dim(s), dim(t)) || nrow(s) < 2) stop("source and target must be (N, 3) arrays of equal length, N >= 2")
  core_coreg_kabsch_2d_yaw(s, t, if (is.null(weights)) NULL else as.double(weights))
}
