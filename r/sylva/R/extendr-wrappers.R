# Calls into the Rust routines registered by src/rust (extendr).
# Keep in step with extendr_module! in src/rust/src/lib.rs.

#' @useDynLib sylva, .registration = TRUE
NULL

core_read <- function(path) .Call(wrap__core_read, path)
core_write <- function(cloud, path) invisible(.Call(wrap__core_write, cloud, path))
core_cloud <- function(cloud) .Call(wrap__core_cloud, cloud)
core_voxel_downsample <- function(cloud, voxel_size, centroid) .Call(wrap__core_voxel_downsample, cloud, voxel_size, centroid)
