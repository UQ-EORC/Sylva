# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

# ------------------------------------------------------------------ settings

#' Settings of the coregistration pipeline
#'
#' Every setting of the Python package's `CoregConfig`, with its defaults
#' (which began as tlsalign's). Scan indices are 1-based (`reference_scan`).
#'
#' @param ground_cell_size Terrain model resolution (m).
#' @param ground_min_coverage Least fraction of the elevations from -30 to
#'   +5 degrees an azimuth must sample for the terrain in that direction to
#'   be fitted from it (scans read from file or given `origin`); `NULL` fits
#'   the terrain from every return.
#' @param stems,matching,icp [stem_detection_config()], [match_config()],
#'   [icp_config()].
#' @param icp_voxel,icp_min_planarity,icp_max_height The ICP subsample:
#'   voxel (m), planarity gate (0 keeps foliage) and the height above ground
#'   (m) above which returns are left out.
#' @param use_reflectors,min_reflector_matches,reflector_tolerance Try
#'   reflective targets before stems where both scans saw enough.
#' @param trusted_reflector_matches,trusted_reflector_rmse A target match
#'   this strong is accepted even when ICP fails its tests (0 turns it off).
#' @param min_match_inliers,max_match_ambiguity,ambiguity_margin,max_match_rmse,max_coarse_stem_rmse
#'   Screening of coarse stem matches.
#' @param height_from_ground,ground_radius,min_ground_cells,max_ground_disagreement
#'   Heights from the shared terrain, and the terrain test of a pair
#'   (`NULL` turns the test off).
#' @param max_pair_distance Pairs whose approximate positions are further
#'   apart (m) are not tried.
#' @param screen_pairs,max_pairs_per_scan Match every pair first and run ICP
#'   only on the survivors, at most `max_pairs_per_scan` per scan (`NULL`:
#'   all).
#' @param min_icp_fitness,max_icp_rmse,min_icp_fitness_above_ground,fitness_min_height
#'   ICP acceptance.
#' @param stem_agreement_tolerance,max_coarse_to_fine_shift When ICP is
#'   overruled by the stems, and when it diverged.
#' @param recover_unregistered,recovery_rounds,recovery_neighbours Retry
#'   unregistered scans against the combined registered survey.
#' @param refine_multiview,refinement_rounds,refinement_voxel_sizes,refinement_max_distances,refinement_points_per_scan,refinement_stem_weight,refinement_stem_radius,refinement_min_voxel_points,refinement_max_shift
#'   Joint multi-view refinement ([refine_joint()]).
#' @param optimise_globally,reference_scan,reject_outlier_edges The pose graph.
#' @param information_patch_points,information_min_sigma Weighting of
#'   pose-graph edges by their point-to-plane correspondences.
#' @param max_prior_shift,max_prior_rotation With priors: results further (m)
#'   or more rotated (degrees; `NULL`: not checked) from the prior are refused.
#' @param workers Scans prepared and pairs registered at once; 0 picks from
#'   the cores and free memory (`memory_per_worker_gb` per preparing scan).
#' @param memory_per_worker_gb See `workers`.
#' @param riscan_filter RiSCAN PRO's RXP import filter when reading:
#'   `"none"`, `"current"` or `"legacy"` ([riscan_like_mask()]).
#' @param riegl_options RXP reading options and bounds ([reading_options()]).
#' @param min_points_per_scan,max_points_per_scan Scans with fewer points
#'   are set aside; `max_points_per_scan` caps each scan (`NULL`: all).
#' @param verbose Print progress when no `progress` function is given.
#' @return A `sylva_coreg_config`.
#' @export
coreg_config <- function(ground_cell_size = 0.5, ground_min_coverage = 0.8, stems = stem_detection_config(),
                         matching = match_config(), icp = icp_config(), icp_voxel = 0.05, icp_min_planarity = 0.35,
                         icp_max_height = 12.0, use_reflectors = TRUE, min_reflector_matches = 3,
                         reflector_tolerance = 0.05, trusted_reflector_matches = 5, trusted_reflector_rmse = 0.03,
                         min_match_inliers = 5, max_match_ambiguity = 0.8, ambiguity_margin = 1.25,
                         max_match_rmse = 0.30, max_coarse_stem_rmse = Inf, height_from_ground = TRUE,
                         ground_radius = 30.0, min_ground_cells = 50, max_ground_disagreement = 0.25,
                         max_pair_distance = 40.0, screen_pairs = TRUE, max_pairs_per_scan = NULL,
                         min_icp_fitness = 0.04, max_icp_rmse = 0.15, min_icp_fitness_above_ground = 0.03,
                         fitness_min_height = 1.0, stem_agreement_tolerance = 0.25, max_coarse_to_fine_shift = 2.0,
                         recover_unregistered = TRUE, recovery_rounds = 2, recovery_neighbours = 6,
                         refine_multiview = FALSE, refinement_rounds = 3, refinement_voxel_sizes = c(0.10, 0.05, 0.03),
                         refinement_max_distances = c(0.30, 0.15, 0.08), refinement_points_per_scan = 400000,
                         refinement_stem_weight = 0.05, refinement_stem_radius = 0.15,
                         refinement_min_voxel_points = 1, refinement_max_shift = 0.30, optimise_globally = TRUE,
                         reference_scan = 1, reject_outlier_edges = TRUE, information_patch_points = 100.0,
                         information_min_sigma = 0.005, max_prior_shift = 5.0, max_prior_rotation = NULL,
                         workers = 0, memory_per_worker_gb = 4.0, riscan_filter = "none", riegl_options = list(),
                         min_points_per_scan = 1000, max_points_per_scan = NULL, verbose = TRUE) {
  structure(as.list(environment()), class = "sylva_coreg_config")
}

#' @export
print.sylva_coreg_config <- function(x, ...) {
  cat("<sylva_coreg_config>\n")
  invisible(x)
}

config_core <- function(config) {
  cfg <- unclass(or_else(config, coreg_config()))
  cfg$stems <- unclass(cfg$stems)
  cfg$matching <- unclass(cfg$matching)
  m <- cfg$matching
  m$use_diameters <- isTRUE(m$use_diameters)
  cfg$matching <- m
  i <- unclass(cfg$icp)
  i$voxel_sizes <- as.double(i$voxel_sizes)
  if (!is.null(i$max_distances)) i$max_distances <- as.double(i$max_distances)
  cfg$icp <- i
  cfg$reference_scan <- as.double(cfg$reference_scan) - 1
  cfg$riegl_options <- as.list(cfg$riegl_options)
  for (k in c("use_reflectors", "height_from_ground", "screen_pairs", "recover_unregistered", "refine_multiview",
              "optimise_globally", "reject_outlier_edges")) {
    cfg[[k]] <- isTRUE(cfg[[k]])
  }
  cfg
}

#' RXP reading options from a RiSCAN PRO export settings file
#'
#' Bounds from a RiSCAN export filter settings file
#' ([read_export_settings()]), overridden by explicit ones: `min_range`,
#' `max_range` (m, from the scanner), `min_deviation`, `max_deviation`,
#' `min_reflectance`, `max_reflectance`, `min_amplitude`, `max_amplitude`,
#' plus the RiVLib options `library`, `echoes`, `stride`, `shot_stride` and
#' `drop_pseudo_echoes`. `NULL` leaves a bound to the file (or open).
#'
#' @param settings The settings file, or `NULL`.
#' @param ... Explicit bounds and options.
#' @return A named list for `coreg_config(riegl_options = )`.
#' @export
reading_options <- function(settings = NULL, ...) {
  options <- list()
  if (!is.null(settings) && nzchar(settings)) {
    s <- read_export_settings(settings)
    for (name in names(s)) {
      if (s[[name]][1] > -Inf) options[[paste0("min_", name)]] <- s[[name]][1]
      if (s[[name]][2] < Inf) options[[paste0("max_", name)]] <- s[[name]][2]
    }
  }
  bounds <- list(...)
  for (k in names(bounds)) if (!is.null(bounds[[k]])) options[[k]] <- bounds[[k]]
  options
}

# --------------------------------------------------------------------- scans

scan_from_core <- function(d) {
  ground <- if (is.null(d$ground)) NULL else ground_model(d$ground$elevation, d$ground$origin, d$ground$cell_size,
                                                             d$ground$observed)
  structure(list(name = d$name, n_points = as.integer(d$n_points), ground = ground,
                 stem_map = stem_map(stem_frame(d$stems), name = d$stem_map_name, ground = ground),
                 icp_points = d$icp_points, reflectors = reflector_frame(d$reflectors), icp_heights = d$icp_heights,
                 levelling = d$levelling, origin = d$origin, source = d$source, seconds = d$seconds, error = d$error),
            class = "sylva_scan_features")
}

scan_core <- function(scan) {
  s <- unclass(scan)
  m <- unclass(s$stem_map)
  refl <- s$reflectors
  if (is.null(refl)) refl <- reflector(double(), double(), double())
  list(name = as.character(s$name), n_points = as.double(s$n_points),
       ground = if (is.null(s$ground)) NULL else unclass(s$ground), stems = as.list(m$stems),
       stem_map_name = as.character(m$name), icp_points = if (length(s$icp_points)) as_points(s$icp_points) else
         matrix(0, 0, 3), reflectors = as.list(refl), icp_heights = as.double(s$icp_heights),
       levelling = as_transform(s$levelling), origin = as.double(s$origin),
       source = if (is.null(s$source)) NULL else as.character(s$source), seconds = as.double(s$seconds),
       error = as.character(s$error))
}

cloud_input <- function(cloud) {
  if (is.character(cloud)) return(path.expand(cloud))
  as_points(cloud, "cloud")
}

#' Prepare one scan for coregistration
#'
#' Fits the terrain, detects the stems and builds the planar ICP subsample
#' of one scan: the expensive part, done once per scan however many pairs
#' are tried. A scan with too few points is returned set aside (its
#' `error` says why), not raised.
#'
#' @param cloud A path (`.rxp` through RiVLib, anything else through
#'   [read_cloud()]), a `sylva_cloud` or an n x 3 matrix.
#' @param config [coreg_config()].
#' @param name Scan name, for reports (the file name for a path).
#' @param reflectors Targets the scan saw, in its own frame ([reflector()]).
#' @param levelling 4 x 4 rotation applied to the points (and targets) first,
#'   for a scanner that was not upright.
#' @param origin Scanner position in the (levelled) cloud; the origin for a
#'   raw scan. With it, or for a scan read from file, the terrain is refitted
#'   without the directions the scanner could not see.
#' @param scan A `sylva_scan_features`.
#' @param pose The scan's `world_from_scan`.
#' @return A `sylva_scan_features`: `name`, `n_points`, `ground`
#'   ([fit_ground()]), `stem_map`, `icp_points` (float32 values),
#'   `reflectors`, `icp_heights`, `levelling`, `origin`, `source`, `seconds`
#'   and `error`. `usable()` says whether it can take part in registration;
#'   `location()` where the scanner stood under a pose.
#' @export
prepare_scan <- function(cloud, config = NULL, name = "", reflectors = NULL, levelling = NULL, origin = NULL) {
  scan_from_core(core_coreg_prepare_scan(cloud_input(cloud), config_core(config), as.character(name),
                                         if (is.null(reflectors)) NULL else as.list(reflectors),
                                         if (is.null(levelling)) NULL else as_transform(levelling),
                                         if (is.null(origin)) NULL else as.double(origin)))
}

#' @rdname prepare_scan
#' @export
usable <- function(scan) !nzchar(scan$error) && NROW(scan$icp_points) > 0

#' @rdname prepare_scan
#' @export
location <- function(scan, pose) transform_points(pose, matrix(as.double(scan$origin), 1))[1, ]

#' @export
print.sylva_scan_features <- function(x, ...) {
  cat(sprintf("<sylva_scan_features> name=%s, points=%d, stems=%d, icp_points=%d%s\n", dQuote(x$name, FALSE),
              x$n_points, length(x$stem_map), NROW(x$icp_points),
              if (nzchar(x$error)) paste0(", set aside: ", x$error) else ""))
  invisible(x)
}

# --------------------------------------------------------------------- pairs

idx1 <- function(k) if (is.null(k) || k < 0) NA_integer_ else as.integer(k + 1)
idx0 <- function(k) if (is.null(k) || is.na(k)) -1 else as.double(k) - 1

reflector_match_result <- function(m) {
  m$correspondences <- m$correspondences + 1
  storage.mode(m$correspondences) <- "integer"
  colnames(m$correspondences) <- c("source", "target")
  m$n_inliers <- as.integer(m$n_inliers)
  structure(m, class = "sylva_reflector_match")
}

icp_result <- function(r) {
  if (is.null(r)) return(NULL)
  r$information <- plane_info(r$information)
  for (k in c("n_correspondences", "iterations")) r[[k]] <- as.integer(r[[k]])
  structure(r, class = "sylva_icp_result")
}

pair_from_core <- function(d) {
  structure(list(i = idx1(d$i), j = idx1(d$j), name_i = d$name_i, name_j = d$name_j, transform = d$transform,
                 coarse_transform = d$coarse_transform,
                 match = if (is.null(d$match_result)) NULL else stem_match(d$match_result),
                 reflector_match = if (is.null(d$reflector_match)) NULL else reflector_match_result(d$reflector_match),
                 icp = icp_result(d$icp), success = d$success, reason = d$reason, seconds = d$seconds,
                 matched_source = d$matched_source, matched_target = d$matched_target,
                 coarse_stem_rmse = d$coarse_stem_rmse, fine_stem_rmse = d$fine_stem_rmse,
                 fitness_above = d$fitness_above, rival = if (is.null(d$rival)) NULL else stem_match(d$rival),
                 used_icp = d$used_icp, ground_offset = d$ground_offset, trusted = d$trusted),
            class = "sylva_pair_result")
}

match_core <- function(m) {
  if (is.null(m)) return(NULL)
  m <- unclass(m)
  m$correspondences <- matrix(as.double(m$correspondences) - 1, ncol = 2)
  if (!is.null(m$rival)) m["rival"] <- list(match_core(m$rival))
  m
}

icp_core <- function(r) {
  if (is.null(r)) return(NULL)
  r <- unclass(r)
  if (!is.null(r$information)) r$information <- unclass(r$information)
  r
}

pair_core <- function(p) {
  p <- unclass(p)
  pts <- function(x) if (length(x)) as_points(x) else matrix(0, 0, 3)
  list(i = idx0(p$i), j = idx0(p$j), name_i = as.character(p$name_i), name_j = as.character(p$name_j),
       transform = as_transform(p$transform), coarse_transform = as_transform(p$coarse_transform),
       match_result = match_core(p$match), reflector_match = match_core(p$reflector_match), icp = icp_core(p$icp),
       success = isTRUE(p$success), reason = as.character(p$reason), seconds = as.double(p$seconds),
       matched_source = pts(p$matched_source), matched_target = pts(p$matched_target),
       coarse_stem_rmse = as.double(p$coarse_stem_rmse), fine_stem_rmse = as.double(p$fine_stem_rmse),
       fitness_above = as.double(p$fitness_above), rival = match_core(p$rival), used_icp = isTRUE(p$used_icp),
       ground_offset = as.double(p$ground_offset), trusted = isTRUE(p$trusted))
}

#' Register one pair of prepared scans
#'
#' Reflective targets where both scans saw them, otherwise a global
#' stem-map match with its height taken from the two terrain models, gives
#' a coarse transform that ICP refines; the pair is accepted only if it fits
#' over all points and above the ground, and its terrain agrees. A target
#' match ICP rejects does not cost the pair its stem match, and a strong one
#' is trusted where ICP fails. The Python package's `register_pair`.
#'
#' `fitness()`, `rmse()`, `n_stem_matches()` and `stem_rmse()` read a
#' result; `summary()` gives its line of the report.
#'
#' @param source,target `sylva_scan_features` ([prepare_scan()]); the result
#'   maps `source` into `target`.
#' @param config [coreg_config()].
#' @param initial Skip the coarse matching and start ICP here.
#' @param match A stem match already computed ([match_stem_maps()]).
#' @param i,j Scan indices recorded on the result.
#' @param target_icp `target`'s [icp_target()], built with `config$icp`.
#' @param pair,object A `sylva_pair_result`.
#' @param ... Unused.
#' @return A `sylva_pair_result`: `i`, `j`, `name_i`, `name_j`, `transform`,
#'   `coarse_transform`, `match`, `reflector_match`, `icp`, `success`,
#'   `reason`, `seconds`, `matched_source`, `matched_target`,
#'   `coarse_stem_rmse`, `fine_stem_rmse`, `fitness_above`, `rival`,
#'   `used_icp`, `ground_offset` and `trusted`, as the Python `PairResult`.
#' @export
register_pair <- function(source, target, config = NULL, initial = NULL, match = NULL, i = 1, j = 2,
                          target_icp = NULL) {
  pair_from_core(core_coreg_register_pair(scan_core(source), scan_core(target), config_core(config),
                                          if (is.null(initial)) NULL else as_transform(initial), match_core(match),
                                          idx0(i), idx0(j),
                                          if (is.null(target_icp)) NULL else unclass(target_icp)$ptr))
}

#' @rdname register_pair
#' @export
fitness <- function(pair) if (is.null(pair$icp)) 0 else pair$icp$fitness

#' @rdname register_pair
#' @export
rmse <- function(pair) if (is.null(pair$icp)) Inf else pair$icp$inlier_rmse

#' @rdname register_pair
#' @export
n_stem_matches <- function(pair) if (is.null(pair$match)) 0L else pair$match$n_inliers

#' @rdname register_pair
#' @export
stem_rmse <- function(pair) if (isTRUE(pair$used_icp)) pair$fine_stem_rmse else pair$coarse_stem_rmse

#' @rdname register_pair
#' @export
summary.sylva_pair_result <- function(object, ...) core_coreg_pair_summary(pair_core(object))

#' @export
print.sylva_pair_result <- function(x, ...) {
  cat(summary(x), "\n", sep = "")
  invisible(x)
}

# ------------------------------------------------------------------- surveys

logger <- function(progress, config) {
  if (!is.null(progress)) return(progress)
  if (isTRUE(or_else(config, coreg_config())$verbose)) function(m) cat(m, "\n", sep = "") else NULL
}

survey_args <- function(pairs, approximate_positions, fixed, priors) {
  if (!is.null(pairs)) {
    if (is.list(pairs) && !is.data.frame(pairs)) pairs <- do.call(rbind, lapply(pairs, as.double))
    pairs <- matrix(as.double(pairs), ncol = 2) - 1
  }
  if (!is.null(approximate_positions)) {
    approximate_positions <- as.matrix(approximate_positions)
    storage.mode(approximate_positions) <- "double"
  }
  list(pairs = pairs, positions = approximate_positions,
       fixed_nodes = if (length(fixed)) as.double(names(fixed)) - 1 else NULL,
       fixed_poses = if (length(fixed)) lapply(unname(fixed), as_transform) else NULL,
       priors = if (is.null(priors)) NULL else lapply(priors, function(p) if (is.null(p)) NULL else as_transform(p)))
}

optimisation_from_core <- function(o, poses) {
  if (is.null(o)) return(NULL)
  structure(list(poses = poses, iterations = as.integer(o$iterations), converged = o$converged,
                 initial_error = o$initial_error, final_error = o$final_error,
                 rejected_edges = as.integer(o$rejected_edges) + 1L, edge_errors = o$edge_errors),
            class = "sylva_optimisation_result")
}

survey_from_core <- function(scans, d) {
  structure(list(scans = scans, pairs = lapply(d$pairs, pair_from_core), poses = d$poses,
                 reference = as.integer(d$reference) + 1L, optimisation = optimisation_from_core(d$optimisation, d$poses),
                 registered = d$registered, seconds = d$seconds, edge_to_pair = as.integer(d$edge_to_pair) + 1L),
            class = "sylva_survey_result")
}

#' Coregister a whole survey
#'
#' `coregister()` prepares every scan ([prepare_scan()], on several threads
#' when all are files; a scan that cannot be read is set aside with the
#' reason) and then runs `coregister_prepared()`: every candidate pair is
#' screened by stem matching and registered ([register_pair()]); accepted
#' pairs become the edges of a pose graph, each weighted by the directions
#' its surfaces constrain, solved with outlier rejection; scans left over
#' are retried against the combined registered survey and tied in by
#' pairwise measurements, or placed from their priors
#' ([place_from_prior()]); optionally every pose is refined jointly. The
#' results do not depend on the number of workers. The Python package's
#' `coregister` and `coregister_prepared`.
#'
#' @param clouds Scans: paths, `sylva_cloud`s or n x 3 matrices.
#' @param scans `sylva_scan_features` from [prepare_scan()].
#' @param config [coreg_config()].
#' @param names Scan names (default `scan_00`, `scan_01`, ...).
#' @param pairs Pairs to try (a two-column matrix or a list of pairs of
#'   1-based scan indices); all usable pairs within `max_pair_distance` if
#'   `NULL`.
#' @param reflectors Per scan, the targets it saw ([reflector()]).
#' @param approximate_positions n x 3 rough scanner positions (GNSS); `NA`
#'   rows are unknown. Taken from `priors` if `NULL`.
#' @param levelling Per scan, the 4 x 4 rotation that levels it, or `NULL`.
#' @param fixed Named list of `world_from_levelled_scan` poses held fixed,
#'   named by 1-based scan index; the others are registered into their frame.
#' @param priors Per scan, an approximate `world_from_levelled_scan` or
#'   `NULL`: results putting a scanner more than `max_prior_shift` from its
#'   prior are refused, and scans whose stems do not match are placed from it.
#' @param progress Called with each progress message; printed if `NULL` and
#'   `config$verbose`.
#' @param started `Sys.time()` at the start, for the reported duration.
#' @return A `sylva_survey_result`: `scans`, `pairs` (`sylva_pair_result`s),
#'   `poses` (`world_from_levelled_scan`), `reference` (1-based),
#'   `optimisation`, `registered`, `seconds` and `edge_to_pair` (the pair
#'   behind each pose-graph edge). See [report()], [consistency()],
#'   [save_survey()] and [transform_for()].
#' @export
coregister <- function(clouds, config = NULL, names = NULL, pairs = NULL, reflectors = NULL,
                       approximate_positions = NULL, levelling = NULL, fixed = NULL, priors = NULL, progress = NULL) {
  n <- length(clouds)
  inputs <- lapply(seq_len(n), function(k) {
    tryCatch(cloud_input(clouds[[k]]), error = function(e) {
      list(failed = paste0("Error: ", conditionMessage(e)),
           source = if (is.character(clouds[[k]])) clouds[[k]] else NULL)
    })
  })
  refl <- lapply(seq_len(n), function(k) {
    r <- if (length(reflectors)) reflectors[[k]] else NULL
    if (is.null(r)) NULL else as.list(r)
  })
  lev <- lapply(seq_len(n), function(k) {
    l <- if (length(levelling)) levelling[[k]] else NULL
    if (is.null(l)) NULL else as_transform(l)
  })
  a <- survey_args(pairs, approximate_positions, fixed, priors)
  r <- core_coreg_coregister(inputs, config_core(config), if (is.null(names)) NULL else as.character(names), refl, lev,
                             a$pairs, a$positions, a$fixed_nodes, a$fixed_poses, a$priors, logger(progress, config))
  survey_from_core(lapply(r$scans, scan_from_core), r$survey)
}

#' @rdname coregister
#' @export
coregister_prepared <- function(scans, config = NULL, pairs = NULL, approximate_positions = NULL, fixed = NULL,
                                priors = NULL, progress = NULL, started = NULL) {
  already <- if (is.null(started)) 0 else as.double(difftime(Sys.time(), started, units = "secs"))
  a <- survey_args(pairs, approximate_positions, fixed, priors)
  r <- core_coreg_coregister_prepared(lapply(scans, scan_core), config_core(config), a$pairs, a$positions,
                                      a$fixed_nodes, a$fixed_poses, a$priors, logger(progress, config), already)
  survey_from_core(scans, r)
}

#' Place one scan from an approximate pose
#'
#' For scans that see too few stems to match. The prior (a RiSCAN SOP, GNSS
#' and compass) is corrected in height by the median offset between the
#' scan's ground and that of the registered scans nearest it, then refined
#' by ICP from a 1.5 m correspondence distance; accepted on ICP fitness,
#' above-ground fitness, RMSE and terrain, and only within
#' `max_prior_shift` of the prior.
#'
#' @param scan The scan to place ([prepare_scan()]).
#' @param survey Registered scans.
#' @param poses `world_from_scan` of each of `survey`.
#' @param prior Approximate `world_from_scan` of `scan`.
#' @param config [coreg_config()].
#' @param neighbours Registered scans nearest the prior that ICP runs
#'   against (`recovery_neighbours` if `NULL`).
#' @return `list(result, used)`: a `sylva_pair_result` whose `transform` is
#'   `world_from_scan` (check `success`), and the 1-based indices into
#'   `survey` of the scans ICP ran against.
#' @export
place_from_prior <- function(scan, survey, poses, prior, config = NULL, neighbours = NULL) {
  r <- core_coreg_place_from_prior(scan_core(scan), lapply(survey, scan_core), lapply(poses, as_transform),
                                   as_transform(prior), config_core(config),
                                   if (is.null(neighbours)) NULL else as.double(neighbours))
  list(result = pair_from_core(r$result), used = as.integer(r$used) + 1L)
}

# ------------------------------------------------------------------- results

#' The transform of one scan of a survey
#'
#' `world_from_scan`, applying directly to the raw points of scan `index`
#' (its pose composed with its levelling).
#'
#' @param result A `sylva_survey_result`.
#' @param index 1-based scan index.
#' @return A 4 x 4 matrix.
#' @export
transform_for <- function(result, index) {
  core_coreg_transform_for(as_transform(result$poses[[index]]), as_transform(result$scans[[index]]$levelling))
}

scan_summaries <- function(result) {
  lapply(result$scans, function(s) {
    list(name = as.character(s$name), n_points = as.double(s$n_points), n_stems = as.double(length(s$stem_map)),
         error = as.character(s$error), source = if (is.null(s$source)) NULL else as.character(s$source),
         levelling = as_transform(s$levelling))
  })
}

optimisation_core <- function(o) {
  if (is.null(o)) return(NULL)
  list(iterations = as.double(o$iterations), converged = isTRUE(o$converged), initial_error = o$initial_error,
       final_error = o$final_error, rejected_edges = as.double(o$rejected_edges) - 1)
}

#' Quality of a coregistration
#'
#' `report()` gives the plain-text summary for a log or a QC record, the
#' Python package's `SurveyResult.report()`, word for word.
#' `consistency()` gives the horizontal stem disagreement per accepted pair
#' under the final poses: the median (or RMS) distance at which the same
#' tree lands from the two scans, a quality check that needs no ground
#' truth. `successful_pairs()`, `rejected_pairs()` (accepted pairs the
#' global solve found inconsistent) and `pair_for_edge()` read the pairs.
#'
#' @param x,result A `sylva_survey_result`.
#' @param robust Median rather than RMSE.
#' @param edge 1-based pose-graph edge.
#' @param ... Unused.
#' @return `report()`: a string; `consistency()`: a data frame with `i`,
#'   `j` (1-based) and `distance` (m).
#' @export
report.sylva_survey_result <- function(x, ...) {
  core_coreg_survey_report(scan_summaries(x), lapply(x$pairs, pair_core), lapply(x$poses, as_transform),
                           as.double(x$reference) - 1, optimisation_core(x$optimisation), as.logical(x$registered),
                           as.double(x$seconds), as.double(x$edge_to_pair) - 1)
}

#' @rdname report.sylva_survey_result
#' @export
consistency <- function(result, robust = TRUE) {
  c <- core_coreg_survey_consistency(lapply(result$pairs, pair_core), lapply(result$poses, as_transform),
                                     isTRUE(robust))
  data.frame(i = as.integer(c$i) + 1L, j = as.integer(c$j) + 1L, distance = c$value)
}

#' @rdname report.sylva_survey_result
#' @export
successful_pairs <- function(result) Filter(function(p) isTRUE(p$success), result$pairs)

#' @rdname report.sylva_survey_result
#' @export
pair_for_edge <- function(result, edge) {
  if (edge < 1 || edge > length(result$edge_to_pair)) return(NULL)
  result$pairs[[result$edge_to_pair[edge]]]
}

#' @rdname report.sylva_survey_result
#' @export
rejected_pairs <- function(result) {
  if (is.null(result$optimisation)) return(list())
  Filter(Negate(is.null), lapply(result$optimisation$rejected_edges, function(k) pair_for_edge(result, k)))
}

#' @export
print.sylva_survey_result <- function(x, ...) {
  cat(sprintf("<sylva_survey_result> %d scans, %d registered, %d pairs\n", length(x$scans), sum(x$registered),
              length(x$pairs)))
  invisible(x)
}

#' Save and load registered transforms
#'
#' `save_survey()` writes the transforms and quality of a survey as JSON
#' (`transforms.json`), the file the Python package's `SurveyResult.save()`
#' writes, byte for byte (`save` itself is base R's). `load_transforms()`
#' reads the `world_from_scan` of every registered scan from one.
#'
#' @param result A `sylva_survey_result`.
#' @param path The JSON file.
#' @return `save_survey()`: `path`, invisibly; `load_transforms()`: a named
#'   list of 4 x 4 matrices.
#' @export
save_survey <- function(result, path) {
  core_coreg_survey_save(path.expand(path), scan_summaries(result), lapply(result$pairs, pair_core),
                         lapply(result$poses, as_transform), as.double(result$reference) - 1,
                         as.logical(result$registered), as.double(result$seconds))
  invisible(path)
}

#' @rdname save_survey
#' @export
load_transforms <- function(path) core_coreg_load_transforms(path.expand(path))

#' Merge registered scans into one cloud
#'
#' Every scan moved into the reference frame ([transform_for()]) and
#' concatenated, with a `scan_id` attribute (0-based, as [merge_scans()]
#' writes it).
#'
#' @param clouds The scans as passed to [coregister()] (paths, clouds or
#'   matrices).
#' @param result The registration.
#' @param voxel Thin each scan, and the merged cloud, to one point per voxel
#'   (m); `NULL` or 0 keeps every point.
#' @param only_registered Leave out unregistered scans.
#' @param riegl_options,riscan_filter Reading of `.rxp` files, as in
#'   [coreg_config()].
#' @return A `sylva_cloud`.
#' @export
merge_clouds <- function(clouds, result, voxel = 0.02, only_registered = TRUE, riegl_options = NULL,
                         riscan_filter = "none") {
  cfg <- coreg_config(riegl_options = or_else(riegl_options, list()), riscan_filter = riscan_filter)
  r <- core_coreg_merge_clouds(lapply(clouds, cloud_input), lapply(result$poses, as_transform),
                               lapply(result$scans, function(s) as_transform(s$levelling)),
                               as.logical(result$registered), isTRUE(only_registered),
                               if (is.null(voxel) || voxel == 0) NULL else as.double(voxel), config_core(cfg))
  new_cloud(r$xyz, list(scan_id = as.integer(r$scan_id)))
}
