# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.

#' Memory limits
#'
#' Allocations that scale with a grid rather than with the data (ray-traced
#' voxel grids, neighbour graphs, the buttress raster) are sized first and
#' refused when they would not fit in the budget, with a message saying
#' what would make them fit. The budget is 80 % of what the system reports
#' as free, or `SYLVA_MEM_BUDGET` in gigabytes, or whatever
#' `set_memory_budget()` was given. These are the Python package's
#' `sylva.limits.available()`, `budget()`, `set_budget()`, `check()` and
#' `human()`.
#'
#' `memory_available()`: memory the system reports as free (bytes), `NULL`
#' where the system will not say. `memory_budget()`: the most one allocation
#' may ask for (bytes), `NULL` when nothing is known (nothing is then
#' refused). `set_memory_budget()`: set the budget for this session; `NULL`
#' goes back to reading the system. `memory_check()`: an error if `cells *
#' per_cell` bytes are over budget. `human_bytes()`: bytes as a person writes
#' them ("3.2 TB").
#'
#' @param gigabytes The limit, or `NULL`.
#' @param cells,per_cell How many of the thing, and bytes each.
#' @param what What is being made, with its dimensions ("a 400 x 400 x 90 grid").
#' @param hint What would make it fit ("a larger voxel, or a smaller area").
#' @param bytes A size.
#' @export
memory_available <- function() core_memory_available()

#' @rdname memory_available
#' @export
memory_budget <- function() core_memory_budget()

#' @rdname memory_available
#' @export
set_memory_budget <- function(gigabytes) {
  invisible(core_set_memory_budget(if (is.null(gigabytes)) 0 else trunc(as.double(gigabytes) * 1e9)))
}

#' @rdname memory_available
#' @export
memory_check <- function(cells, per_cell, what, hint) {
  invisible(core_memory_check(trunc(as.double(cells)), trunc(as.double(per_cell)), as.character(what), as.character(hint)))
}

#' @rdname memory_available
#' @export
human_bytes <- function(bytes) vapply(as.double(bytes), core_memory_human, "")
