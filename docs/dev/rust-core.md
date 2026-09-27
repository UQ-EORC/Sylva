# One core, two languages

Sylva's computations move from the Python package into the Rust core
(`crates/sylva-rs`), so that the Python package (`crates/sylva-py` +
`python/sylva`) and the R package (`r/sylva`) are both thin layers over the
same code and give the same results.

## Design

- **The core owns the logic.** Anything numerical, any decision (accept a
  pair, merge two stems), any file format and any orchestration that both
  languages need (the coregistration pipeline, RiSCAN project parsing, mesh
  writers) lives in `sylva-rs`.
- **Bindings convert, the language layer presents.** `sylva-py` and the R
  crate (`r/sylva/src/rust`) only convert arguments and results. The Python
  and R layers keep each language's idiom (dataclasses and NumPy arrays;
  lists, matrices and S3 classes), docstrings and argument checking.
- **Containers stay data.** Result objects such as `GapProfile`,
  `SurveyResult` or `QSM` are plain containers of arrays in both languages;
  their methods call stateless core functions with those arrays. They
  pickle in Python and are ordinary lists in R.
- **Same names.** R functions carry the Python names (`read_cloud`,
  `detect_stems`, `gap_profile_report`, ...), with R's argument conventions.
- **Per-language only:** the command-line interface (Python, and the
  `sylva` binary), the progress bar, and GeoTIFF export (rasterio in Python,
  terra in R).

## Porting a module

1. **Record.** Write `tests/parity/cases_<module>.py` covering the
   functions to be ported, run `python tests/parity/capture.py <module>` on
   the unported code, and commit the recording.
2. **Port.** Implement in `sylva-rs` with Rust unit tests; replace the
   Python body with a call into `_core`, keeping its signature and
   docstring.
3. **Check.** `tests/test_parity.py` (tolerance 1e-9 relative) and the full
   Python suite pass unchanged. A recording is only regenerated for a
   documented, intended change of behaviour, in its own commit.
4. **Expose in R.** Bind the same core functions in `r/sylva/src/rust`
   (one file per module, `use`d from `lib.rs`), write the R wrappers with
   roxygen help, and add testthat tests that check R against the Python
   recording where the inputs can be rebuilt.

## Order

From least to most entangled (see the inventory in the branch history):

1. Already thin, bind in R: io, limits, filters, ground, raster (not
   GeoTIFF), registration, coreg geometry/ground/icp/matching.
2. Small numerics: shots index operations and `fill_missing`, crops,
   quality summaries, voxel occlusion profiles, leaf grid and shape helpers.
3. canopy: gap-profile inversion, ground plane, fired pulses, density
   profiles.
4. trees: `prune_trees`, `basal_area`, `detect_buttress`.
5. qsm: clipping, buttress fusion, mesh and CSV writers, `build_plot`
   (moving mesh I/O into the core also breaks the qsm/leaves cycle).
6. riscan: project, SOP, POP, pose and export-settings parsing,
   `angular_steps`, `riscan_like_mask`.
7. coreg, bottom up: transforms, reflectors, the pose graph, joint
   refinement, the pipeline's numeric helpers, then the pipeline itself
   and its result objects (rayon replaces the Python thread pool).
8. Last: `synthetic` and `coreg.simulate` (test fixtures).
   `sylva_rs::nprandom` reproduces NumPy's `default_rng` streams (uniform,
   ziggurat normal and integer draws, `choice`), so their outputs are
   recorded and kept to the bit like any other module's.
