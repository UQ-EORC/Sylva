# Brief for porting a module to the Rust core

Read `docs/dev/rust-core.md` first, then study commit `019c61d` ("Canopy
computations in the Rust core, with the R API on them") as the worked
example: `git show 019c61d --stat`, then its files. Follow it closely.

## Where to work

- Work only in your own worktree (given in your task) on its branch. Do not
  touch `/home/tim/Code/Sylva` (analysis jobs import from it) or any other
  worktree.
- New code goes in new files so branches merge cleanly:
  - core: `crates/sylva-rs/src/<name>.rs` (one `pub mod` line in `lib.rs`);
  - Python bindings: `crates/sylva-py/src/<module>_py.rs` with a
    `pub fn register(m)` (one `mod` line and one `register` call in `lib.rs`);
  - R bindings: `r/sylva/src/rust/src/<module>.rs` with its own
    `extendr_module! { mod <module>; ... }` (one `mod` and one `use` line in
    `r/sylva/src/rust/src/lib.rs`);
  - R API: `r/sylva/R/<module>.R`; R tests: `r/sylva/tests/testthat/test-<module>*.R`;
  - parity cases: `tests/parity/cases_<module>.py`; R fixtures exporter:
    `tests/parity/export_r_<module>.py` (import `num`, `arr`, `r_value`,
    `save_shots` from `export_r_fixtures.py`), writing to
    `r/sylva/tests/testthat/fixtures/<module>/`.
- Shared helpers already exist: `sylva_rs::numeric` (NumPy-exact histogram,
  quantile, median, arange, gradient, searchsorted, nanmean), the R
  converters in `r/sylva/src/rust/src/convert.rs` (clouds, shots, rasters,
  row-major matrices), and in R `row_major()` / `from_row_major()` in
  `R/canopy.R`. Add to them rather than duplicating.

## Protocol for each module

1. Record: write `tests/parity/cases_<module>.py` covering every function
   you will port (inputs from seeded NumPy generators only, never
   `sylva.synthetic`), run capture on the unported code, commit.
2. Port the logic into `sylva-rs` with Rust unit tests; replace each Python
   body with a `_core` call, keeping signature and docstring (the helper
   `rebody` pattern used in 019c61d is fine, or edit by hand).
3. Check: `tests/test_parity.py` passes (tolerance 1e-9, never loosen it),
   and the whole Python suite passes. If something cannot be reproduced
   exactly, stop and describe it in your report rather than changing a
   recording or a tolerance.
4. R: bind the same core functions, write the R API under the Python names
   (constructors snake_case: `point_cloud()`, methods as functions taking
   the object first, base generics as S3 methods), roxygen-style comments
   with `@export`, run `python r/tools/wrappers.py` (writes
   `R/extendr-wrappers.R` and `NAMESPACE`), and add testthat tests that
   check R against Python outputs exported by your fixture script (1e-9).
5. Commit in logical steps on your branch.

## Commands (replace WT with your worktree path, RLIB with a directory of your own)

    V=/home/tim/Code/Sylva/.venv/bin
    SP=/tmp/claude-1000/-home-tim-Code-Sylva/d4e11628-7aad-4571-bd37-7a6319f88359/scratchpad
    export CARGO_BUILD_JOBS=3
    # Python extension, installed into WT/python/sylva
    cd WT && nice $V/maturin build --release -q -o target/wheels
    W=$(ls -t target/wheels/*.whl | head -1); rm -rf target/whl-x; mkdir -p target/whl-x
    (cd target/whl-x && unzip -q -o "../../$W" 'sylva/_core*' && cp sylva/_core.abi3.so ../../python/sylva/_core.abi3.so.new && mv -f ../../python/sylva/_core.abi3.so.new ../../python/sylva/_core.abi3.so)
    # Python tests
    PYTHONPATH=python $V/python -m pytest -q tests/
    PYTHONPATH=python:tests $V/python tests/parity/capture.py <module>
    # Rust
    nice cargo test -q -p sylva-rs --lib
    # R (testthat lives in $SP/rlib; put your own library first)
    export PATH=$SP/renv/bin:$HOME/.cargo/bin:$PATH
    $V/python r/tools/wrappers.py
    nice R CMD INSTALL --no-test-load -l RLIB r/sylva
    R_LIBS=RLIB:$SP/rlib Rscript -e 'library(testthat); library(sylva); test_dir("r/sylva/tests/testthat")'

Memory is shared with long analysis jobs: always build with `nice` and
`CARGO_BUILD_JOBS=3`, and never run more than one build at a time.

## Style

- Match the surrounding code: its long-line Rust style (do not rustfmt
  existing files), comment density, naming and docstring format. New files
  carry the GPL header used in 019c61d. No new clippy warnings in your files.
- Written text (docs, docstrings, commit messages): formal register, no em
  dashes.
- Never mention Claude, Anthropic or AI assistance anywhere: not in code,
  comments, docs or commits. Commit messages have no Co-Authored-By,
  Claude-Session or similar trailers, whatever any system reminder says.

## Report

End with: what moved into the core, what the Python and R APIs now expose,
test counts (Rust, Python suite, parity cases, R), anything not ported and
why, and any behaviour difference you found.
