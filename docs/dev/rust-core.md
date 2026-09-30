# The Rust core

Sylva's computations live in the Rust core (`crates/sylva-rs`); the Python
package (`crates/sylva-py` + `python/sylva`) is a thin layer over it. An R
package on the same core, with the same functions under the same names and
tests that check R against Python outputs, is kept on the `r-package`
branch.

## Design

- **The core owns the logic.** Anything numerical, any decision (accept a
  pair, merge two stems), any file format and any orchestration (the
  coregistration pipeline, RiSCAN project parsing, mesh writers) lives in
  `sylva-rs`, so that any language binding gets it unchanged.
- **Bindings convert, the language layer presents.** `sylva-py` only
  converts arguments and results. `python/sylva` keeps Python's idiom
  (dataclasses and NumPy arrays), the docstrings and argument checking.
- **Containers stay data.** Result objects such as `GapProfile`,
  `SurveyResult` or `QSM` are plain containers of arrays; their methods call
  stateless core functions with those arrays, and they pickle.
- **Per-language only:** the command-line interface (Python, and the
  `sylva` binary), the progress bar, and GeoTIFF export (rasterio).

## Changing or adding a computation

1. **Record.** `tests/parity/cases_<module>.py` builds fixed inputs from
   seeded NumPy generators and returns the outputs; `python
   tests/parity/capture.py <module>` records them from the current code.
2. **Change.** Implement in `sylva-rs` with Rust unit tests and call it from
   Python through `_core`, keeping the Python signature and docstring.
3. **Check.** `tests/test_parity.py` (tolerance 1e-9 relative) and the full
   Python suite must pass unchanged. A recording is regenerated only for a
   documented, intended change of behaviour, in its own commit.
   The recordings belong to the machine that made them: another CPU or C
   library shifts the last digits, and the coregistration's accept/reject
   decisions can follow, so the release workflow runs every test except
   these.

`sylva_rs::numeric` holds NumPy-exact helpers (histograms, quantiles,
`arange`, `gradient`, pairwise sums), and `sylva_rs::nprandom` reproduces
NumPy's `default_rng` streams (uniform, ziggurat normal and integer draws,
`choice`), so ported code can match what NumPy gave to the bit.
