# Development

## Layout

```
crates/sylva-rs     Rust library (no Python dependency); rivlib-shim/ builds the RiVLib bridge
crates/sylva-py     PyO3 bindings -> sylva._core
crates/sylva-cli    Rust CLI (optional; a subset of the Python CLI's commands)
python/sylva        Python API and the `sylva` command (cli.py); _core.pyi is its type stub
tests/              pytest suite on synthetic forests; tests/parity holds golden outputs
docs/               this documentation, and the example notebooks
scripts/            the type-stub generator, the MTA model trainer, a plot pipeline
.github/workflows   docs, release and container builds
```

## Build and test

uv manages the Python side; it needs `cargo` and a C compiler on the PATH
(conda's `rust` and `c-compiler`, or rustup — see the [install guide](install.md)):

```bash
uv sync --extra dev              # .venv: Sylva built through maturin, pytest, ruff, scipy / matplotlib
uv run python scripts/gen_core_stub.py   # refresh python/sylva/_core.pyi after changing a binding
uv run pytest
cargo test -p sylva-rs
```

`uv sync` installs Sylva editable, so Python changes are picked up at once;
Rust changes need a rebuild — `uv run maturin develop --release` for a fast
release build, or simply re-run `uv sync`.

The compiled module `sylva._core` carries no type information of its own, so
`python/sylva/_core.pyi` records every binding's signature for editors and
type checkers; `tests/test_stub.py` fails when it is out of date. Every
public Python function needs a numpy-style docstring with a `Parameters`
section (`tests/test_docs.py`), and the same file checks that the command
lines and `sylva.*` names the guides quote exist. In the Rust crate, every
public item carries a `///` comment; `RUSTFLAGS="-W missing_docs" cargo
check -p sylva-rs` lists any that do not.

## The Rust core

The computations live in `crates/sylva-rs`; the Python package is a thin layer
over it. [The Rust core](dev/rust-core.md) explains that division and the
protocol for changing a computation, and [Reading the core without
Rust](dev/reading-the-core.md) is for anyone who wants to check what a
function does without taking up the language.

## Documentation

```bash
uv sync --extra docs           # mkdocs-material, mkdocstrings, mkdocs-jupyter
uv run mkdocs serve            # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict   # what CI runs
```

The API reference is generated from the docstrings in `python/sylva`, so the
extension does not have to be built to build the docs.
