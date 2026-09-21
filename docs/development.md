# Development

## Layout

```
crates/sylva-rs   Rust library (no Python dependency)
crates/sylva-py     PyO3 bindings -> sylva._core
crates/sylva-cli    Rust CLI (optional; the Python CLI covers the same commands)
python/sylva        Python API
tests/              pytest suite on synthetic forests
docs/               this documentation, and the example notebooks
```

## Build and test

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest ruff numpy
maturin develop --release
pytest
cargo test -p sylva-rs
```

## Documentation

```bash
pip install -e '.[docs]'     # or: pip install mkdocs-material 'mkdocstrings[python]'
mkdocs serve                 # live preview at http://127.0.0.1:8000
mkdocs build --strict        # what CI runs
```

The API reference is generated from the docstrings in `python/sylva`, so the
extension does not have to be built to build the docs.
