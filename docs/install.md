# Install

A conda environment carries the Python and the Rust toolchain together, so
the core builds without anything installed system-wide:

```bash
conda create -n sylva -c conda-forge python=3.12 rust maturin
conda activate sylva
maturin develop --release        # builds the Rust core into the environment
```

`maturin develop` puts an editable install in the active environment: Python
changes are picked up at once, Rust changes after re-running it. `pip install
-e .` builds the same way through maturin.

Without conda, any Python >= 3.10 with a Rust toolchain from
[rustup](https://rustup.rs) and `pip install maturin` does the same job, in a
virtual environment of your choice.

Reading RIEGL `.rxp` files needs RiVLib's `libscanifc` (proprietary; download
from RIEGL). Point `RIVLIB_PATH` at the extracted directory or pass
`library=` to `sylva.io.read_rxp`.
