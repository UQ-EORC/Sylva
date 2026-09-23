# Install

A conda environment carries the Python, the Rust toolchain and the C linker
together, so the core builds with nothing installed system-wide:

```bash
conda create -n sylva -c conda-forge python=3.12 rust maturin c-compiler
conda activate sylva
maturin develop --release        # builds the Rust core into the environment
```

Or from the environment file in the repository, which adds pytest and the
notebook extras:

```bash
conda env create -f environment.yml
conda activate sylva
maturin develop --release
```

`maturin develop` puts an editable install in the active environment: Python
changes are picked up at once, Rust changes after re-running it. `pip install
-e .` builds the same way through maturin.

!!! warning "`error: linker `cc` not found`"
    `rustc` hands the final link to a program called `cc`. Conda's `rust`
    package does not bring one, so an environment of `python rust maturin`
    alone fails at the first crate that has a build script. Adding
    `c-compiler` installs `gcc` under the plain `cc` name inside the
    environment, and its activation script sets `CC`, `CFLAGS` and `LDFLAGS`
    to match. On a machine that already has a system compiler the build
    works without it, which is why it is easy to miss.

Without conda, any Python >= 3.10 with a Rust toolchain from
[rustup](https://rustup.rs), a system C compiler (`gcc` or `build-essential`
on Linux, the Xcode command line tools on macOS) and `pip install maturin`
does the same job, in a virtual environment of your choice.

Reading RIEGL `.rxp` files needs RiVLib's `libscanifc` (proprietary; download
from RIEGL). Point `RIVLIB_PATH` at the extracted directory or pass
`library=` to `sylva.io.read_rxp`.
