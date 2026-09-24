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

## Containers

The `Dockerfile` builds an image with the `sylva` command and nothing else to
install. Run it on the files in the current directory:

```bash
docker build -t sylva .
docker run --rm -v "$PWD":/data sylva trees plot_norm.laz
```

RIEGL does not allow RiVLib to be redistributed, so it isn't in the image.
Mount your own copy at `/opt/rivlib` to read `.rxp`:

```bash
docker run --rm -v ~/.local/lib/rivlib:/opt/rivlib:ro -v "$PWD":/data sylva coreg survey.PROJ
```

For development, `.devcontainer/` opens the repository in VS Code (or any
dev-container tool) with Python, the Rust toolchain and a C compiler. Sylva
is installed editable in `/opt/venv` with the test, lint and docs extras.
After a Rust change, run `maturin develop --release`. Serve the docs with
`mkdocs serve -a 0.0.0.0:8000`.

Two things to know:

- **RiVLib.** Uncomment the RiVLib mount in `devcontainer.json` to read
  `.rxp`.
- **Rootless Podman.** Add `"runArgs": ["--userns=keep-id"]` so files keep
  your ownership.

Podman works in place of Docker for both.
