# Install

```bash
pip install sylva-rs             # Python: import sylva
cargo add sylva-rs               # Rust library: use sylva_rs::...
cargo install sylva-cli          # the sylva command without Python
```

The wheels on PyPI carry the compiled core for Linux (x86-64 and arm64), macOS
(Intel and Apple silicon) and Windows, and serve every Python from 3.10. The
distribution is called `sylva-rs` because an unrelated project owns the name
`sylva` on PyPI; the package you import is `sylva`, and the two cannot be
installed in the same environment.

## From source

Development uses [uv](https://docs.astral.sh/uv/), which builds the Rust core
through maturin and installs Sylva editable into a `.venv`. uv cannot install
a Rust toolchain or a C compiler, so a conda environment supplies those, and
nothing else needs to be installed system-wide:

```bash
conda create -n sylva -c conda-forge rust c-compiler
conda activate sylva
uv sync --extra dev        # venv + Sylva + the dev extras (pytest, ruff, ...)
uv run pytest
```

Or from the environment file in the repository, which adds the Python
interpreter and maturin (the latter for `maturin develop --release`
iteration):

```bash
conda env create -f environment.yml
conda activate sylva
uv sync --extra dev
```

`uv sync` installs Sylva editable: Python changes are picked up at once,
Rust changes after re-running it. The tools run from the venv with
`uv run pytest`, `uv run ruff check .` and `uv run mkdocs serve` (the docs
extras need `uv sync --extra docs`).

!!! warning "`error: linker `cc` not found`"
    `rustc` hands the final link to a program called `cc`. Conda's `rust`
    package does not bring one, so an environment of `python rust maturin`
    alone fails at the first crate that has a build script. Adding
    `c-compiler` installs `gcc` under the plain `cc` name inside the
    environment, and its activation script sets `CC`, `CFLAGS` and `LDFLAGS`
    to match. On a machine that already has a system compiler the build
    works without it, which is why it is easy to miss.

Without conda, uv provides the Python too: a Rust toolchain from
[rustup](https://rustup.rs), a system C compiler (`gcc` or `build-essential`
on Linux, the Xcode command line tools on macOS) and `uv sync --extra dev`
are all that is needed.

Reading RIEGL `.rxp` files needs RiVLib's `libscanifc` (proprietary; download
from RIEGL). Point `RIVLIB_PATH` at the extracted directory or pass
`library=` to `sylva.io.read_rxp`. To convert `.rxp` the way RiSCAN PRO does
(MTA resolution and its filters, see the [I/O guide](guide/io.md)), build the
small shim once from the same SDK; it needs a C++ compiler:

```bash
RIVLIB=/path/to/rivlib-2_6_0-x86_64-linux-gcc11 crates/sylva-rs/rivlib-shim/build.sh
```

Reading `.rdbx` databases (RiSCAN PRO's imported scans) likewise needs
`librdb` from RIEGL's RDB SDK: point `RDBLIB_PATH` at the extracted directory
or pass `library=` to `sylva.io.read_rdbx`. All of these are loaded at run
time; nothing else is affected when they are absent.

## Containers

The `Dockerfile` builds an image with the `sylva` command and nothing else to
install. Run it on the files in the current directory:

```bash
docker build -t sylva .
docker run --rm -v "$PWD":/data sylva trees plot_norm.laz
```

Each push to `main` also publishes the image to GitHub's container registry
as `ghcr.io/uq-eorc/sylva` (`:latest`, and `:X.Y.Z` for releases). It is
private like the repository, so `docker login ghcr.io` with a GitHub token
that can read packages first:

```bash
docker run --rm -v "$PWD":/data ghcr.io/uq-eorc/sylva trees plot_norm.laz
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
