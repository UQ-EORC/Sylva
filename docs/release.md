# Releasing

Sylva is published in two places under one version number:

| Registry | Package | What it is | Install |
|---|---|---|---|
| PyPI | `sylva-rs` | Python package `sylva` with the compiled core | `pip install sylva-rs` |
| crates.io | `sylva-rs` | the Rust library (`use sylva_rs::...`) | `cargo add sylva-rs` |
| crates.io | `sylva-cli` | the `sylva` binary without Python | `cargo install sylva-cli` |

`sylva-py` (the PyO3 bindings) is marked `publish = false`: it reaches users
only inside the wheels.

The PyPI *distribution* is `sylva-rs` but the *import* is `sylva`. An unrelated
project already owns the name `sylva` on PyPI; it also installs a top-level
`sylva` package, so the two cannot share an environment.

## One-time setup

Both registries use trusted publishing (OpenID Connect from GitHub Actions), so
no API tokens are stored in the repository.

1. **PyPI.** On <https://pypi.org/manage/account/publishing/> add a *pending
   publisher*: project `sylva-rs`, owner `TerraSpatial`, repository `Sylva`,
   workflow `release.yml`, environment `pypi`. The first release creates the
   project.
2. **crates.io.** Trusted publishing can only be switched on for a crate that
   exists, so publish the first version by hand:

    ```bash
    cargo login                      # token from https://crates.io/settings/tokens
    cargo publish -p sylva-rs
    cargo publish -p sylva-cli       # after sylva-rs is indexed
    ```

    Then, in each crate's settings on crates.io, add a trusted publisher:
    repository `TerraSpatial/Sylva`, workflow `release.yml`, environment
    `crates-io`.
3. **GitHub.** Create the environments `pypi` and `crates-io` (Settings →
   Environments); add yourself as a required reviewer if releases should wait
   for approval.

A crates.io release is permanent: versions can be yanked but never deleted or
replaced, and the name is claimed for good. PyPI is the same in practice.

## Cutting a release

1. Set the version in `Cargo.toml` (`[workspace.package]`), in `pyproject.toml`,
   and in the `sylva-rs` dependency of `crates/sylva-cli/Cargo.toml`; run
   `cargo check` so `Cargo.lock` follows.
2. Check locally:

    ```bash
    cargo test --release -p sylva-rs && pytest
    cargo publish --dry-run -p sylva-rs
    maturin sdist -o dist && maturin build --release -o dist && twine check dist/*
    ```

3. Commit, tag and push: `git tag v0.1.0 && git push origin main v0.1.0`.

The `release` workflow then tests, checks that the tag matches both version
numbers, builds abi3 wheels (one per platform serves every Python ≥ 3.10) for
Linux x86-64 / aarch64 (manylinux2014), macOS x86-64 / arm64 and Windows x86-64
plus an sdist, and publishes to PyPI and crates.io. Running the workflow by
hand (*Run workflow*) builds everything without publishing, which is the way
to try a platform before tagging.

## Notes

- Reading RIEGL `.rxp` needs RiVLib at run time; it is loaded dynamically, so
  the wheels carry no RIEGL code and build without it.
- docs.rs builds the crate documentation automatically after each crates.io
  release.
- While the GitHub repository is private, the links and the icon in the README
  shown on PyPI and crates.io lead nowhere. Make the repository public before
  the first release, or accept broken links until then.
