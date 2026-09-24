# syntax=docker/dockerfile:1
#
# Three stages:
#   dev      Python, the Rust toolchain and a C compiler: the dev container
#            (.devcontainer/) builds this one and mounts the checkout.
#   build    the wheel, built from the sources in the build context.
#   runtime  (default) the wheel in a slim Python, with the `sylva` command.
#
#   docker build -t sylva .
#   docker run --rm -v "$PWD":/data sylva trees plot_norm.laz
#
# Reading .rxp needs RiVLib's libscanifc, which RIEGL does not allow to be
# shipped: mount the extracted RiVLib at /opt/rivlib (RIVLIB_PATH), e.g.
#   docker run --rm -v ~/.local/lib/rivlib:/opt/rivlib:ro -v "$PWD":/data sylva coreg survey.PROJ

ARG PYTHON=3.12

FROM python:${PYTHON}-slim-bookworm AS dev
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cargo/bin:$PATH \
    RIVLIB_PATH=/opt/rivlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal \
        --default-toolchain stable --component clippy,rustfmt \
    && python -m venv /opt/venv \
    && pip install --no-cache-dir "maturin>=1.5,<2" \
    && chmod -R a+rwX /usr/local/rustup /usr/local/cargo /opt/venv
WORKDIR /workspace

FROM dev AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    maturin build --release --out /dist

FROM python:${PYTHON}-slim-bookworm AS runtime
# libstdc++ for RiVLib when it is mounted; the wheel itself needs only libc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,from=build,source=/dist,target=/dist \
    pip install --no-cache-dir "$(ls /dist/*.whl)[geotiff]"
ENV RIVLIB_PATH=/opt/rivlib
RUN useradd --create-home sylva
USER sylva
WORKDIR /data
ENTRYPOINT ["sylva"]
CMD ["--help"]
