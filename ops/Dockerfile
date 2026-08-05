# Build the quantization binaries for Hugging Face Jobs.
#
# Two images from one file, because the flavor decision is not settled — see
# `ops/README.md`. The run is ~60 % Leech encoding, which is pure CPU, so a CPU
# flavor may well beat a GPU one on cost; the 8B step measures it.
#
#   CPU (works today):
#     docker build -f ops/Dockerfile -t <user>/llvq:cpu .
#
#   CUDA, local docker only — the `cuda` feature exists
#   (`llvq-llm/Cargo.toml`), but a Space builds its Dockerfile **as written**
#   and never sees `--build-arg`, so the Hub path is `ops/Dockerfile.cuda`:
#     docker build -f ops/Dockerfile -t <user>/llvq:cuda \
#       --build-arg BUILD_BASE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
#       --build-arg RUNTIME_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#       --build-arg FEATURES=cuda,fast-linalg .
#
# Build the context from the repo root, not from `ops/` — the Dockerfile needs
# the whole workspace.
#
# `Cargo.lock` is tracked by git — `.gitignore` holds two lines, `target/` and
# `__pycache__/` — and the build below is pinned to it with `--locked`, so the
# image is built from the dependency set the committed tree records.

ARG BUILD_BASE=debian:bookworm-slim
ARG RUNTIME_BASE=debian:bookworm-slim

FROM ${BUILD_BASE} AS build
ARG FEATURES=fast-linalg
# rustup rather than the `rust:` image, so the same stage works on top of a
# CUDA devel base without maintaining a second Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates build-essential pkg-config libssl-dev \
 && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY . .
# `--locked`: without it cargo re-resolves dependencies at build time, so the
# image is not built from the committed tree and a number it produces is
# traceable to no commit. For a timed kernel, that is the difference between a
# measurement and an anecdote.
RUN cargo build --release --locked -p llvq-llm --features ${FEATURES} \
      --bin smoke --bin ppl --bin mmlu --bin seal --bin run --bin oracle

FROM ${RUNTIME_BASE}
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libssl3 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build \
      /src/target/release/smoke \
      /src/target/release/ppl \
      /src/target/release/mmlu \
      /src/target/release/seal \
      /src/target/release/run \
      /src/target/release/oracle \
      /usr/local/bin/
# The Hub cache lands on the ephemeral disk, which every flavor has in the
# hundreds of GB — the container filesystem does not.
ENV HF_HOME=/scratch/hf
WORKDIR /scratch
# Only for the Space's own runtime: a Job always overrides the command. This
# image serves nothing, it is a binary container that Spaces happens to be a
# convenient builder for — so the Space will sit idle on free CPU rather than
# crash-loop.
CMD ["sleep", "infinity"]
