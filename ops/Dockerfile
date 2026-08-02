# Build the quantization binaries for Hugging Face Jobs.
#
# Two images from one file, because the flavor decision is not settled — see
# `ops/README.md`. The run is ~60 % Leech encoding, which is pure CPU, so a CPU
# flavor may well beat a GPU one on cost; the 8B step measures it.
#
#   CPU (works today):
#     docker build -f ops/Dockerfile -t <user>/llvq:cpu .
#
#   CUDA (needs code item C1 — the `cuda` feature does not exist yet):
#     docker build -f ops/Dockerfile -t <user>/llvq:cuda \
#       --build-arg BUILD_BASE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
#       --build-arg RUNTIME_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
#       --build-arg FEATURES=cuda,fast-linalg .
#
# Build the context from the repo root, not from `ops/` — the Dockerfile needs
# the whole workspace.
#
# ⚠️ `Cargo.lock` is in `.gitignore`, so this build resolves dependencies fresh
# and is **not** bit-reproducible across time. For a project whose stated frame
# is sovereignty and reproducible builds, that is worth fixing before the first
# real run: commit the lockfile.

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
RUN cargo build --release -p llvq-llm --features ${FEATURES} \
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
