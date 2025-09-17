# syntax=docker/dockerfile:1.7

FROM ubuntu:22.04

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# ---------- Base OS deps ----------
RUN set -euo pipefail \
 && apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
      ca-certificates gnupg software-properties-common \
      gcc-11 g++-11 libasan6 \
      git wget curl unzip autoconf make pkg-config \
      cmake ninja-build vim-common nano \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ---------- LLVM 15 repo (for lld-15) ----------
RUN set -euo pipefail \
 && install -d -m 0755 /etc/apt/keyrings \
 && wget -qO /etc/apt/keyrings/llvm.asc https://apt.llvm.org/llvm-snapshot.gpg.key \
 && echo "deb [signed-by=/etc/apt/keyrings/llvm.asc] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-15 main" \
      >/etc/apt/sources.list.d/llvm15.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends lld-15 \
 && rm -rf /var/lib/apt/lists/*

# ---------- Update alternatives (gcc/g++/ld.lld) ----------
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
 && update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-15 100

# ---------- amd64-specific tools ----------
RUN set -euo pipefail \
 && apt-get update \
 && apt-get install -y --no-install-recommends nasm patch \
 && rm -rf /var/lib/apt/lists/*

# ---------- Miniconda (x86_64) ----------
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN set -euo pipefail \
 && wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p "$CONDA_DIR" \
 && rm -f /tmp/miniconda.sh \
 && conda config --set always_yes yes --set changeps1 no \
 && conda clean -afy

# ---------- Bazelisk (bazel wrapper, amd64) ----------
RUN set -euo pipefail \
 && wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64 -O /usr/bin/bazel \
 && chmod +x /usr/bin/bazel

# ---------- Python dependencies ----------
WORKDIR /workspace
COPY requirements-dev.txt /workspace/requirements-dev.txt
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r /workspace/requirements-dev.txt

CMD ["/bin/bash"]