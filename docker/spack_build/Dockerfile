# Use AlmaLinux as the base image
FROM almalinux:9

# Maintainer and metadata
LABEL maintainer="ajpotts@datacraft-solutions.com"
LABEL description="Arkouda development image with Spack and Chapel pinned to compatible versions"

# Set non-interactive installation
ENV TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------
# Basic system setup
# ---------------------------
RUN dnf install -y --allowerasing \
    git \
    gcc \
    gcc-c++ \
    make \
    cmake \
    python3 \
    python3-devel \
    python3-pip \
    curl \
    vim \
    tar \
    gzip \
    wget \
    which \
    findutils \
    gawk \
    patch \
    file \
    less \
    hostname \
    unzip \
    diffutils \
    procps-ng \
    libcurl-devel \
    && dnf clean all

# Set up Python venv if needed (optional)
RUN python3 -m pip install --upgrade pip setuptools wheel

# ---------------------------
# Clone and bootstrap Spack
# ---------------------------
WORKDIR /opt
RUN git clone -b develop https://github.com/spack/spack.git spack-develop
ENV SPACK_ROOT=/opt/spack-develop
ENV PATH=$SPACK_ROOT/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Initialize Spack environment
RUN . $SPACK_ROOT/share/spack/setup-env.sh && spack compiler find

# ---------------------------
# Copy your patched local Arkouda repo
# ---------------------------

RUN source $SPACK_ROOT/share/spack/setup-env.sh && spack repo create /opt/local-spack-repo bears_r_us



RUN source $SPACK_ROOT/share/spack/setup-env.sh && spack repo add /opt/local-spack-repo/spack_repo/bears_r_us

# ---------------------------
# Patch Arkouda dependencies
# ---------------------------

# Replace the Chapel dependency in the Arkouda package definition
RUN sed -i 's|depends_on("chapel@[^"]*", type=.*)|depends_on("chapel@2.4.0:2.4.99 +hdf5 +zmq", type=("build", "link", "run", "test"))|' \
    /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages/arkouda/package.py

# Replace the Arrow dependency in the Arkouda package definition
RUN sed -i 's|depends_on("arrow[^"]*lz4[^"]*", type=.*)|depends_on("arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd", type=("build", "link", "run", "test"))|' \
    /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages/arkouda/package.py

# ---------------------------
# Build Chapel and Arkouda via Spack
# ---------------------------
RUN . $SPACK_ROOT/share/spack/setup-env.sh && \
    spack install arkouda

# ---------------------------
# Set ARKOUDA_HOME (required for server run)
# ---------------------------
ENV ARKOUDA_HOME=/opt/spack-develop/opt/spack/linux-skylake/arkouda-2024.10.02-*

# Expose Arkouda's default server port
EXPOSE 5555

# ---------------------------
# Entrypoint or test command
# ---------------------------
# Optional: Start server automatically or override with docker run command
CMD ["/bin/bash"]

