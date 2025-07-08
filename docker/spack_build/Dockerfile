# Use AlmaLinux as the base image
FROM almalinux:9

# Maintainer and metadata
LABEL maintainer="ajpotts@datacraft-solutions.com"
LABEL description="Arkouda development image with Spack and Chapel pinned to compatible versions"

# Set non-interactive installation
ENV TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive

# Use Bash as the shell for all RUN statements
SHELL ["/bin/bash", "-c"]

# ---------------------------
# Install system dependencies
# --allowerasing helps resolve curl-minimal and other conflicts
# ---------------------------
RUN dnf install -y --allowerasing \
    git gcc gcc-c++ make cmake \
    python3 python3-devel python3-pip \
    curl tar gzip wget which \
    findutils gawk patch file less hostname unzip \
    diffutils procps-ng libcurl-devel bzip2 \
    && dnf clean all

# ---------------------------
# Clone Spack and set environment
# ---------------------------
WORKDIR /opt
#RUN git clone -b develop https://github.com/spack/spack.git spack-develop
ENV DATE=2025-07-08
# clone the spack-develop repo from https://github.com/spack/spack
RUN git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git spack-develop-$DATE
# create a symbolic link
RUN ln -s spack-develop-$DATE spack-develop


ENV SPACK_ROOT=/opt/spack-develop
ENV PATH=$SPACK_ROOT/bin:$PATH
ENV CHPL_GMP=bundled

# ---------------------------
# Initialize Spack and find available compilers
# ---------------------------
RUN . $SPACK_ROOT/share/spack/setup-env.sh && spack compiler find

# ---------------------------
# Create a local in-container Spack repo and override Arkouda package
# ---------------------------

#RUN source spack-develop/share/spack/setup-env.sh && spack repo create /opt/local-spack-repo bears_r_us

RUN source $SPACK_ROOT/share/spack/setup-env.sh && \
    spack repo create /opt/local-spack-repo bears_r_us && \
    spack repo add /opt/local-spack-repo/spack_repo/bears_r_us #&& \
    #mkdir -p /opt/local-spack-repo/spack_repo/bears_r_us/packages/arkouda && \
    #cp $(find /root/.spack/package_repos -name package.py | grep /arkouda/package.py) /opt/local-spack-repo/spack_repo/bears_r_us/packages/arkouda/

# Let Spack initialize the default Arkouda package by installing once
#RUN . $SPACK_ROOT/share/spack/setup-env.sh && \
#    spack fetch arkouda@2024.10.02

# Create custom repo and override Arkouda after it's been registered
#RUN . $SPACK_ROOT/share/spack/setup-env.sh && \
#    spack repo create /opt/local-spack-repo bears_r_us && \
#    spack repo add /opt/local-spack-repo/spack_repo/bears_r_us && \
#    mkdir -p /opt/local-spack-repo/spack_repo/bears_r_us/packages/arkouda && \
#    cp $(spack location -r arkouda)/package.py /opt/local-spack-repo/spack_repo/bears_r_us/packages/arkouda/

# Now patch arkouda/package.py in-place
#RUN sed -i 's|depends_on("chapel@[^"]*", type=(.*))|depends_on("chapel@2.4.0:2.4.99 +hdf5 +zmq", type=\1)|' \
#    /opt/local-spack-repo/spack_repo/bears_r_us/packages/arkouda/package.py && \
#    sed -i 's|arrow +parquet +snappy +zlib +brotli +bz2 +lz4 +zstd|arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd|' \
#    /opt/local-spack-repo/spack_repo/bears_r_us/packages/arkouda/package.py


# ---------------------------
# Apply in-place sed patches to the Arkouda package.py
#   1. Pin chapel@2.4.0:2.4.99 +hdf5 +zmq
#   2. Disable lz4 in arrow (~lz4)
# ---------------------------
# Replace full chapel dependency line
RUN export ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos -name package.py | grep /arkouda/package.py) && \
    sed -i 's|depends_on("chapel@[^"]*", type=(.*))|depends_on("chapel@2.4.0:2.4.99 +hdf5 +zmq", type=("build", "link", "run", "test"))|' \
    $ARKOUDA_PACKAGE_FILE && \
    sed -i 's|arrow +parquet +snappy +zlib +brotli +bz2 +lz4 +zstd|arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd|' \
    $ARKOUDA_PACKAGE_FILE


# ---------------------------
# Install Chapel 2.4.0 explicitly (pinned version)
# ---------------------------
RUN . $SPACK_ROOT/share/spack/setup-env.sh && \
    spack install chapel@2.4.0 +hdf5 +zmq

# ---------------------------
# Install Arkouda with patched Spack package
# ---------------------------
#RUN . $SPACK_ROOT/share/spack/setup-env.sh && \
#    spack install arkouda@2024.10.02

# ---------------------------
# Set environment variables for runtime
# ---------------------------
#ENV CHPL_HOME=$(spack location -i chapel@2.4.0)
#ENV ARKOUDA_HOME=$(spack location -i arkouda)
#ENV PATH="$CHPL_HOME/bin:$ARKOUDA_HOME/bin:$PATH"
#ENV PATH="$ARKOUDA_HOME/bin:$PATH"


# ---------------------------
# Set default working directory and command
# ---------------------------
#WORKDIR $ARKOUDA_HOME
#EXPOSE 5555
#CMD ["arkouda_server"]

