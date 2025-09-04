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




#################################################################################3



RUN source spack-develop/share/spack/setup-env.sh && spack compilers



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
# Install Arkouda with patched Spack package
# ---------------------------
RUN source spack-develop/share/spack/setup-env.sh && spack install arkouda@2024.10.02


# ---------------------------
# Set environment variables for runtime
# ---------------------------
RUN . $SPACK_ROOT/share/spack/setup-env.sh && \
    CHPL_HOME=$(spack location -i chapel@2.4.0) && \
    ARKOUDA_HOME=$(spack location -i arkouda@2024.10.02) && \
    printf "export CHPL_HOME=%s\n" "$CHPL_HOME" > /etc/profile.d/arkouda_env.sh && \
    printf "export ARKOUDA_HOME=%s\n" "$ARKOUDA_HOME" >> /etc/profile.d/arkouda_env.sh && \
    printf 'export PATH=$CHPL_HOME/bin:$ARKOUDA_HOME/bin:$PATH\n' >> /etc/profile.d/arkouda_env.sh




# ---------------------------
# Expose the default arkouda_server port
# ---------------------------
EXPOSE 5555


# ---------------------------
# Install arkouda python front end
# Wheel files will be saved here /opt/wheels
# So the project can be installed offline later
# ---------------------------
#RUN dnf install -y wget
#RUN wget https://github.com/Bears-R-Us/arkouda/archive/refs/tags/v2024.10.02.tar.gz && tar -xf v2024.10.02.tar.gz && cd arkouda-2024.10.02 && \
#    mkdir -p /opt/wheels && python3 -m pip download -d /opt/wheels '.[dev]' && python3 -m pip install --no-index --find-links=/opt/wheels '.[dev]' 

RUN dnf -y update && dnf -y install gfortran
RUN spack compiler find
RUN spack install py-arkouda ^py-setuptools@68.2.2 ^openblas@0.3.28
RUN spack install py-arkouda ^py-setuptools@68.2.2 ^openblas@0.3.28 #installed py-arkouda@2024.10.02









