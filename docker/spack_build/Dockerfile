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


RUN source spack-develop/share/spack/setup-env.sh && spack install arrow


##  This section should be unnecessary and is just to break the chapel install into smaller steps for debugging purposes:
RUN source spack-develop/share/spack/setup-env.sh && spack install boost --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install pigz --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install protobuf --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install pcre2 --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install libzmq --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install llvm --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install libsigsegv --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install libevent --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install findutils --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install m4 --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install autoconf --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install automake --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install libtool --reuse
RUN source spack-develop/share/spack/setup-env.sh && spack install gmp --reuse



RUN source spack-develop/share/spack/setup-env.sh && spack install chapel@2.4.0

# The path to chpl is found using:
# find $(spack location -i chapel) -type f -name chpl
#ENV PATH="/opt/spack-develop-2025-07-02/opt/spack/linux-skylake/chapel-2.5.0-rlqgyjmqp5f5y72rqqnqdyr2pyaj2p2f/bin:${PATH}"






RUN source spack-develop/share/spack/setup-env.sh && spack repo create /opt/local-spack-repo bears_r_us


#   get the package file with:
#  find /root/.spack/package_repos -name package.py | grep arkouda
#   Then use sed to replace
#   depends_on("chapel@2.1: +hdf5 +zmq", type=("build", "link", "run", "test"))
#   with
#   depends_on("chapel@2.4.0:2.4.99 +hdf5 +zmq", type=("build", "link", "run", "test"))
RUN sed -E -i 's|depends_on\("chapel@[^"]*", type=\(([^)]*)\)\)|depends_on("chapel@2.4.0:2.4.99 +hdf5 +zmq", type=(\1))|' \
    /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages/arkouda/package.py

#   Now replace:
#   depends_on(
#        "arrow +parquet +snappy +zlib +brotli +bz2 +lz4 +zstd",
#        type=("build", "link", "run", "test"),
#    )
#   with
#   depends_on(
#    "arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd",
#    type=("build", "link", "run", "test"),
#   )
RUN perl -0777 -pi -e 's|depends_on\(\n\s*"arrow \+parquet \+snappy \+zlib \+brotli \+bz2 \+lz4 \+zstd",\n\s*type=\("build", "link", "run", "test"\),\n\s*\)|depends_on(\n    "arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd",\n    type=("build", "link", "run", "test"),\n)|g' \
    /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages/arkouda/package.py


RUN source spack-develop/share/spack/setup-env.sh && spack repo add /opt/local-spack-repo/spack_repo/bears_r_us
RUN source spack-develop/share/spack/setup-env.sh && spack install arkouda

#   To get ARKOUDA_HOME:
#   export ARKOUDA_HOME=$(spack location -i arkouda)/bin
ENV ARKOUDA_HOME=/opt/spack-develop-2025-07-03/opt/spack/linux-skylake/arkouda-2024.10.02-j2xdyerofa7gzxmjrykvh2xfzf5cgab5/bin

WORKDIR $ARKOUDA_HOME







ENTRYPOINT ["/bin/bash", "-l", "-c", "source /opt/spack-develop/share/spack/setup-env.sh && exec bash"]













#################################################################################3


# ---------------------------
# Create a local in-container Spack repo and override Arkouda package
# ---------------------------

#RUN source spack-develop/share/spack/setup-env.sh && spack repo create /opt/local-spack-repo bears_r_us

#RUN source $SPACK_ROOT/share/spack/setup-env.sh && \
#    spack repo create /opt/local-spack-repo bears_r_us && \
#    spack repo add /opt/local-spack-repo/spack_repo/bears_r_us #&& \
#    #mkdir -p /opt/local-spack-repo/spack_repo/bears_r_us/packages/arkouda && \
#    #cp $(find /root/.spack/package_repos -name package.py | grep /arkouda/package.py) /opt/local-spack-repo/spack_repo/bears_r_us/packages/arkouda/

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
#RUN export ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos -name package.py | grep /arkouda/package.py) && \
#    sed -i 's|depends_on("chapel@[^"]*", type=(.*))|depends_on("chapel@2.4.0:2.4.99 +hdf5 +zmq", type=("build", "link", "run", "test"))|' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i 's|arrow +parquet +snappy +zlib +brotli +bz2 +lz4 +zstd|arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd|' \
#    $ARKOUDA_PACKAGE_FILE


# ---------------------------
# Install Chapel 2.4.0 explicitly (pinned version)
# ---------------------------
#RUN . $SPACK_ROOT/share/spack/setup-env.sh && \
#    spack install chapel@2.4.0 +hdf5 +zmq

# ---------------------------
# Install Arkouda with patched Spack package
# ---------------------------
#RUN source spack-develop/share/spack/setup-env.sh && spack install arrow

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

