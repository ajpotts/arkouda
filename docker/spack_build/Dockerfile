# docker build command
# cls;cd /home/mwmarde/arkouda-util/Docker && docker build --build-arg RELEASE_VERSION=2025.07.03 -t arkouda_v2025.07.03:0.13 .
# cls;cd /home/mwmarde/arkouda-util/Docker && docker build -t arkouda_v2025.07.03:0.13 .
# cd /home/mwmarde/arkouda-util/Docker &&  docker run -it arkouda_v2025.07.03:0.13 

#docker build   --build-arg RELEASE_VERSION=2025.08.20   -t spack_test:20205.08.20 .

# Use AlmaLinux as the base image
FROM almalinux:9

# Maintainer and metadata
LABEL maintainer="ajpotts@datacraft-solutions.com"
LABEL description="Arkouda development image with Spack and Chapel pinned to compatible versions"

# Arkouda release to be build
ARG RELEASE_VERSION
ENV RELEASE_VERSION=${RELEASE_VERSION:+@${RELEASE_VERSION}}
RUN echo "RELEASE_VERSION: ${RELEASE_VERSION}"

# Set non-interactive installation
ENV TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive

# Use Bash as the shell for all RUN statements
SHELL ["/bin/bash", "-c"]

# ---------------------------
# Add aliases directly to .bashrc (or create one if it doesn't exist)
# ---------------------------
RUN echo "alias cls='clear'" >> /root/.bashrc && \
    echo "alias e='exit'" >> /root/.bashrc && \
    echo "alias h='history'" >> /root/.bashrc && \
    echo "alias ll='ls -alF'" >> /root/.bashrc && \
    echo "alias lrt='ls -lrt --color=auto'" >> /root/.bashrc

# ---------------------------
# Install system dependencies
# --allowerasing helps resolve curl-minimal and other conflicts
# ---------------------------
RUN dnf install -y --allowerasing \
    git gcc gcc-c++ gcc-gfortran make cmake lsof \
    python3 python3-devel python3-pip \
    curl tar gzip wget which \
    findutils gawk patch file less hostname unzip \
    diffutils procps-ng libcurl-devel bzip2 \
    which \
    && dnf clean all



# ---------------------------
# Clone Spack and set environment
# ---------------------------
WORKDIR /opt
RUN git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git spack-develop

ENV SPACK_ROOT=/opt/spack-develop
ENV PATH=$SPACK_ROOT/bin:$PATH
ENV CHPL_GMP=bundled

# ---------------------------
# Initialize Spack and find available compilers
# ---------------------------
# Initialize Spack env and register the system compiler & externals
RUN . spack-develop/share/spack/setup-env.sh \
 && spack compiler find \
 && spack external find --not-buildable gcc binutils glibc \
 && spack config get compilers \
 && spack config get packages



#################################################################################



RUN source spack-develop/share/spack/setup-env.sh && spack compilers

# ---------------------------
# Apply in-place sed patches to the Arkouda package.py
#   1. Pin chapel@2.4.0:2.4.99 +hdf5 +zmq
#   2. Disable lz4 in arrow (~lz4)
# ---------------------------
# Replace full chapel dependency line
#RUN export ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /arkouda/package.py) && \
#    sed -i 's|version("master", branch="master")|version("main", branch="main")|' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2024.12.06", sha256="92ca11319a9fdeeb8879afbd1e0c9c1b1d14aa2496781c1481598963d3c37b46")' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.01.13", sha256="bb53bab92fedf43a47aadd9195eeedebe5f806d85887fa508fb5c69f2a4544ea")' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.07.03", sha256="eb888fac7b0eec6b4f3bfa0bfe14e5c8f15b449286e84c45ba95c44d8cd3917a")' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.08.20", sha256="3e305930905397ff3a7a28a5d8cc2c9adca4194ca7f6ee51f749f427a2dea92c")' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i 's|depends_on("chapel@[^"]*", type=(.*))|depends_on("chapel@2.4.0:2.4.99 +hdf5 +zmq", type=("build", "link", "run", "test"))|' \
#    $ARKOUDA_PACKAGE_FILE && \
#    sed -i 's|arrow +parquet +snappy +zlib +brotli +bz2 +lz4 +zstd|arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd|' \
#    $ARKOUDA_PACKAGE_FILE

RUN echo "TEST4"


RUN source spack-develop/share/spack/setup-env.sh && spack -d -v install glibc@2.34
RUN source spack-develop/share/spack/setup-env.sh && spack -d -v install gcc@11.5.0
RUN source spack-develop/share/spack/setup-env.sh && spack -d -v install ca-certificates-mozilla@2025-05-20
RUN source spack-develop/share/spack/setup-env.sh && spack -d -v install gcc-runtime@11.5.0
RUN source spack-develop/share/spack/setup-env.sh && spack -d -v install boost@1.88.0
RUN source spack-develop/share/spack/setup-env.sh && spack -d -v install gmake@4.4.1


COPY arkouda-package.py /opt/arkouda-package.py

RUN echo "TEST 7" && export ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /arkouda/package.py) \
    &&  mv /opt/arkouda-package.py $ARKOUDA_PACKAGE_FILE  \
    && touch ~/.bashrc.local \
    && echo "export ARKOUDA_PACKAGE_FILE=\"${ARKOUDA_PACKAGE_FILE}\"" >> ~/.bashrc.local


# ---------------------------
# Install Arkouda with patched Spack package
# ---------------------------
ENV SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
RUN dnf install -y gcc gcc-c++ gcc-gfortran make which file findutils ca-certificates && update-ca-trust

















RUN source /opt/spack-develop/share/spack/setup-env.sh && spack -d -v install arkouda${RELEASE_VERSION}



# ---------------------------
# Expose the default arkouda_server port
# ---------------------------
#EXPOSE 5555

# ---------------------------
# Apply in-place sed patches to the py_arkouda/package.py
# ---------------------------
#RUN export ARKOUDA_CLIENT_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /py_arkouda/package.py) && \
#    sed -i 's|version("master", branch="master")|version("main", branch="main")|' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE && \
#    sed -i 's|maintainers("arezaii")|maintainers("ajpotts", "arezaii")|' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2024.12.06", sha256="92ca11319a9fdeeb8879afbd1e0c9c1b1d14aa2496781c1481598963d3c37b46")' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.01.13", sha256="bb53bab92fedf43a47aadd9195eeedebe5f806d85887fa508fb5c69f2a4544ea")' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE && \
#    sed -i '30i \ \ \ \ version("2025.07.03", sha256="eb888fac7b0eec6b4f3bfa0bfe14e5c8f15b449286e84c45ba95c44d8cd3917a")' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE \
#    sed -i '30i \ \ \ \ version("2025.08.20", sha256="3e305930905397ff3a7a28a5d8cc2c9adca4194ca7f6ee51f749f427a2dea92c")' \
#    $ARKOUDA_CLIENT_PACKAGE_FILE

COPY py_arkouda-package.py /opt/py_arkouda-package.py
RUN export ARKOUDA_CLIENT_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /py_arkouda/package.py) && \
    mv /opt/py_arkouda-package.py $ARKOUDA_CLIENT_PACKAGE_FILE \
    && touch ~/.bashrc.local \
    && echo "export ARKOUDA_CLIENT_PACKAGE_FILE=\"${ARKOUDA_CLIENT_PACKAGE_FILE}\"" >> ~/.bashrc.local



###################
#   Install the old way for testing
##################
RUN dnf install -y python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN python3 -m ensurepip --default-pip
RUN git clone https://github.com/Bears-R-Us/arkouda.git

WORKDIR /opt/arkouda

RUN if [ "$RELEASE_VERSION" = "@2025.08.20" ]; then \
        cd /opt/arkouda && git checkout 147a08b2f2479358cd2c4ef0a75650d082a80779 ; \
    elif [ "$RELEASE_VERSION" = "@2025.07.03" ]; then \
        cd /opt/arkouda && git checkout 88f2a83d9c491fd785e1e2f67a0445ed11dd2d98 ; \
    elif [ "$RELEASE_VERSION" = "@2025.01.13" ]; then \
        cd /opt/arkouda && git checkout a3aa4c376e4c79b3342ac3e932dd1a9761462de4 ; \
    elif [ "$RELEASE_VERSION" = "@2024.12.06" ]; then \
        cd /opt/arkouda && git checkout 9eba2ce562ca8435dd3e5896eb2be703e4708e47 ; \
    elif [ "$RELEASE_VERSION" = "@2024.10.02" ]; then \
        cd /opt/arkouda && git checkout a44dd0f4d7b77cb119ea89dd66e8773ca1815848 ; \
    elif [ "$RELEASE_VERSION" = "@2024.06.21" ]; then \
        cd /opt/arkouda && git checkout cf6eeacdecd82b2b61b4f0e9444a5106e4bf3754 ; \
    else \
        git checkout main ; \
    fi

RUN source /opt/spack-develop/share/spack/setup-env.sh && spack load chapel arkouda && ln -s $(spack location -i arkouda)/bin/arkouda_server arkouda_server
RUN python3 -m pip install .[dev]

RUN source /opt/spack-develop/share/spack/setup-env.sh && spack load chapel arkouda && python3 -m pip install .[dev] &&  python3 -m pytest -c pytest.ini --size=100  --skip_doctest="True"


# ---------------------------
# Install arkouda python front end
# Wheel files will be saved here /opt/wheels
# So the project can be installed offline later
# ---------------------------
#RUN dnf -y update && dnf -y install gfortran
#RUN spack compiler find
#RUN spack install py-arkouda ^py-setuptools@68.2.2 ^openblas@0.3.28
#RUN echo "TEST2"
#RUN spack install py-arkouda${RELEASE_VERSION}


# ---------------------------
# Copy a simple test script
# ---------------------------




WORKDIR /root

RUN env >> ~/.bashrc.local && echo 'source ~/.bashrc.local' >> ~/.bashrc

CMD ["/bin/bash"]







