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

#   Example build
# docker build --build-arg RELEASE_VERSION=2025.09.30 -t spack_test:2025.09.30 .

# ---------------------------
# Install system dependencies
# --allowerasing helps resolve curl-minimal and other conflicts
# ---------------------------
RUN dnf install -y --allowerasing \
    git \
    make \
    patch \
    file \
    which \
    findutils \
    tar \
    ca-certificates vim && \
    dnf clean all && \
    update-ca-trust

RUN dnf install -y zlib-devel xz xz-devel

#RUN dnf install -y 
#    git \
#    tar \
#    bzip2 bzip2-devel \
#    lbzip2 \
#    xz xz-devel \
#    gzip \
#    unzip \
#    patch \
#    make \
#    which \
#    findutils \
#    && dnf clean all


RUN dnf install -y dnf-plugins-core && \
    dnf config-manager --set-enabled crb && \
    dnf install -y epel-release && \
    dnf install -y gcc-toolset-13 gcc-toolset-13-gcc gcc-toolset-13-gcc-c++


ENV TOOLSET_ROOT=/opt/rh/gcc-toolset-13/root/usr
ENV PATH="${TOOLSET_ROOT}/bin:${PATH}"
#ENV LD_LIBRARY_PATH="${TOOLSET_ROOT}/lib64:${TOOLSET_ROOT}/lib:${LD_LIBRARY_PATH}"


# ---------------------------
# Clone Spack and set environment
# ---------------------------
WORKDIR /opt
RUN git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git spack-develop

ENV SPACK_ROOT=/opt/spack-develop
ENV PATH=$SPACK_ROOT/bin:$PATH
ENV SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
RUN . /opt/spack-develop/share/spack/setup-env.sh \
 && spack compiler find 
RUN . /opt/spack-develop/share/spack/setup-env.sh \
 && spack compiler find \
 && spack external find --not-buildable gcc binutils glibc

#RUN . /opt/spack-develop/share/spack/setup-env.sh && spack find gcc@12.3.0 || spack install gcc@12.3.0


# ---------------------------
# Compute builtin dir:
# ---------------------------

RUN set -euo pipefail; \
    BUILTIN_PKGS="$(find /root/.spack/package_repos -type d -path '*/repos/spack_repo/builtin/packages' -print -quit)"; \
    test -n "$BUILTIN_PKGS"; \
    echo "export SPACK_BUILTIN_PACKAGES=\"$BUILTIN_PKGS\"" >> ~/.bashrc.local

#####################


    #   add the wrong gcc version to reproduce error
    #&&  spack add gcc@13.2.0 \

# 1) Create & activate an environment
RUN source /opt/spack-develop/share/spack/setup-env.sh && spack env create ark-env \
    &&  spack env activate -p ark-env \
    # 2) Tell Spack to unify deps across all roots in this env
    &&  spack config add "concretizer:unify:true" \
    # 3) Add BOTH roots (same release tag here as an example)
    &&  spack add arkouda${RELEASE_VERSION}  \
    &&  spack add py-arkouda${RELEASE_VERSION}
    # 4) Check the unified solve
    #&&  spack concretize -f  \
    #&&  spack spec -Il
    

    
# 5) Install (one build plan, shared deps)
#RUN source /opt/spack-develop/share/spack/setup-env.sh \
#    &&  spack env activate -p ark-env \
#    &&  spack -d -v  install -j $(nproc)

#spack add chapel@2.5.0 %gcc@13.3.1
#spack install -v chapel@2.5.0 %gcc@13.3.1




###################
#   Install the old way for testing
##################
#RUN git clone https://github.com/Bears-R-Us/arkouda.git

#WORKDIR /opt/arkouda
# after all spack installs
#ENV PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"


#RUN cd /opt/arkouda && \
#    if [ "$RELEASE_VERSION" = "main" ]; then \
#        git checkout main ; \
#    else \
#        git fetch --tags && \
#        git checkout $(git rev-list -n 1 "v${RELEASE_VERSION#@}") ; \
#    fi



# ensure bash in RUNs (optional but nice)
#SHELL ["/bin/bash", "-lc"]

#RUN . /opt/spack-develop/share/spack/setup-env.sh && \
#    spack -e ark-env env status >/dev/null && \
#    ln -sf "$(spack -e ark-env location -i arkouda)/bin/arkouda_server" ./arkouda_server



#RUN source /opt/spack-develop/share/spack/setup-env.sh && \
#    spack env activate -p ark-env && \
#    spack load python chapel py-arkouda c-blosc2 && \
#    spack add py-pytest && spack install py-pytest && \
#    python3 -m ensurepip --default-pip && \ 
#    export LD_LIBRARY_PATH=$(spack find -p c-blosc2 | awk '/c-blosc2/{print $2}')/lib64:$LD_LIBRARY_PATH  && \
#    # Install the plugin into THAT interpreter’s site-packages
#    python3 -m pip install --upgrade pip && \ 
#    python3 -m pip install pytest-env pytest-subtests && \ 
#    python --version && \
#    python3 -m pytest -c pytest.ini --size=100  --skip_doctest="True"



# ---------------------------
# Finish Up
# ---------------------------
WORKDIR /root

RUN echo 'source ~/.bashrc.local' >> ~/.bashrc && echo 'source /opt/spack-develop/share/spack/setup-env.sh' >> ~/.bashrc.local

CMD ["/bin/bash"]







