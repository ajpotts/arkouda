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
# Install system dependencies
# --allowerasing helps resolve curl-minimal and other conflicts
# ---------------------------
RUN dnf install -y --allowerasing \
    gcc gcc-c++ gcc-gfortran \
    git  make cmake lsof \
    curl tar gzip wget which \
    findutils gawk patch file less hostname unzip \
    diffutils procps-ng libcurl-devel bzip2 \
    ca-certificates vim && \
    dnf clean all && \
    update-ca-trust


# ---------------------------
# Clone Spack and set environment
# ---------------------------
WORKDIR /opt
RUN git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git spack-develop

ENV SPACK_ROOT=/opt/spack-develop
ENV PATH=$SPACK_ROOT/bin:$PATH
ENV SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
RUN . /opt/spack-develop/share/spack/setup-env.sh \
 && spack compiler find \
 && spack external find --not-buildable gcc binutils glibc

#RUN . /opt/spack-develop/share/spack/setup-env.sh && spack find gcc@12.3.0 || spack install gcc@12.3.0

# ---------------------------
# Update the arkouda spack package:
# ---------------------------

COPY arkouda-package.py /opt/arkouda-package.py

RUN export ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /arkouda/package.py) \
    &&  mv /opt/arkouda-package.py $ARKOUDA_PACKAGE_FILE  \
    && touch ~/.bashrc.local \
    && echo "export ARKOUDA_PACKAGE_FILE=\"${ARKOUDA_PACKAGE_FILE}\"" >> ~/.bashrc.local


# ---------------------------
# Update the py-arkouda spack package:
# ---------------------------

COPY py_arkouda-package.py /opt/py_arkouda-package.py
COPY pytables_package.py /opt/pytables_package.py


RUN export PY_ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /py_arkouda/package.py) \
    &&  mv /opt/py_arkouda-package.py $PY_ARKOUDA_PACKAGE_FILE  \
    && echo "export PY_ARKOUDA_PACKAGE_FILE=\"${PY_ARKOUDA_PACKAGE_FILE}\"" >> ~/.bashrc.local

# ---------------------------
# Install Py-Arkouda with patched Spack package
# ---------------------------

RUN export PY_TABLES_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /py_tables/package.py) \
    &&  mv /opt/pytables_package.py $PY_TABLES_PACKAGE_FILE  \
    && echo "export PY_TABLES_PACKAGE_FILE=\"${PY_TABLES_PACKAGE_FILE}\"" >> ~/.bashrc.local


#####################

# 1) Create & activate an environment
RUN source /opt/spack-develop/share/spack/setup-env.sh && spack env create ark-env \
    &&  spack env activate -p ark-env \
    # 2) Tell Spack to unify deps across all roots in this env
    &&  spack config add "concretizer:unify:true" \
    # 3) Add BOTH roots (same release tag here as an example)
    &&  spack add arkouda${RELEASE_VERSION}  \
    &&  spack add py-arkouda${RELEASE_VERSION}  \
    # 4) Check the unified solve
    &&  spack concretize -f  \
    &&  spack spec -Il
    
    
## 5) Install (one build plan, shared deps)
RUN source /opt/spack-develop/share/spack/setup-env.sh \
    &&  spack env activate -p ark-env \   
    &&  spack -d -v  install -j $(nproc)




###################
#   Install the old way for testing
##################
RUN git clone https://github.com/Bears-R-Us/arkouda.git

WORKDIR /opt/arkouda
# after all spack installs
ENV PATH="/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"


RUN cd /opt/arkouda && \
    if [ "$RELEASE_VERSION" = "main" ]; then \
        git checkout main ; \
    else \
        git fetch --tags && \
        git checkout $(git rev-list -n 1 "v${RELEASE_VERSION#@}") ; \
    fi



# ensure bash in RUNs (optional but nice)
SHELL ["/bin/bash", "-lc"]

RUN . /opt/spack-develop/share/spack/setup-env.sh && \
    spack -e ark-env env status >/dev/null && \
    ln -sf "$(spack -e ark-env location -i arkouda)/bin/arkouda_server" ./arkouda_server



RUN source /opt/spack-develop/share/spack/setup-env.sh && \
    spack env activate -p ark-env && \
    spack load python chapel py-arkouda c-blosc2 && \
    spack add py-pytest && spack install py-pytest && \
    python3 -m ensurepip --default-pip && \ 
    export LD_LIBRARY_PATH=$(spack find -p c-blosc2 | awk '/c-blosc2/{print $2}')/lib64:$LD_LIBRARY_PATH  && \
    # Install the plugin into THAT interpreter’s site-packages
    python3 -m pip install --upgrade pip && \ 
    python3 -m pip install pytest-env pytest-subtests && \ 
    python --version && \
    python3 -m pytest -c pytest.ini --size=100  --skip_doctest="True"



# ---------------------------
# Finish Up
# ---------------------------
WORKDIR /root

RUN echo 'source ~/.bashrc.local' >> ~/.bashrc && echo 'source /opt/spack-develop/share/spack/setup-env.sh' >> ~/.bashrc.local

CMD ["/bin/bash"]







