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
    git gcc gcc-c++ gcc-gfortran make cmake lsof \
    curl tar gzip wget which \
    findutils gawk patch file less hostname unzip \
    diffutils procps-ng libcurl-devel bzip2 \
    ca-certificates && \
    dnf clean all && \
    update-ca-trust

# ---------------------------
# Clone Spack and set environment
# ---------------------------
WORKDIR /opt
RUN git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git spack-develop

ENV SPACK_ROOT=/opt/spack-develop
ENV PATH=$SPACK_ROOT/bin:$PATH
ENV CHPL_GMP=bundled
ENV SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
RUN . /opt/spack-develop/share/spack/setup-env.sh \
 && spack compiler find \
 && spack external find --not-buildable gcc binutils glibc

# ---------------------------
# Create local spack mirror to speed up incremental builds:
# ---------------------------
RUN spack mirror create -d /opt/spack-mirror --dependencies arkouda ^lz4 build_system=cmake
RUN spack mirror add local file:///opt/spack-mirror


# ---------------------------
# Update the arkouda spack package:
# ---------------------------

COPY arkouda-package.py /opt/arkouda-package.py

RUN export ARKOUDA_PACKAGE_FILE=$(find /root/.spack/package_repos/fncqgg4/repos/spack_repo/builtin/packages | grep /arkouda/package.py) \
    &&  mv /opt/arkouda-package.py $ARKOUDA_PACKAGE_FILE  \
    && touch ~/.bashrc.local \
    && echo "export ARKOUDA_PACKAGE_FILE=\"${ARKOUDA_PACKAGE_FILE}\"" >> ~/.bashrc.local


# ---------------------------
# Install Arkouda with patched Spack package
# ---------------------------
RUN source /opt/spack-develop/share/spack/setup-env.sh && spack -d -v install -j $(nproc) arkouda${RELEASE_VERSION}  


###################
#   Install the old way for testing
##################
#RUN dnf install -y python3.11 \
#    python3 python3-devel python3-pip 
    
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
#RUN python3 -m ensurepip --default-pip
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


#RUN source /opt/spack-develop/share/spack/setup-env.sh && \
#    spack load arkouda chapel && \
#    ln -s $(spack location -i arkouda)/bin/arkouda_server arkouda_server


#RUN source /opt/spack-develop/share/spack/setup-env.sh && \
#    python3 -m pip install .[dev] && \ 
#    python3 -m pytest -c pytest.ini --size=100  #--skip_doctest="True"


# ---------------------------
# Finish Up
# ---------------------------
WORKDIR /root

RUN env >> ~/.bashrc.local && echo 'source ~/.bashrc.local' >> ~/.bashrc

CMD ["/bin/bash"]







