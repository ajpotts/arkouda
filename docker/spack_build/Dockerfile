FROM almalinux:9.0

# Set user to root
USER root


WORKDIR /opt


RUN dnf groupinstall -y "Development Tools"
RUN dnf install -y git #gcc

#   These ones are for debugging and can be removed in the final version:
RUN dnf install -y vim

# spack-develop-setup-Amanda

# Start with a clean environment
#cd; rm -rf .spack .cache /tmp/mwmarde

# remove sym links as needed
#rm -f spack-develop spack-packages

# clone the spack-develop repo from https://github.com/spack/spack
RUN git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git spack-develop-$(date +%Y-%m-%d)
# create a symbolic link
RUN ln -s spack-develop-$(date +%Y-%m-%d) spack-develop

# install libelf, a library that provide an API for reading, writing and manipulating
# ELF (Executable and Linkable Format) files.
RUN spack-develop/bin/spack install libelf


#RUN spack compiler find
# bootstrapping clingo
#RUN spack spec zlib

## disable the bootstrap of clingo
RUN source spack-develop/share/spack/setup-env.sh && spack bootstrap disable github-actions-v0.6
RUN source spack-develop/share/spack/setup-env.sh && spack bootstrap disable github-actions-v0.5

# verify the active bootstrap repositories were disabled
RUN source spack-develop/share/spack/setup-env.sh && spack bootstrap list

# install a compiler that has fortran
RUN source spack-develop/share/spack/setup-env.sh && spack install gcc@11.5.0 cmake@3.31:
#spack compilers

RUN source spack-develop/share/spack/setup-env.sh && spack compiler find && spack spec zlib



RUN source spack-develop/share/spack/setup-env.sh && spack compilers

#spack edit arrow
# add
#list_url = "https://dlcdn.apache.org/arrow/" #mwm
#version("20.0.0", sha256="67e31a4f46528634b8c3cbb0dc60ac8f85859d906b400d83d0b6f732b0c5b0e3") #mwm

RUN source spack-develop/share/spack/setup-env.sh && spack install arrow  #arrow@19.0.1


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








RUN source spack-develop/share/spack/setup-env.sh && spack install chapel #chapel@2.4.0

# The path to chpl is found using:
# find $(spack location -i chapel) -type f -name chpl
ENV PATH="/opt/spack-develop-2025-07-02/opt/spack/linux-skylake/chapel-2.5.0-rlqgyjmqp5f5y72rqqnqdyr2pyaj2p2f/bin:${PATH}"
ENV CHPL_GMP=bundled


#ENV SPACK_ADD_CMAKE_ARGS="-DLZ4_LIBRARY=/opt/spack-develop-2025-07-02/opt/spack/linux-skylake/lz4-1.10.0-*/lib/liblz4.so -DLZ4_INCLUDE_DIR=/opt/spack-develop-2025-07-02/opt/spack/linux-skylake/lz4-1.10.0-*/include"
#RUN source spack-develop/share/spack/setup-env.sh && spack install arrow $SPACK_ADD_CMAKE_ARGS
# Manually resolve the LZ4 directory with shell globbing
#RUN source spack-develop/share/spack/setup-env.sh && \
#    LZ4_DIR=$(find /opt/spack-develop-2025-07-02/opt/spack/linux-skylake/ -type d -name 'lz4-1.10.0-*' | head -n1) && \
#    SPACK_CMAKE_ARGS="-DLZ4_LIBRARY=${LZ4_DIR}/lib/liblz4.so -DLZ4_INCLUDE_DIR=${LZ4_DIR}/include" \
#    spack install arrow

RUN source spack-develop/share/spack/setup-env.sh && \
    spack uninstall --force arrow || true 
    
RUN source spack-develop/share/spack/setup-env.sh && \
    spack install arrow~lz4



RUN source spack-develop/share/spack/setup-env.sh && \
    spack uninstall --force arkouda || true 
    #&& \
    #spack install arkouda

#RUN source spack-develop/share/spack/setup-env.sh && \
#    spack gc -y && \
#    spack clean --all && \
#    spack install --fresh arkouda

#RUN source spack-develop/share/spack/setup-env.sh && \
#    spack build-env arkouda bash


#RUN source spack-develop/share/spack/setup-env.sh && \
#    spack spec arkouda && \
#    spack install --fresh arkouda

#RUN source spack-develop/share/spack/setup-env.sh && \
#    spack -d install arkouda || (echo "===== Arkouda install failed =====" && \
#    tail -n 1000 ~/.spack/spack-build-out.txt 2>/dev/null || true && \
#    tail -n 1000 ~/.spack/spack-build-env.txt 2>/dev/null || true && \
#    tail -n 1000 ~/.spack/errors/* 2>/dev/null || true && exit 1)

#RUN source spack-develop/share/spack/setup-env.sh && spack install arkouda

## Stop here

RUN source spack-develop/share/spack/setup-env.sh && \
    spack install thrift+openssl flex bison py-pip py-setuptools py-six py-wheel libidn2 libunistring

#   lz4 location found with:
#   spack location -i lz4

# This assumes `spack` is already initialized earlier
RUN mkdir -p /root/.spack/linux && \
    echo "packages:" > /root/.spack/linux/packages.yaml && \
    echo "  arrow:" >> /root/.spack/linux/packages.yaml && \
    echo "    cmake_args:" >> /root/.spack/linux/packages.yaml && \
    echo "      - -DLZ4_LIB=$(spack location -i lz4)/lib/liblz4.so" >> /root/.spack/linux/packages.yaml

RUN source spack-develop/share/spack/setup-env.sh && spack install arrow



RUN mkdir -p /root/.spack/linux && \
    cat <<EOF > /root/.spack/linux/packages.yaml
packages:
  arrow:
    cmake_args:
      - -DLZ4_LIB=$(spack location -i lz4)/lib/liblz4.so
      - -DLZ4_INCLUDE_DIR=$(spack location -i lz4)/include
EOF



RUN source spack-develop/share/spack/setup-env.sh && \
    spack install arkouda









#   Set up environment variables
#COPY bashrc.chpl /root/
#RUN echo 'source ~/.bashrc.chpl' >> ~/.bashrc && mv /root/bashrc.chpl /root/.bashrc.chpl >>  ~/.bashrc.chpl

#  Install dependencies
#RUN dnf update -y && dnf install -y ca-certificates wget python3-pip && dnf update -y && dnf -y upgrade 
#RUN dnf install -y gcc gcc-c++ m4 perl python3.12 python3-devel bash make gawk git cmake which diffutils llvm-devel clang clang-devel libcurl-devel

#   Download Chapel source
#RUN wget https://github.com/chapel-lang/chapel/releases/download/2.3.0/chapel-2.3.0.tar.gz && tar -xvf chapel-2.3.0.tar.gz

#   Install Chapel
#RUN source ~/.bashrc && cd $CHPL_HOME && make && chpl --version

# install chapel-py
#RUN source ~/.bashrc && cd  $CHPL_HOME  && make chapel-py-venv

#WORKDIR /root



ENTRYPOINT ["/bin/bash", "-l", "-c", "source spack-develop/share/spack/setup-env.sh && exec bash"]



