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

# source the spack-develop setup-env
RUN source spack-develop/share/spack/setup-env.sh && spack compiler find && spack spec zlib

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




#SHELL ["/bin/bash", "-l", "-c", "source spack-develop/share/spack/setup-env.sh && bash -c"]


RUN source spack-develop/share/spack/setup-env.sh && spack compilers

#spack edit arrow
# add
#list_url = "https://dlcdn.apache.org/arrow/" #mwm
#version("20.0.0", sha256="67e31a4f46528634b8c3cbb0dc60ac8f85859d906b400d83d0b6f732b0c5b0e3") #mwm

#spack install arrow  #arrow@19.0.1
#spack install chapel #chapel@2.4.0


## Stop here











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



