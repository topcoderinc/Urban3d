# FROM nvidia/cuda:8.0-devel-ubuntu16.04
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

## Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        cmake \
        git \
        wget \
        nano \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy \
        python-opencv \
        software-properties-common

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH
RUN /opt/conda/bin/pip install keras==2.0.8 tensorflow-gpu==1.2.1 pyyaml opencv-python h5py tifffile dask
RUN /opt/conda/bin/pip install --upgrade scikit-learn

WORKDIR /home/zfturbo/project/

# copy entire directory where docker file is into docker container at /work
COPY . /home/zfturbo/project/

RUN chmod 777 train.sh
RUN chmod 777 test.sh