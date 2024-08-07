ARG PYTORCH="2.3.0"
ARG CUDA="12.1"
ARG CUDNN="8"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# update NVIDIA repo key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
ARG distro=ubuntu2204
ARG arch=x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/3bf863cc.pub

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir faster-whisper==1.0.2 "certifi>=2024.07.04"
RUN pip install --no-cache-dir \
        "simple-file-poller>=0.0.9" \
        redis \
        "redis-docker-harness==0.0.4"

# original path, which doesn't contain any CUDA libraries:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda-11/lib64
ENV PYTHONPATH=/opt/faster-whisper

COPY bash.bashrc /etc/bash.bashrc
COPY *.py /opt/faster-whisper/
COPY fw* /usr/bin/
