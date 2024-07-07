ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

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

ENV PYTHONPATH=/opt/faster-whisper

COPY bash.bashrc /etc/bash.bashrc
COPY *.py /opt/faster-whisper/
COPY fw* /usr/bin/
