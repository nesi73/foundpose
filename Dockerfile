# Use image Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    python3-pip \
    git \
    python3-setuptools \
    python3-wheel \
    pkg-config \
    libcairo2-dev \ 
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip \
    && pip3 install meson==0.63.3 \
	&& pip3 install --no-cache-dir -r requirements.txt

CMD ["bash"]
