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
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
	&& pip install -r requirements

CMD ["bash"]
