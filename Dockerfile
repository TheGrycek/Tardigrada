FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 AS build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils build-essential pkg-config libssl-dev libffi-dev libxcb-xinerama0 git \
    python3.8 python3-dev python3.8-venv python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM nvidia/cuda:11.3.1-base-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /tarmass

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/tarmass/src"

RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils python3-opencv unzip python3 libxkbcommon-x11-0 && \
    python3 -c "import easyocr ; easyocr.Reader(['en'])" && \
    python -c "from torch.utils.model_zoo import load_url; load_url('https://download.pytorch.org/models/resnet50-11ad3fa6.pth')" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
