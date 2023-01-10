FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 AS build
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential libssl-dev libffi-dev python3-dev gcc
RUN apt-get install -y --no-install-recommends python3 python3.8-venv python3-pip git-all
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

FROM nvidia/cuda:11.3.1-base-ubuntu20.04
WORKDIR /tarmass

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update
RUN apt-get install -y --no-install-recommends unzip python3 git
RUN python3 -c "import easyocr ; easyocr.Reader(['en'])"
RUN python -c "from torch.utils.model_zoo import load_url; load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')"
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
