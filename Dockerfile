FROM nvcr.io/nvidia/pytorch:21.10-py3
RUN apt-get update -y && apt-get install -y \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender-dev \
libgl1-mesa-dev

WORKDIR /biomass_calculation/
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python -c "import easyocr ; easyocr.Reader(['en'])"
RUN python -c "from torch.utils.model_zoo import load_url; load_url('https://download.pytorch.org/models/resnet50-0676ba61.pth')"