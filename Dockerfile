FROM nvcr.io/nvidia/pytorch:20.12-py3
WORKDIR /biomass_calculation/
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt