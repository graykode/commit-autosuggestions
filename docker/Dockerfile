FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
LABEL maintainer="nlkey2022@gmail.com"

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install curl python3-pip git \
 && rm -rf /var/lib/apt/lists/*

ARG PYTORCH_WHEEL="https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp36-cp36m-linux_x86_64.whl"
ARG ADDED_MODEL="1YrkwfM-0VBCJaa9NYaXUQPODdGPsmQY4"
ARG DIFF_MODEL="1--gcVVix92_Fp75A-mWH0pJS0ahlni5m"

RUN git clone https://github.com/graykode/commit-autosuggestions.git /app/commit-autosuggestions \
    && cd /app/commit-autosuggestions && python3 setup.py install

WORKDIR /app/commit-autosuggestions

RUN pip3 install ${PYTORCH_WHEEL} gdown
RUN gdown https://drive.google.com/uc?id=${ADDED_MODEL} -O weight/added/
RUN gdown https://drive.google.com/uc?id=${DIFF_MODEL} -O weight/diff/

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "app.py"]
