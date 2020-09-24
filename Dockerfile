FROM tensorflow/tensorflow:2.2.0-jupyter
# ubuntu 18.04
# 参考：https://colab.research.google.com/drive/1YpSHRBRPBI7cnTkQn1UcVTWEQVbsUm1S

ENV LANG C.UTF-8

RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get -y update && \ 
    apt-get -y upgrade && \
    apt-get install --yes --no-install-recommends \
    libsndfile1 git vim && \
    apt-get -y clean && \  
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p /app
WORKDIR /app
COPY . /app
# COPY ./nltk_data/ /root/nltk_data/

# Packages needed for web server plus scipy...
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install flask flask-cors scipy dataclasses gdown
# download model 需要科学上网，可自行下载
RUN gdown --id {"17Db2R2k8mJmbO5utX80AVK8ssAr-JpCB"} -O mb.melgan-920k.h5 && \
    gdown --id {"1n36vcrEPQ0SyL7wVrYsNVrPiuGhiOF4h"} -O tacotron2-100k.h5 && \
    gdown --id {"1cS8jNEgovxUNuCVQSOM68HujrzMGKXNB"} -O baker_mapper.json

# TensorFlowTTS
RUN cd /app/TensorFlowTTS && \
    git clone https://github.com/TensorSpeech/TensorFlowTTS.git && \
    sed -i "s/tensorflow-gpu/tensorflow-cpu/g" setup.py
    pip install .

EXPOSE 5000

ENTRYPOINT ["python", "/app/app.py"]
