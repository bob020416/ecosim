# FROM python:3.11
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get -y install build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \ 
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    curl \
    wget \
    tk-dev \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get purge -y imagemagick imagemagick-6-common 

RUN apt-get install -y ffmpeg # for logging video

RUN cd /usr/src \
    && wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz \
    && tar -xzf Python-3.11.9.tgz \
    && cd Python-3.11.9 \
    && ./configure --enable-optimizations \
    && make install

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
ADD ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install torch_geometric
RUN pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
RUN pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.4

CMD ["/bin/bash"]

