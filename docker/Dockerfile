FROM ubuntu:16.04
#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

############################################# Proxy definition
# This is necessary and specific to CentraleSupelec
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}
ENV no_proxy ${no_proxy}

# Setup User to match Host User, and give superuser permissions
# 
ARG USER_ID=0
ARG GROUP_ID=0
RUN groupadd -g ${GROUP_ID} code_executor && useradd -m code_executor -u ${USER_ID} -g ${GROUP_ID}
###########################

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libavcodec-dev \
    libavformat-dev \
    libboost1.58-all-dev \
    libgphoto2 \
    libgtk2.0-dev \
    libjasper-dev \
    libjpeg-dev \
    libpng-dev \
    libopencv-dev \
    libopenmpi-dev \
    libtbb2 \
    libtbb-dev \
    libswscale-dev \
    libtiff-dev \
    libdc1394-22-dev \
    pkg-config \
    python-dev \
    python-numpy \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-setuptools \
    python3-scipy \
    unzip \
    vim \
    wget \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*


ENV WS /home/ws
WORKDIR $WS


ENV INSTALL_DIR /home/ws/install
WORKDIR $INSTALL_DIR

# Install opencv
RUN wget https://github.com/opencv/opencv/archive/3.2.0.zip && \
unzip 3.2.0.zip && \
rm 3.2.0.zip

WORKDIR /home/ws/install/opencv-3.2.0/

RUN mkdir build && cd build && \
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
make -j"$(nproc)" && \
sudo make install


# python packages
RUN for req in $(cat requirements.txt) pydot; do pip3 install $req; done 

##ENV CODE_DIR /root/code
##ENV VENV /root/venv
##
##RUN \
##    pip install virtualenv && \
##    virtualenv $VENV --python=python3 && \
##    . $VENV/bin/activate && \
##    pip install --upgrade pip
#
##ENV PATH=$VENV/bin:$PATH
#ENV PATH=$WS/bin:$PATH
#
#COPY . $WS
##WORKDIR $CODE_DIR
#
## Clean up pycache and pyc files
#RUN rm -rf __pycache__ && \
#    find . -name "*.pyc" -delete && \
#    pip3 install --upgrade pip==9.0.1 && \
#    for req in $(cat requirements.txt) pydot; do pip3 install $req; done 
##&& \
##    pip install -e .[test]
#
#ENV CUDA_VISIBLE_DEVICES 0
##ENV PYTHONPATH $CODE_DIR:$CODE_DIR/slim:$PYTHONPATH
#ENV PYTHONPATH $WS:$PYTHONPATH
CMD /bin/bash
