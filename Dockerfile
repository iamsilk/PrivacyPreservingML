FROM ubuntu

# Install build requirements
RUN apt-get update && apt-get install -y \
    cmake \
    libboost-all-dev \
    libprotobuf-dev \
    protobuf-compiler \
    clang

RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100

RUN apt-get install -y \
    git

# Install Microsoft SEAL
RUN cd / && \
    git clone -b v3.6.6 https://github.com/microsoft/SEAL.git && \
    cd SEAL && \
    cmake -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF . && \
    make -j && \
    make install

RUN apt-get install -y \
    python3 \
    python3-pip

# Install EVA
RUN cd / && \
    git clone https://github.com/microsoft/EVA.git && \
    cd EVA && \
    git submodule update --init && \
    cmake . && \
    make -j && \
    python3 -m pip install -e python/

# Install numpy
RUN python3 -m pip install numpy

# Setup app environment
WORKDIR /app
COPY ckks_example.py .

# Run the app
CMD ["python3", "ckks_example.py"]