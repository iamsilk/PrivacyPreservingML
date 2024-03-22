FROM ubuntu

# Install build requirements
RUN apt-get update && apt-get install -y \
    cmake \
    libboost-all-dev \
    libprotobuf-dev \
    protobuf-compiler \
    clang \
    git \
    python3 \
    python3-pip

RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100

# Install numpy
RUN python3 -m pip install numpy

# Install Microsoft SEAL
RUN cd / && \
    git clone -b v3.6.6 https://github.com/microsoft/SEAL.git && \
    cd SEAL && \
    cmake -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=OFF . && \
    make -j && \
    make install

# Install EVA
RUN cd / && \
    git clone https://github.com/microsoft/EVA.git && \
    cd EVA && \
    git submodule update --init && \
    cmake . && \
    make -j && \
    python3 -m pip install -e python/

# Setup app environment
WORKDIR /app
COPY ckks_example.py .

# Run the app
CMD ["python3", "ckks_example.py"]