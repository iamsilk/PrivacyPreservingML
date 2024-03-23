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

# Install tensorflow
RUN python3 -m pip install tensorflow

# Setup app environment
WORKDIR /app
COPY dataset/test/ /app/dataset/test
COPY model/ /app/model
COPY src/ /app

# Run the app

# CKKS Example
# CMD ["python3", "/app/ckks_example.py"]

# Predict
# CMD ["python3", "/app/main.py", "predict", "--model", "/app/model/model.h5", "--vocab", "/app/model/vocab.json", "--text-file", "/app/dataset/test/spam/spam_10.txt"]

# Privacy Predict
CMD ["python3", "/app/main.py", "privacy-predict", "--model", "/app/model/model.h5", "--vocab", "/app/model/vocab.json", "--text-file", "/app/dataset/test/spam/spam_10.txt"]