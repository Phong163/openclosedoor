# Jetson Nano JetPack 4.6.1
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Asia/Ho_Chi_Minh

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel
# Install ONNX Runtime for Jetson
RUN wget https://nvidia.box.com/shared/static/8sc6j25orjcpl6vhq3a4ir8v219fglng.whl \
-O onnxruntime_gpu-1.6.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install onnxruntime_gpu-1.6.0-cp36-cp36m-linux_aarch64.whl
# Core python deps
RUN python3 -m pip install --no-cache-dir \
    numpy==1.19.4 \
    cython

# install librdkafka
RUN git clone https://github.com/edenhill/librdkafka.git && \
    cd librdkafka && \
    git checkout v1.9.0 && \
    ./configure && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Kafka python client
RUN python3 -m pip install confluent-kafka==1.9.0
RUN python3 -m pip install pytz

# Copy requirements
# COPY requirements.txt .

# # Install project dependencies
# RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Schedule script
COPY run_schedule.sh /run_schedule.sh
RUN chmod +x /run_schedule.sh

ENTRYPOINT ["/run_schedule.sh"]