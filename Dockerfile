# Use a CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils \
    git curl ca-certificates build-essential \
    libgl1-mesa-glx libglib2.0-0 \
 && ln -s /usr/bin/python3.10 /usr/bin/python \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
 && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Clone your object detection repo (🔁 replace with your repo)
RUN git clone https://github.com/basaanithanaveenkumar/object-detection-BBD.git .
# Copy entire local repository into container
COPY *.py ./
COPY scripts/*.py ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# # download the dataset
# RUN  mkdir -p data
# CMD ["python" "scripts/download_dataset.py"]

#sudo docker run -it --rm --gpus all object-detection
