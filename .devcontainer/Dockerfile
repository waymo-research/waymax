# Use a CUDA image with Python 3.10 for compatibility with Waymax
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils curl && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Copy requirements and install dependencies using pip3.10
COPY requirements.txt /tmp/requirements.txt
RUN pip3.10 install --upgrade pip && pip3.10 install -r /tmp/requirements.txt

# Copy the Waymax repository files into the container
COPY . /app
WORKDIR /app

# Install Waymax dependencies
RUN pip3.10 install -e .

# Install JAX with CUDA support
RUN pip3.10 install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Expose any necessary ports, e.g., 8888 if running Jupyter Notebook
EXPOSE 8888

# Default command
CMD ["python3", "-m", "waymax.main"]