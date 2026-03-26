# Multi-stage Dockerfile for multitask-perception
# Optimized for CUDA 11.8 and PyTorch 2.2

# Stage 1: Builder
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel AS builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.0

# Copy dependency files
COPY pyproject.toml ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root --no-interaction

# Stage 2: Runtime
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies for OpenCV and other libs
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy project files
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/pretrained
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/outputs /workspace/pretrained

# Expose ports for TensorBoard and Jupyter
EXPOSE 6006 8888

# Default command
CMD ["python", "src/multitask_perception/tools/train.py", "--help"]
