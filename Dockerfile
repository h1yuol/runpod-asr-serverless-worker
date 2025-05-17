# Using a placeholder base image from RunPod or a community-verified one
# Example: runpod/pytorch:py312-cu121-torch251-ubuntu22.04 or similar
# Needs to have Python 3.11, CUDA (e.g., 12.1), and PyTorch pre-installed.
# Base image tag - using runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 which was successful
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
# - ffmpeg for audio processing
# - git for whisperx dependencies that might pull from git (e.g. pyannote)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# The base image should have pip. Ensure Python version matches base image (py3.11).
# Adding PyTorch index as an --extra-index-url so PyPI is still primary for other packages.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the handler script and any other necessary files
COPY runpod_handler.py .
# If you have other local modules or files, copy them too:
# COPY ./my_module /app/my_module

# Set environment variables that might be useful (optional)
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Command to run the RunPod handler
# The -u flag ensures that prints are sent straight to stdout without being buffered
CMD ["python", "-u", "runpod_handler.py"] 