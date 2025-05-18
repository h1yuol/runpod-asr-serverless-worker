FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install --no-cache-dir -r requirements.txt

# Final stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy handler code
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# Run the handler
CMD ["python3", "handler.py"] 