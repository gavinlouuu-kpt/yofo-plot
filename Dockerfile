# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create uploads and logs directories
RUN mkdir -p uploads logs

# Expose ports
EXPOSE 5000 5006

# Create gunicorn configuration file
COPY gunicorn.conf.py .

# Set entrypoint to use Gunicorn
ENTRYPOINT ["gunicorn", "--config", "gunicorn.conf.py", "web_plot:create_app()"]
