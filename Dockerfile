# Use Fedora-based Python image for better compatibility with Fedora host
FROM fedora:38

# Set working directory
WORKDIR /app

# Install Python and essential system dependencies
RUN dnf update -y && \
    dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    cmake \
    make \
    git \
    wget \
    curl \
    libjpeg-turbo-devel \
    libpng-devel \
    libtiff-devel \
    pkg-config \
    && dnf clean all

# Create symbolic links for Python (Fedora uses python3 by default)
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python dependencies
COPY requirements-docker.txt /app/requirements.txt

# Update requirements.txt to work with Docker environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/media/profile_pics
RUN mkdir -p /app/logs

# Set Python path
ENV PYTHONPATH=/app/StaffFaceRecognition:/app/StaffFaceRecognition/backend

# Set environment variables
ENV DJANGO_SETTINGS_MODULE=StaffFaceRecognition.settings
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 5600

# Copy and make start script executable
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
