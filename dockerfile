# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (with torch packages removed from requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support using the Linux command
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Copy project files
COPY . .

# Create a directory for model storage
RUN mkdir -p /app/medical_rag_model

# Expose port
EXPOSE 5000

# Run the application with Uvicorn
CMD ["python", "app.py"]