# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads chroma_for_policychatbot

# Expose port
EXPOSE 8000

# Set environment variables
ENV ENVIRONMENT=production
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application
CMD ["uvicorn", "api_backend:app", "--host", "0.0.0.0", "--port", "8000"]
