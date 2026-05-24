# Use official slim Python runtime
FROM python:3.9-slim

# Install system dependencies needed for some Python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set default port to 7860 (Hugging Face standard)
ENV PORT=7860

# Expose port
EXPOSE 7860

# Start app using Gunicorn on 0.0.0.0:7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
