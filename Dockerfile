# Use official slim Python runtime
FROM python:3.9-slim

# Install system dependencies needed for some Python builds and C++ tree boosters (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set default port to 7860 (Hugging Face standard, Render will override dynamically via $PORT)
ENV PORT=7860

# Expose port
EXPOSE 7860

# Start app dynamically binding to $PORT using sh shell expansion
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:$PORT --timeout 120 app:app"]
