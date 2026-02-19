FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Create output and sandbox directories
RUN mkdir -p /app/output /app/sandbox

# Data will be mounted as a volume (read-only)
# Output and sandbox will be mounted as volumes (read-write)

ENTRYPOINT ["python", "main.py"]
