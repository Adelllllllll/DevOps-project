# Base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY data/processed/ data/processed/

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
