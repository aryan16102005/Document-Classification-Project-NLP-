FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker cache efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Create necessary directories
RUN mkdir -p data/sample artifacts/models artifacts/plots logs

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
