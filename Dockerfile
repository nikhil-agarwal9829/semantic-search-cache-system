FROM python:3.11-slim

WORKDIR /app

# Install system deps that some Python packages (like numpy/scipy) benefit from.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the image.
COPY . .

# Expose the port that Uvicorn will listen on.
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Start the FastAPI app with Uvicorn.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

