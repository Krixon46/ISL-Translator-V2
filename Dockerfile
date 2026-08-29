FROM python:3.11-slim-bookworm

# Install system dependencies required by MediaPipe and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend ./backend

# Start FastAPI
CMD ["uvicorn", "server:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "10000"]