FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# System dependencies (useful for the shell tool)
# procps => provides `ps` and `pkill` (the slim image has neither)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
