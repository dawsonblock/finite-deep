
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal build deps and clean up
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only metadata first to leverage docker layer caching
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# Now copy the rest of the source (non-editable install already done)
COPY . /app

ENV DEEPSEEK_URL=https://api.deepseek.com/chat/completions \
    DEEPSEEK_MODEL=deepseek-chat \
    DEEPSEEK_TIMEOUT_S=60
EXPOSE 8080
CMD ["python","server.py"]
