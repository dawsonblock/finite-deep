
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e . fastapi uvicorn
ENV DEEPSEEK_URL=https://api.deepseek.com/chat/completions     DEEPSEEK_MODEL=deepseek-chat     DEEPSEEK_TIMEOUT_S=60
EXPOSE 8080
CMD ["python","server.py"]
