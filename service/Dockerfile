# service/Dockerfile
FROM python:3.11-slim

# environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=600

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple/ --retries 5 --timeout 200 --prefer-binary --progress-bar off

RUN addgroup --system appgroup && \
    adduser --ingroup appgroup --home /home/appuser --disabled-password --gecos "" appuser
RUN mkdir -p /app && chown -R appuser:appgroup /app

USER appuser

COPY app.py .
COPY agents.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
