FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY deployment/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

COPY deployment /app/deployment
COPY models/LeafFusionNet.keras /app/models/LeafFusionNet.keras
COPY results/reports/class_names.json /app/results/reports/class_names.json
COPY results/reports/model_metadata.json /app/results/reports/model_metadata.json

EXPOSE 8000

CMD ["sh", "-c", "uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
