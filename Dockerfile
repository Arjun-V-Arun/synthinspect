# The inspection API: exported model + FastAPI, no PyTorch.
#   docker build --build-arg MODEL_DIR=runs/metal_nut/package -t synthinspect .
#   docker run -p 8080:8080 synthinspect
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 \
    SYNTHINSPECT_MODEL_DIR=/app/model SYNTHINSPECT_THREADS=1
WORKDIR /app

# opencv-headless needs libgl-free runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY synthinspect/__init__.py synthinspect/io.py synthinspect/
COPY synthinspect/detect/__init__.py synthinspect/detect/base.py synthinspect/detect/runtime.py synthinspect/detect/
COPY synthinspect/eval/ synthinspect/eval/
COPY synthinspect/xai/__init__.py synthinspect/xai/drise.py synthinspect/xai/
COPY synthinspect/serve/__init__.py synthinspect/serve/app.py synthinspect/serve/monitoring.py synthinspect/serve/

ARG MODEL_DIR=runs/metal_nut/package
COPY ${MODEL_DIR}/config.json model/config.json
COPY ${MODEL_DIR}/model.onnx model/model.onnx
COPY ${MODEL_DIR}/openvino model/openvino

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app
EXPOSE 8080
CMD ["sh", "-c", "uvicorn synthinspect.serve.app:app --host 0.0.0.0 --port ${PORT}"]
