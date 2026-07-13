# LeafFusionNet Deployment API

A production-ready FastAPI service for serving the trained LeafFusionNet plant disease classifier. The deployment code is isolated in this directory and reads the existing model artifacts without modifying the training or inference pipeline.

## Requirements

- Python 3.11
- `models/LeafFusionNet.keras`
- `results/reports/class_names.json`
- `results/reports/model_metadata.json`

Run commands below from the repository root unless stated otherwise.

## Install and run locally

Create and activate a virtual environment, then install only the backend dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r deployment/requirements.txt
```

Start the API:

```powershell
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

For auto-reload during development:

```powershell
uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger UI is available at `http://localhost:8000/docs`; ReDoc is at `http://localhost:8000/redoc`.

## Configuration

The model is loaded once during FastAPI startup and retained in memory for all requests. Uploads are limited to 10 MB by default. Set `MAX_UPLOAD_BYTES` to change that limit.

Vercel preview and production origins are allowed automatically. For a custom frontend domain, set a comma-separated `CORS_ORIGINS` value, for example:

```powershell
$env:CORS_ORIGINS="https://your-app.vercel.app,https://www.example.com"
```

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Returns service identity and running status. |
| GET | `/health` | Reports model loading health. |
| GET | `/model-info` | Returns `model_metadata.json`. |
| GET | `/classes` | Returns all ordered class names. |
| POST | `/predict` | Predicts a JPG, JPEG, or PNG in multipart field `file`. |

Example prediction request:

```powershell
curl.exe -X POST "http://localhost:8000/predict" -F "file=@C:\path\to\leaf.jpg"
```

`/predict` returns the best class, confidence as a percentage, the top five ranked classes, and inference time in milliseconds. Empty uploads, invalid images, and unsupported formats return meaningful JSON errors.

## Docker

Build from the repository root so Docker can include the model artifacts:

```powershell
docker build -f deployment/Dockerfile -t leaffusionnet-api .
docker run --rm -p 8000:8000 -e PORT=8000 leaffusionnet-api
```

## Deploy to Render

1. Push the repository, including the model and report artifacts, to a Git provider connected to Render.
2. In Render, create a new Blueprint and select the repository. Render reads `deployment/render.yaml` and builds using the included Dockerfile.
3. Confirm the service uses the Free plan and deploy it. Render sets `PORT` automatically.
4. Add `CORS_ORIGINS` in the Render environment if the frontend uses a custom domain.
5. Use the deployed `/health` URL to verify the model loaded, then configure the Vercel frontend to use the service URL.

The Docker image runs a single Uvicorn worker intentionally: TensorFlow model loading is memory-intensive, and one worker ensures one in-memory model instance on Render Free.
