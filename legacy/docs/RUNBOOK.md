# Leather App: Quick Runbook

This document is a concise reference for you and future assistants to build, test, and deploy the Leather classification web app quickly.

## Goal
- Serve a Flask web UI that lets a user upload an image and get predictions from Keras/TensorFlow models (InceptionV3 or AlexNet) on AWS Elastic Beanstalk (EB) using Docker.

## Tech Stack
- Python 3.10, Flask, TensorFlow/Keras, Pillow
- Docker (Gunicorn WSGI server)
- AWS Elastic Beanstalk (Docker on Amazon Linux 2)
- SQLite for demo persistence

## Repo Layout (key items)
- `app.py` — Flask app (loads models from env paths/URLs)
- `Dockerfile` — Builds runtime image and launches Gunicorn
- `requirements.txt` — Python deps
- `templates/` and `static/` — UI and assets
- `database.db` — SQLite demo DB (ephemeral in container)
- Model files in repo root:
  - `inceptionNetV3_50e_v2v3_v1_final_TRIAL2_Leather_species_identification_4 classes.h5`
  - `Alexnet_cs4600_Leather authentication_cross_section_2 classes.h5`

## Environment Variables (required)
- `SECRET_KEY` — any long random string
- `MODEL_INCEPTION_PATH` — container path to Inception model file, e.g.
  - `/app/inceptionNetV3_50e_v2v3_v1_final_TRIAL2_Leather_species_identification_4 classes.h5`
- `MODEL_ALEXNET_PATH` — container path to AlexNet model file, e.g.
  - `/app/Alexnet_cs4600_Leather authentication_cross_section_2 classes.h5`

Optional (only if you prefer runtime download instead of bundling in ZIP):
- `MODEL_INCEPTION_URL` — HTTPS URL to `.h5` file
- `MODEL_ALEXNET_URL` — HTTPS URL to `.h5` file

Notes:
- Inside the container, the project root is `/app` (from Dockerfile `WORKDIR /app` and `COPY . /app`).
- app.py creates an admin user for demo: `admin` / `admin123`.

## Local Test (Docker)
Prereq: Docker Desktop running on Windows.

1) Build image
```powershell
cd C:\Users\Reetardish\Downloads\leather-master
docker build -t leather:latest .
```

2) Run container
```powershell
docker run -p 5000:5000 `
  -e SECRET_KEY=test-secret `
  -e MODEL_INCEPTION_PATH="/app/inceptionNetV3_50e_v2v3_v1_final_TRIAL2_Leather_species_identification_4 classes.h5" `
  -e MODEL_ALEXNET_PATH="/app/Alexnet_cs4600_Leather authentication_cross_section_2 classes.h5" `
  leather:latest
```

3) Open http://localhost:5000 and test a prediction.

If a filename has spaces, quoting the entire value is sufficient in PowerShell (as above).

## Package for AWS EB (Console deploy)
Create a ZIP of the project root so that the ZIP contains `Dockerfile`, `app.py`, `requirements.txt`, `.dockerignore`, folders, and the `.h5` files at the top level.

```powershell
cd C:\Users\Reetardish\Downloads\leather-master
Compress-Archive -Path * -DestinationPath leather-master.zip -Force
```

## Deploy on Elastic Beanstalk (Console)
1) AWS Console → Elastic Beanstalk → Create application
- Application name: `leather-app`
- Platform: `Docker`
- Platform branch: `Docker running on 64bit Amazon Linux 2`
- Application code: Upload `leather-master.zip`
- Environment type: `Single instance`
- Instance type: `t3.large` (recommended for TensorFlow)

2) After environment is created → Configuration → Software → Edit → Environment properties
- `SECRET_KEY = <your-secret>`
- `MODEL_INCEPTION_PATH = /app/inceptionNetV3_50e_v2v3_v1_final_TRIAL2_Leather_species_identification_4 classes.h5`
- `MODEL_ALEXNET_PATH = /app/Alexnet_cs4600_Leather authentication_cross_section_2 classes.h5`

3) Save/Apply; EB will redeploy. When Health is Green, click the environment URL to open the app and test.

## Post‑Deploy Checks
- Home page loads at EB URL
- Upload image → Predict (Inception first)
- Admin login at `/admin/login`: `admin` / `admin123` (demo only)
- Logs: Environment → Logs → Request logs (last 100 lines)

## Troubleshooting Quick Guide
- "Model file not found":
  - Env var path must start with `/app/` and match the filename exactly (including spaces)
- Import/server errors:
  - Check EB logs and verify `requirements.txt` installed; app entry is `app:app` in Gunicorn
- OOM/slow:
  - Use `t3.xlarge` or reduce workers/threads in Dockerfile CMD
- Files/DB persistence:
  - SQLite DB and uploads inside the container are ephemeral across deploys; use S3/RDS for production

## Security To‑Dos (after demo)
- Replace default admin with env‑driven credentials (or disable auto‑create)
- Add HTTPS via ACM on the EB load balancer
- Move uploads to S3 and DB to RDS for multi‑instance scaling

## Reference Commands
- Build: `docker build -t leather:latest .`
- Run: `docker run -p 5000:5000 -e SECRET_KEY=... -e MODEL_INCEPTION_PATH=... -e MODEL_ALEXNET_PATH=... leather:latest`
- Zip: `Compress-Archive -Path * -DestinationPath leather-master.zip -Force`

## Notes for Assistants
- Do not rename files unless updating env vars accordingly
- Avoid modifying comments/docs unless asked
- Prefer EB Console deployment for Windows to avoid EB CLI + IIS DLL quirks
