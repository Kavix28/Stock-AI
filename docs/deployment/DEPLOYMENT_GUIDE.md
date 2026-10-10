# Stock-AI Deployment & Operations Guide

This guide describes local development, containerization, and production deployment options for the Stock-AI platform.

---

## 1. Local Development Architecture

```
User Browser (http://localhost:5173)
       │
       ▼
Vite Dev Server (Frontend Proxy)
       │ (routes /api to backend)
       ▼
FastAPI REST API (http://localhost:8000)
       │
       ▼
Inference Engine & Processed Data Parquets
```

### Running Locally

**Terminal 1 (Backend):**

```bash
cd C:\PROJECTS\multimodal-stock-ai
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**

```bash
cd C:\PROJECTS\multimodal-stock-ai\frontend
npm run dev
```

---

## 2. Docker Deployment

### Dockerfile (Backend)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn

COPY src/ ./src/
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY data/processed/ ./data/processed/
COPY results/latest/ ./results/latest/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run

```bash
docker build -t stock-ai-backend:v0.2.0 .
docker run -p 8000:8000 stock-ai-backend:v0.2.0
```

---

## 3. Production Hardening Checklist

- [x] CORS origins restricted via `.env` configuration.
- [x] Absolute paths replaced with relative project-root paths.
- [x] Point-in-time constraints enforced on all prediction endpoints.
- [x] Clear financial risk disclaimers displayed on all user-facing interfaces.
- [x] Research paper PDFs excluded from public repository distributions.
