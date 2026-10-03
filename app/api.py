from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline import ParkingCompliancePipeline
from app.vision.yolo_models import ModelFilesMissing

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

app = FastAPI(title="YOLO Parking Compliance Detector", version="2.0.0")
pipeline = ParkingCompliancePipeline()
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health():
    status = pipeline.model_status()
    return {
        "status": "ok" if all(status[k] for k in ("vehicle_ready", "boundary_ready", "cat_eye_ready")) else "models_missing",
        "runtime": "YOLO / CPU",
        "external_api": False,
        "models": status,
    }


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "Upload at least one image")
    payload: list[tuple[str, bytes]] = []
    for f in files:
        if not f.filename:
            continue
        if not f.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            raise HTTPException(400, f"Unsupported file: {f.filename}")
        payload.append((f.filename, await f.read()))
    try:
        return pipeline.analyze_files(payload)
    except ModelFilesMissing as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
