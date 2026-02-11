from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .predictor import Predictor

FEATURE_SPECS = [
    {"name": "age", "label": "อายุ (age)", "type": "number", "step": "1", "placeholder": "เช่น 22"},
    {"name": "years_experience", "label": "ประสบการณ์ทำงาน (ปี) (years_experience)", "type": "number", "step": "0.1", "placeholder": "เช่น 1.5"},
    {"name": "gpa", "label": "GPA (gpa)", "type": "number", "step": "0.01", "placeholder": "เช่น 3.25"},
    {"name": "aptitude_test", "label": "คะแนน Aptitude (aptitude_test)", "type": "number", "step": "1", "placeholder": "เช่น 75"},
    {"name": "interview_score", "label": "คะแนนสัมภาษณ์ (interview_score)", "type": "number", "step": "1", "placeholder": "เช่น 80"},
    {"name": "english_score", "label": "คะแนนอังกฤษ (english_score)", "type": "number", "step": "1", "placeholder": "เช่น 70"},
    {"name": "tech_skill", "label": "ทักษะเทคนิค (tech_skill)", "type": "number", "step": "1", "placeholder": "เช่น 85"},
    {"name": "soft_skill", "label": "ทักษะ Soft skill (soft_skill)", "type": "number", "step": "1", "placeholder": "เช่น 78"},
    {"name": "portfolio_score", "label": "คะแนนพอร์ต (portfolio_score)", "type": "number", "step": "1", "placeholder": "เช่น 90"},
    {"name": "cert_count", "label": "จำนวนใบเซอร์ (cert_count)", "type": "number", "step": "1", "placeholder": "เช่น 2"},
    {"name": "expected_salary", "label": "เงินเดือนที่คาดหวัง (expected_salary)", "type": "number", "step": "1", "placeholder": "เช่น 25000"},
    {"name": "notice_days", "label": "วันแจ้งลาออก (notice_days)", "type": "number", "step": "1", "placeholder": "เช่น 30"},
    {"name": "distance_km", "label": "ระยะทาง (กม.) (distance_km)", "type": "number", "step": "0.1", "placeholder": "เช่น 12.5"},
    {"name": "education_level", "label": "ระดับการศึกษา (education_level)", "type": "text", "placeholder": "เช่น bachelor"},
    {"name": "role_code", "label": "โค้ดตำแหน่ง (role_code)", "type": "text", "placeholder": "เช่น dev"},
    {"name": "source_code", "label": "แหล่งที่มา (source_code)", "type": "text", "placeholder": "เช่น linkedin"},
    {"name": "salary_per_exp", "label": "salary_per_exp (ปล่อยว่างให้คำนวณได้)", "type": "number", "step": "0.01", "placeholder": "เช่น 15000"},
    {"name": "skill_per_exp", "label": "skill_per_exp (ปล่อยว่างให้คำนวณได้)", "type": "number", "step": "0.01", "placeholder": "เช่น 40"},
    {"name": "portfolio_to_salary", "label": "portfolio_to_salary (ปล่อยว่างให้คำนวณได้)", "type": "number", "step": "0.0001", "placeholder": "เช่น 0.0036"},
]

app = FastAPI(title="Job Application Outcome Predictor", version="2.4.1")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

MODEL_PATH = os.getenv("MODEL_PATH", "best_model_job_app.joblib")
ARTIFACTS_DIR = os.getenv("MODEL_ARTIFACTS_DIR", "model_artifacts")

predictor = Predictor(default_model_path=MODEL_PATH, artifacts_dir=ARTIFACTS_DIR)


class PredictRequest(BaseModel):
    input: Dict[str, Any] = Field(...)
    model_key: str = Field("best_model_job_app")
    top_k: int = Field(5, ge=1, le=20)


@app.on_event("startup")
def startup():
    predictor.load()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc), "where": "server_unhandled_exception"})


@app.get("/health")
def health():
    return {"status": "ok", "models": predictor.list_models()}


@app.get("/api/models")
def api_models():
    return {"models": predictor.list_models()}


# ✅ เอาตัวชี้วัดออก: ลบ /api/metrics ไปเลย


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    try:
        input_data = req.input or {}
        has_any_value = any((v is not None) and (str(v).strip() != "") for v in input_data.values())
        if not has_any_value:
            return JSONResponse(status_code=400, content={"error": "กรุณากรอกข้อมูลอย่างน้อย 1 ช่องก่อนกด Predict"})

        result = predictor.predict_one(req.model_key, input_data, top_k=req.top_k)
        return {
            "model": req.model_key,
            "predicted_class": result["predicted_class"],
            "probabilities": result["probabilities"],
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e), "where": "api_predict_try_except"})


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "features": FEATURE_SPECS})
