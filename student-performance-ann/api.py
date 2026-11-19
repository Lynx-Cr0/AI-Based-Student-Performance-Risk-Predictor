from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_student, FEATURES

app=FastAPI(title="Student Performance Predictor API",version="1.0.0",
 description="Predict Low, Medium or High performance from student features.")
class StudentFeatures(BaseModel):
    age: float = Field(...,ge=10,le=100)
    attendance_pct: float = Field(...,ge=0,le=100)
    study_hours_per_week: float = Field(...,ge=0,le=168)
    previous_score: float = Field(...,ge=0,le=100)
    assignment_completion_pct: float = Field(...,ge=0,le=100)
    internal_assessment_score: float = Field(...,ge=0,le=100)
    participation_score: float = Field(...,ge=0,le=10)
    sleep_hours_per_day: float = Field(...,ge=0,le=24)
    extracurricular_hours_per_week: float = Field(...,ge=0,le=168)
    failed_subjects: float = Field(...,ge=0,le=30)
@app.get("/")
def root(): return {"message":"Student Performance Predictor API","docs":"/docs"}
@app.post("/predict")
def predict(payload: StudentFeatures):
    try: return predict_student(payload.model_dump())
    except Exception as exc: raise HTTPException(status_code=400,detail=str(exc))
