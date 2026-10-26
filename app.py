"""
Early Disease Prediction System - Backend Web Server (FastAPI)
Provides REST API endpoints and web interface for disease risk predictions:
- Multi-Disease Diagnostic Screening
- Specialized Diabetes Assessment
- Specialized Heart Disease Assessment
- Specialized Chronic Kidney Disease Assessment
- Model Performance Analytics & Metrics
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import joblib
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "models", "saved")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(
    title="MediPredict AI - Early Disease Prediction System",
    description="Machine Learning based Early Diagnostic Health Risk Screening",
    version="1.0.0"
)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global caches for models & artifacts
models_cache = {}

def load_models():
    """Load pre-trained models, scalers, and encoders into memory."""
    try:
        # Multi-disease screening
        models_cache["multi_model"] = joblib.load(os.path.join(SAVED_DIR, "multi_disease_model.joblib"))
        models_cache["multi_scaler"] = joblib.load(os.path.join(SAVED_DIR, "multi_disease_scaler.joblib"))
        models_cache["multi_encoder"] = joblib.load(os.path.join(SAVED_DIR, "multi_disease_encoder.joblib"))

        # Specialized models
        if os.path.exists(os.path.join(SAVED_DIR, "diabetes_model.joblib")):
            models_cache["dia_model"] = joblib.load(os.path.join(SAVED_DIR, "diabetes_model.joblib"))
            models_cache["dia_scaler"] = joblib.load(os.path.join(SAVED_DIR, "diabetes_scaler.joblib"))

        if os.path.exists(os.path.join(SAVED_DIR, "heart_model.joblib")):
            models_cache["heart_model"] = joblib.load(os.path.join(SAVED_DIR, "heart_model.joblib"))
            models_cache["heart_scaler"] = joblib.load(os.path.join(SAVED_DIR, "heart_scaler.joblib"))

        if os.path.exists(os.path.join(SAVED_DIR, "kidney_model.joblib")):
            models_cache["kidney_model"] = joblib.load(os.path.join(SAVED_DIR, "kidney_model.joblib"))
            models_cache["kidney_scaler"] = joblib.load(os.path.join(SAVED_DIR, "kidney_scaler.joblib"))

        # Load metrics summary
        metrics_file = os.path.join(SAVED_DIR, "model_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                models_cache["metrics"] = json.load(f)

        print("[OK] Successfully loaded all models and preprocessing pipelines.")
    except Exception as e:
        print(f"[WARNING] Error loading models: {e}. Models might need training first.")

# Pydantic Schemas for Requests
class MultiDiseaseInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    gender: int = Field(..., ge=0, le=1, description="0: Female, 1: Male")
    systolic_bp: float = Field(..., ge=60, le=260, description="Systolic Blood Pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=160, description="Diastolic Blood Pressure (mmHg)")
    cholesterol: float = Field(..., ge=80, le=500, description="Total Serum Cholesterol (mg/dL)")
    fasting_glucose: float = Field(..., ge=40, le=500, description="Fasting Blood Glucose (mg/dL)")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index (kg/m²)")
    heart_rate: float = Field(..., ge=35, le=220, description="Resting Heart Rate (bpm)")
    serum_creatinine: float = Field(..., ge=0.1, le=15.0, description="Serum Creatinine (mg/dL)")
    alt_liver_enzyme: float = Field(..., ge=5.0, le=350.0, description="ALT / SGPT Liver Enzyme (U/L)")
    hba1c: float = Field(..., ge=3.5, le=18.0, description="Glycated Hemoglobin HbA1c (%)")
    smoking: int = Field(..., ge=0, le=1, description="Smoking status: 0=No, 1=Yes")
    physical_activity_hours: float = Field(..., ge=0.0, le=40.0, description="Weekly exercise hours")
    family_history: int = Field(..., ge=0, le=1, description="Family History: 0=No, 1=Yes")
    chest_pain_type: int = Field(..., ge=0, le=2, description="Chest pain: 0=None, 1=Mild/Occasional, 2=Angina")
    fatigue_score: int = Field(..., ge=1, le=10, description="Fatigue Severity: 1-10")

class DiabetesInput(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20)
    glucose: float = Field(..., ge=40, le=400)
    blood_pressure: float = Field(..., ge=40, le=180)
    skin_thickness: float = Field(..., ge=0, le=99)
    insulin: float = Field(..., ge=0, le=800)
    bmi: float = Field(..., ge=12.0, le=65.0)
    dpf: float = Field(..., ge=0.05, le=3.0)
    age: int = Field(..., ge=18, le=110)

class HeartInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: int = Field(..., ge=0, le=1)
    chest_pain_type: int = Field(..., ge=0, le=3)
    resting_bp: float = Field(..., ge=70, le=240)
    cholesterol: float = Field(..., ge=100, le=550)
    fasting_bs: int = Field(..., ge=0, le=1)
    resting_ecg: int = Field(..., ge=0, le=2)
    max_hr: int = Field(..., ge=60, le=220)
    exercise_angina: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0.0, le=7.0)
    st_slope: int = Field(..., ge=0, le=2)

class KidneyInput(BaseModel):
    age: int = Field(..., ge=10, le=100)
    blood_pressure: float = Field(..., ge=50, le=200)
    specific_gravity: float = Field(..., ge=1.000, le=1.035)
    albumin: int = Field(..., ge=0, le=5)
    sugar: int = Field(..., ge=0, le=5)
    blood_urea: float = Field(..., ge=10, le=250)
    serum_creatinine: float = Field(..., ge=0.2, le=15.0)
    sodium: float = Field(..., ge=100, le=170)
    potassium: float = Field(..., ge=2.0, le=9.0)
    hemoglobin: float = Field(..., ge=4.0, le=20.0)

def generate_health_advice(disease: str, confidence: float, vitals: dict) -> dict:
    """Generate intelligent clinical insights and personalized recommendations."""
    abnormal_biomarkers = []
    recommendations = []

    # Analyze Biomarkers
    if vitals.get("fasting_glucose", 0) > 125 or vitals.get("hba1c", 0) > 6.4:
        abnormal_biomarkers.append({"name": "Fasting Glucose / HbA1c", "value": f"{vitals.get('fasting_glucose')} mg/dL | {vitals.get('hba1c')}%", "status": "High (Diabetic Range)"})
    elif vitals.get("fasting_glucose", 0) > 100 or vitals.get("hba1c", 0) > 5.7:
        abnormal_biomarkers.append({"name": "Fasting Glucose / HbA1c", "value": f"{vitals.get('fasting_glucose')} mg/dL | {vitals.get('hba1c')}%", "status": "Pre-diabetic Range"})

    if vitals.get("systolic_bp", 0) >= 140 or vitals.get("diastolic_bp", 0) >= 90:
        abnormal_biomarkers.append({"name": "Blood Pressure", "value": f"{vitals.get('systolic_bp')}/{vitals.get('diastolic_bp')} mmHg", "status": "Stage 2 Hypertension"})
    elif vitals.get("systolic_bp", 0) >= 130 or vitals.get("diastolic_bp", 0) >= 80:
        abnormal_biomarkers.append({"name": "Blood Pressure", "value": f"{vitals.get('systolic_bp')}/{vitals.get('diastolic_bp')} mmHg", "status": "Elevated / Pre-hypertension"})

    if vitals.get("cholesterol", 0) > 240:
        abnormal_biomarkers.append({"name": "Total Cholesterol", "value": f"{vitals.get('cholesterol')} mg/dL", "status": "High (Cardiovascular Risk)"})
    elif vitals.get("cholesterol", 0) > 200:
        abnormal_biomarkers.append({"name": "Total Cholesterol", "value": f"{vitals.get('cholesterol')} mg/dL", "status": "Borderline High"})

    if vitals.get("serum_creatinine", 0) > 1.3:
        abnormal_biomarkers.append({"name": "Serum Creatinine", "value": f"{vitals.get('serum_creatinine')} mg/dL", "status": "Elevated (Renal Function Alert)"})

    if vitals.get("alt_liver_enzyme", 0) > 50:
        abnormal_biomarkers.append({"name": "ALT Liver Enzyme", "value": f"{vitals.get('alt_liver_enzyme')} U/L", "status": "Elevated (Hepatic Inflammation)"})

    if vitals.get("bmi", 0) >= 30:
        abnormal_biomarkers.append({"name": "BMI", "value": f"{vitals.get('bmi')} kg/m²", "status": "Obese Class I/II"})
    elif vitals.get("bmi", 0) >= 25:
        abnormal_biomarkers.append({"name": "BMI", "value": f"{vitals.get('bmi')} kg/m²", "status": "Overweight"})

    # Specific Recommendations by Disease
    if disease == "Healthy":
        risk_level = "Optimal / Low Risk"
        recommendations = [
            "Maintain balanced Mediterranean-style nutrition rich in vegetables, lean proteins, and fiber.",
            "Continue getting at least 150 minutes of moderate aerobic exercise per week.",
            "Schedule annual comprehensive preventive health checkups to track baseline vitals.",
            "Stay well hydrated and maintain 7-8 hours of quality sleep nightly."
        ]
    elif disease == "Diabetes":
        risk_level = "High Risk" if confidence > 75 else "Moderate Risk"
        recommendations = [
            "Consult an endocrinologist for confirmation with an Oral Glucose Tolerance Test (OGTT).",
            "Significantly reduce refined carbohydrates, sugary beverages, and processed snacks.",
            "Engage in post-meal brisk walking (15-20 minutes) to naturally improve insulin sensitivity.",
            "Monitor fasting blood glucose and maintain HbA1c below 6.5% under physician guidance.",
            "Have routine retinal, renal (microalbuminuria), and diabetic foot screenings."
        ]
    elif disease == "Heart Disease":
        risk_level = "High Risk" if confidence > 70 else "Moderate Risk"
        recommendations = [
            "Schedule an urgent evaluation with a cardiologist (ECG, Echocardiogram, and Stress Test).",
            "Adopt a strict Heart-Healthy diet (low sodium < 2000mg/day, eliminate trans fats, high omega-3).",
            "Avoid strenuous unmonitored exertion until cardiac clearance is obtained.",
            "Cease tobacco/smoking immediately; smoking doubles coronary heart disease recurrence.",
            "Strictly monitor blood pressure and lipid profile every 3-6 months."
        ]
    elif disease == "Hypertension":
        risk_level = "High Risk" if confidence > 75 else "Moderate Risk"
        recommendations = [
            "Adopt the DASH (Dietary Approaches to Stop Hypertension) diet with strict salt reduction (< 1500mg sodium/day).",
            "Log home blood pressure twice daily (morning & evening) before doctor consultation.",
            "Incorporate daily mindfulness, meditation, or stress management techniques.",
            "Engage in daily moderate aerobic exercise (brisk walking, cycling, swimming).",
            "Avoid NSAIDs, excessive caffeine, and alcohol which can elevate arterial pressure."
        ]
    elif disease == "Kidney Disease":
        risk_level = "High Risk" if confidence > 70 else "Moderate Risk"
        recommendations = [
            "Consult a nephrologist for eGFR calculation, urine albumin-to-creatinine ratio (uACR), and renal ultrasound.",
            "Control blood pressure strictly (< 130/80 mmHg) to prevent glomerular damage.",
            "Moderate protein and sodium intake as directed by a renal dietitian.",
            "Avoid nephrotoxic substances, especially over-the-counter NSAID painkillers (e.g., ibuprofen).",
            "Maintain optimal hydration without fluid overloading."
        ]
    elif disease == "Liver Disease":
        risk_level = "High Risk" if confidence > 70 else "Moderate Risk"
        recommendations = [
            "Consult a gastroenterologist/hepatologist for liver ultrasound and comprehensive hepatic panel.",
            "Completely abstain from alcohol consumption to halt hepatic steatosis or inflammation.",
            "Focus on weight reduction (5-10% body weight loss resolves early hepatic steatosis).",
            "Avoid unnecessary dietary supplements or hepatotoxic medications.",
            "Get vaccinated against Hepatitis A and Hepatitis B if not already immunized."
        ]
    else:
        risk_level = "Evaluation Needed"
        recommendations = ["Consult your primary healthcare provider for clinical evaluation."]

    return {
        "risk_level": risk_level,
        "abnormal_biomarkers": abnormal_biomarkers,
        "recommendations": recommendations
    }

@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>MediPredict AI</h1><p>Frontend template initializing...</p>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(models_cache.keys()),
        "ready": "multi_model" in models_cache
    }

@app.get("/api/metrics")
async def get_metrics():
    if "metrics" in models_cache:
        return models_cache["metrics"]
    metrics_file = os.path.join(SAVED_DIR, "model_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            models_cache["metrics"] = data
            return data
    return {"message": "Metrics not yet available. Train models to generate."}

@app.post("/api/predict/multi")
async def predict_multi_disease(payload: MultiDiseaseInput):
    if "multi_model" not in models_cache:
        load_models()
        if "multi_model" not in models_cache:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train models first.")

    model = models_cache["multi_model"]
    scaler = models_cache["multi_scaler"]
    encoder = models_cache["multi_encoder"]

    features_list = [
        payload.age, payload.gender, payload.systolic_bp, payload.diastolic_bp,
        payload.cholesterol, payload.fasting_glucose, payload.bmi, payload.heart_rate,
        payload.serum_creatinine, payload.alt_liver_enzyme, payload.hba1c,
        payload.smoking, payload.physical_activity_hours, payload.family_history,
        payload.chest_pain_type, payload.fatigue_score
    ]

    arr = np.array([features_list])
    arr_scaled = scaler.transform(arr)

    pred_idx = model.predict(arr_scaled)[0]
    predicted_disease = encoder.inverse_transform([pred_idx])[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr_scaled)[0]
        for idx, cls_name in enumerate(encoder.classes_):
            probabilities[cls_name] = round(float(probs[idx]) * 100, 2)
        primary_confidence = round(float(probs[pred_idx]) * 100, 2)
    else:
        primary_confidence = 95.0
        probabilities[predicted_disease] = 95.0

    advice = generate_health_advice(predicted_disease, primary_confidence, payload.dict())

    return {
        "success": True,
        "predicted_disease": predicted_disease,
        "confidence": primary_confidence,
        "risk_level": advice["risk_level"],
        "probabilities": probabilities,
        "abnormal_biomarkers": advice["abnormal_biomarkers"],
        "recommendations": advice["recommendations"],
        "input_summary": payload.dict()
    }

@app.post("/api/predict/diabetes")
async def predict_diabetes(payload: DiabetesInput):
    if "dia_model" not in models_cache:
        load_models()
        if "dia_model" not in models_cache:
            raise HTTPException(status_code=503, detail="Diabetes model not loaded.")

    model = models_cache["dia_model"]
    scaler = models_cache["dia_scaler"]

    features = [
        payload.pregnancies, payload.glucose, payload.blood_pressure,
        payload.skin_thickness, payload.insulin, payload.bmi,
        payload.dpf, payload.age
    ]

    arr_scaled = scaler.transform(np.array([features]))
    pred = int(model.predict(arr_scaled)[0])
    prob = model.predict_proba(arr_scaled)[0]
    prob_diabetic = round(float(prob[1]) * 100, 2)

    status = "Diabetic / High Risk" if pred == 1 else "Non-Diabetic / Normal"
    risk_level = "High Risk" if prob_diabetic >= 65 else "Moderate Risk" if prob_diabetic >= 40 else "Optimal"

    return {
        "success": True,
        "prediction": status,
        "is_diabetic": bool(pred == 1),
        "probability": prob_diabetic,
        "risk_level": risk_level,
        "recommendation": "Consult a physician for HbA1c screening and lifestyle adjustment." if pred == 1 else "Normal glucose metabolic profile. Continue healthy diet."
    }

@app.post("/api/predict/heart")
async def predict_heart(payload: HeartInput):
    if "heart_model" not in models_cache:
        load_models()
        if "heart_model" not in models_cache:
            raise HTTPException(status_code=503, detail="Heart model not loaded.")

    model = models_cache["heart_model"]
    scaler = models_cache["heart_scaler"]

    features = [
        payload.age, payload.sex, payload.chest_pain_type, payload.resting_bp,
        payload.cholesterol, payload.fasting_bs, payload.resting_ecg,
        payload.max_hr, payload.exercise_angina, payload.oldpeak, payload.st_slope
    ]

    arr_scaled = scaler.transform(np.array([features]))
    pred = int(model.predict(arr_scaled)[0])
    prob = model.predict_proba(arr_scaled)[0]
    prob_hd = round(float(prob[1]) * 100, 2)

    status = "Elevated Heart Disease Risk" if pred == 1 else "Normal Cardiac Profile"
    risk_level = "High Risk" if prob_hd >= 65 else "Moderate Risk" if prob_hd >= 40 else "Low Risk"

    return {
        "success": True,
        "prediction": status,
        "has_heart_disease": bool(pred == 1),
        "probability": prob_hd,
        "risk_level": risk_level,
        "recommendation": "Cardiology consultation recommended (ECG, echocardiogram)." if pred == 1 else "Cardiovascular indicators within healthy ranges."
    }

@app.post("/api/predict/kidney")
async def predict_kidney(payload: KidneyInput):
    if "kidney_model" not in models_cache:
        load_models()
        if "kidney_model" not in models_cache:
            raise HTTPException(status_code=503, detail="Kidney model not loaded.")

    model = models_cache["kidney_model"]
    scaler = models_cache["kidney_scaler"]

    features = [
        payload.age, payload.blood_pressure, payload.specific_gravity,
        payload.albumin, payload.sugar, payload.blood_urea,
        payload.serum_creatinine, payload.sodium, payload.potassium, payload.hemoglobin
    ]

    arr_scaled = scaler.transform(np.array([features]))
    pred = int(model.predict(arr_scaled)[0])
    prob = model.predict_proba(arr_scaled)[0]
    prob_ckd = round(float(prob[1]) * 100, 2)

    status = "Early Chronic Kidney Disease Risk" if pred == 1 else "Normal Kidney Function"
    risk_level = "High Risk" if prob_ckd >= 65 else "Moderate Risk" if prob_ckd >= 40 else "Optimal"

    return {
        "success": True,
        "prediction": status,
        "has_ckd": bool(pred == 1),
        "probability": prob_ckd,
        "risk_level": risk_level,
        "recommendation": "Nephrology assessment and renal function monitoring advised." if pred == 1 else "Renal biomarkers indicate normal filtration capability."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
