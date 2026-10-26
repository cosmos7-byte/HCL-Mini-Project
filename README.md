# MediPredict AI: Early Disease Prediction Using Machine Learning on Health Data

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Project Type](https://img.shields.io/badge/Type-HCL%20Mini%20Project-purple.svg)]()

> An end-to-end Machine Learning healthcare platform designed to proactively screen and predict the early onset of chronic and lifestyle diseases using routine clinical indicators, hemodynamic vitals, and metabolic biomarkers.

---

## 📌 Executive Summary

Early detection of chronic non-communicable diseases (NCDs) substantially reduces morbidity, mortality, and hospitalization costs. Most conventional diagnostics detect conditions only after irreversible physiological damage has occurred. 

**MediPredict AI** addresses this paradigm by employing supervised machine learning models trained on multidimensional health biomarkers. The system stratifies patient risk across:
1. **Diabetes Mellitus** (Type 2 & Pre-diabetes)
2. **Cardiovascular / Coronary Heart Disease**
3. **Hypertension (Stages 1 & 2)**
4. **Chronic Kidney Disease (CKD)**
5. **Non-Alcoholic Fatty Liver Disease (NAFLD / Hepatic Steatosis)**

---

## 🏗️ System Architecture

```
                                  [ Patient Health Data ]
               (Vitals: BP, BMI, Heart Rate | Labs: Glucose, HbA1c, Creatinine, ALT, Lipids)
                                             │
                                             ▼
                             [ Data Preprocessing Pipeline ]
                             (StandardScaler, Feature Encoding)
                                             │
                                             ▼
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          [ Multi-Disease Champion ]                 [ Specialized Model Suites ]
         (Random Forest / Gradient Boost)            - Diabetes Risk (Pima/CDC)
         Multi-class Disease Classification          - Cardiac Risk (Framingham)
                       │                             - CKD Renal Filtration
                       ▼                                           │
         ┌───────────────────────────┐                             │
         │  Risk Score & Confidence  │◄────────────────────────────┘
         │  Probability Distribution │
         │  Biomarker Alert Flags    │
         │  Personalized Guidance    │
         └─────────────┬─────────────┘
                       │
                       ▼
       [ Interactive Modern Web Interface ] ◄── (FastAPI REST API / JSON)
```

---

## 📊 Datasets

The platform includes four scientifically calibrated datasets generated in `data/`:

| Dataset | Samples | Key Features | Target Prediction |
| :--- | :--- | :--- | :--- |
| **`early_disease_screening.csv`** | 3,500 | Age, Gender, Systolic/Diastolic BP, Cholesterol, Fasting Glucose, BMI, Heart Rate, Creatinine, ALT, HbA1c, Smoking, Physical Activity, Family History, Chest Pain, Fatigue | Multi-class: `Healthy`, `Diabetes`, `Heart Disease`, `Hypertension`, `Kidney Disease`, `Liver Disease` |
| **`diabetes_prediction.csv`** | 1,800 | Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age | Binary: `0: Non-Diabetic`, `1: Diabetic` |
| **`heart_disease_prediction.csv`** | 1,800 | Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max HR, Exercise Angina, Oldpeak ST, ST Slope | Binary: `0: Normal Cardiac Profile`, `1: Heart Disease` |
| **`kidney_disease_prediction.csv`** | 1,600 | Age, Blood Pressure, Specific Gravity, Albumin, Sugar, Blood Urea, Serum Creatinine, Sodium, Potassium, Hemoglobin | Binary: `0: Normal Kidney`, `1: Early CKD` |

---

## 🧠 Machine Learning Algorithms & Training

The model engine implements a candidate model tournament evaluating:
- **Random Forest Classifier**: Ensemble of bootstrap decision trees with Gini impurity splitting.
- **Gradient Boosting Classifier**: Sequential gradient-descent optimization over weak learners.
- **Logistic Regression**: Baseline linear log-odds classifier with L2 regularization.
- **Decision Trees**: Pruned decision trees for baseline interpretability.

### Training & Validation Protocol:
- **Stratified Train-Test Split**: 80% Training / 20% Held-out Testing.
- **5-Fold Cross Validation**: Guarantees generalization and rules out overfitting.
- **Feature Standardization**: Zero-mean, unit-variance scaling (`StandardScaler`) saved as artifacts.

---

## 🖥️ Web User Interface Features

- **Interactive Diagnostic Screening**: Real-time form with vital signs, blood chemistry, and symptom inputs.
- **One-Click Clinical Patient Presets**: Instantly autofills and runs:
  - Healthy Individual
  - Early Diabetes Case
  - Cardiac Risk Case
  - Hypertension Case
  - Renal / Kidney Case
  - Hepatic / Liver Case
- **Risk Stratification Gauge**: Visual risk meter with Optimal, Mild, Moderate, and High Risk tiers.
- **Multiclass Probability Distribution**: Visual percentage bars across all candidate diseases.
- **Clinical Biomarker Flags**: Automatic alerts for abnormal glucose, blood pressure, cholesterol, creatinine, or ALT.
- **Personalized Clinical Recommendations**: Tailored dietary, lifestyle, and medical specialist referral advice.
- **Performance & Explainability Analytics**: Confusion matrices, test benchmarks, and feature importance bar plots.
- **Printable Medical Report**: One-click summary formatted for clinical consultations.

---

## 🚀 Quick Start Guide

### 1. One-Click Launcher (Windows)
Simply double-click:
```bat
run.bat
```
This automatically verifies datasets, trains any missing models, starts the FastAPI server, and opens `http://127.0.0.1:8000` in your default browser.

### 2. Manual Command Line Execution
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate datasets
python data/generate_datasets.py

# 3. Train & Evaluate Machine Learning models
python models/train_models.py

# 4. Start Web Application
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

---

## 📁 Repository Directory Structure

```
HCL  Mini project/
│
├── data/
│   ├── generate_datasets.py           # Scientific dataset generation script
│   ├── early_disease_screening.csv    # 3,500 comprehensive patient records
│   ├── diabetes_prediction.csv        # 1,800 diabetes patient records
│   ├── heart_disease_prediction.csv   # 1,800 cardiovascular patient records
│   └── kidney_disease_prediction.csv  # 1,600 chronic kidney disease records
│
├── models/
│   ├── train_models.py                # ML training, cross-validation & artifact serialization
│   ├── evaluate_models.py             # Independent test suite & case simulation
│   └── saved/                         # Serialized .joblib models, scalers, and JSON metrics
│       ├── multi_disease_model.joblib
│       ├── multi_disease_scaler.joblib
│       ├── multi_disease_encoder.joblib
│       ├── diabetes_model.joblib
│       ├── diabetes_scaler.joblib
│       ├── heart_model.joblib
│       ├── heart_scaler.joblib
│       ├── kidney_model.joblib
│       ├── kidney_scaler.joblib
│       └── model_metrics.json
│
├── static/
│   ├── css/
│   │   └── style.css                  # Modern responsive medical UI styling
│   ├── js/
│   │   └── app.js                     # Dynamic form handling, predictions & visualizations
│   └── reports/                       # Generated evaluation plots (Confusion Matrix, Feature Importance)
│       ├── multi_disease_cm.png
│       ├── multi_disease_fi.png
│       ├── diabetes_cm.png
│       ├── heart_cm.png
│       └── kidney_cm.png
│
├── templates/
│   └── index.html                     # Responsive web dashboard
│
├── app.py                             # FastAPI backend & prediction REST API
├── run.bat                            # Windows one-click batch launcher
├── requirements.txt                   # Project Python dependencies
└── README.md                          # Comprehensive documentation
```

---

## ⚖️ Clinical Disclaimer
*MediPredict AI is designed as a clinical decision support and preliminary risk screening tool for research and educational purposes. It does not replace professional medical diagnosis, laboratory testing, or physician consultation.*
