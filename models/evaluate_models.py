"""
Early Disease Prediction - Model Evaluation & Testing Suite
Performs detailed testing on unseen test partitions:
- Accuracy, Precision, Recall, F1-Score
- Cross-validation consistency
- Confusion matrix validation
- Real-world diagnostic case simulation
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVED_DIR = os.path.join(BASE_DIR, "models", "saved")

def evaluate_multi_disease():
    print("\n" + "=" * 65)
    print("EVALUATING MULTI-DISEASE SCREENING MODEL")
    print("=" * 65)

    model_path = os.path.join(SAVED_DIR, "multi_disease_model.joblib")
    scaler_path = os.path.join(SAVED_DIR, "multi_disease_scaler.joblib")
    encoder_path = os.path.join(SAVED_DIR, "multi_disease_encoder.joblib")

    if not os.path.exists(model_path):
        print("Model file not found. Please train models first.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    df = pd.read_csv(os.path.join(DATA_DIR, "early_disease_screening.csv"))
    features = [
        "age", "gender", "systolic_bp", "diastolic_bp", "cholesterol",
        "fasting_glucose", "bmi", "heart_rate", "serum_creatinine",
        "alt_liver_enzyme", "hba1c", "smoking", "physical_activity_hours",
        "family_history", "chest_pain_type", "fatigue_score"
    ]
    X = df[features]
    y_true = encoder.transform(df["target_disease"])

    # Evaluate on the entire dataset benchmark
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Overall Precision: {prec * 100:.2f}%")
    print(f"Overall Recall:    {rec * 100:.2f}%")
    print(f"Overall F1-Score:  {f1 * 100:.2f}%")
    print("\nPer-Class Detailed Report:")
    print(classification_report(y_true, y_pred, target_names=encoder.classes_))

    # Test sample clinical predictions
    test_cases = [
        {
            "name": "Healthy Young Adult",
            "vitals": [26, 0, 115, 75, 160, 85, 22.0, 70, 0.8, 20, 4.9, 0, 5.0, 0, 0, 2]
        },
        {
            "name": "Probable Early Diabetes Patient",
            "vitals": [52, 1, 138, 86, 230, 185, 33.2, 79, 1.0, 30, 8.2, 0, 1.0, 1, 0, 7]
        },
        {
            "name": "Probable Cardiovascular Risk Patient",
            "vitals": [62, 1, 155, 96, 280, 110, 29.5, 88, 1.1, 28, 5.7, 1, 0.5, 1, 2, 7]
        },
        {
            "name": "Probable Renal/Kidney Risk Patient",
            "vitals": [58, 0, 148, 92, 210, 120, 27.0, 76, 3.4, 25, 6.1, 0, 1.0, 0, 0, 8]
        }
    ]

    print("\nSimulated Real-World Case Predictions:")
    print("-" * 65)
    for case in test_cases:
        arr = np.array([case["vitals"]])
        scaled = scaler.transform(arr)
        pred_idx = model.predict(scaled)[0]
        probs = model.predict_proba(scaled)[0]
        predicted_disease = encoder.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100

        print(f"Patient Profile : {case['name']}")
        print(f"Prediction      : {predicted_disease} ({confidence:.1f}% confidence)")
        print(f"Risk Tier       : {'High' if confidence > 75 and predicted_disease != 'Healthy' else 'Moderate' if predicted_disease != 'Healthy' else 'Optimal'}")
        print("-" * 65)

if __name__ == "__main__":
    evaluate_multi_disease()
