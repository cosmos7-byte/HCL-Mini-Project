"""
Early Disease Prediction - Clinical Health Dataset Generator
Generates realistic, scientifically calibrated synthetic health datasets for:
1. Multi-Disease Comprehensive Early Screening (Diabetes, Heart Disease, Hypertension, Kidney Disease, Liver Disease, Healthy)
2. Specialized Diabetes Prediction (Pima/CDC calibrated)
3. Specialized Heart Disease Prediction (Cleveland/Framingham calibrated)
4. Specialized Chronic Kidney Disease (CKD) Prediction
"""

import os
import csv
import random
import math

# Set deterministic seed for reproducibility
random.seed(42)

def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))

def generate_multi_disease_dataset(filepath, num_samples=3500):
    """
    Generates a realistic multi-disease screening dataset.
    Features:
      - age (years): 18 - 85
      - gender: 0 (Female), 1 (Male)
      - systolic_bp: mmHg (90 - 200)
      - diastolic_bp: mmHg (60 - 120)
      - cholesterol: mg/dL (120 - 360)
      - fasting_glucose: mg/dL (65 - 320)
      - bmi: kg/m^2 (16.0 - 46.0)
      - heart_rate: bpm (50 - 125)
      - serum_creatinine: mg/dL (0.4 - 5.5)
      - alt_liver_enzyme: U/L (8 - 180)
      - hba1c: % (4.2 - 12.5)
      - smoking: 0 (No), 1 (Yes)
      - physical_activity_hours: hrs/week (0 - 14)
      - family_history: 0 (No), 1 (Yes)
      - chest_pain_type: 0 (None), 1 (Mild/Non-cardiac), 2 (Angina)
      - fatigue_score: 1 to 10
      - target_disease: Healthy, Diabetes, Heart Disease, Hypertension, Kidney Disease, Liver Disease
    """
    diseases = ["Healthy", "Diabetes", "Heart Disease", "Hypertension", "Kidney Disease", "Liver Disease"]
    weights = [0.25, 0.18, 0.18, 0.15, 0.12, 0.12] # Prevalence distribution

    headers = [
        "age", "gender", "systolic_bp", "diastolic_bp", "cholesterol",
        "fasting_glucose", "bmi", "heart_rate", "serum_creatinine",
        "alt_liver_enzyme", "hba1c", "smoking", "physical_activity_hours",
        "family_history", "chest_pain_type", "fatigue_score", "target_disease"
    ]

    rows = []

    for _ in range(num_samples):
        # Choose assigned profile
        assigned = random.choices(diseases, weights=weights, k=1)[0]

        gender = random.choice([0, 1])
        smoking = random.choices([0, 1], weights=[0.75, 0.25] if assigned == "Healthy" else [0.55, 0.45])[0]
        family_history = random.choices([0, 1], weights=[0.8, 0.2] if assigned == "Healthy" else [0.45, 0.55])[0]

        if assigned == "Healthy":
            age = int(random.gauss(38, 12))
            systolic_bp = round(random.gauss(118, 8), 1)
            diastolic_bp = round(random.gauss(76, 6), 1)
            cholesterol = round(random.gauss(175, 20), 1)
            fasting_glucose = round(random.gauss(88, 9), 1)
            bmi = round(random.gauss(23.2, 2.5), 1)
            heart_rate = int(random.gauss(72, 7))
            serum_creatinine = round(random.gauss(0.85, 0.15), 2)
            alt_liver_enzyme = round(random.gauss(24, 7), 1)
            hba1c = round(random.gauss(5.1, 0.3), 1)
            physical_activity = round(max(0.5, random.gauss(4.5, 2.0)), 1)
            chest_pain = 0
            fatigue = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.2, 0.05])[0]

        elif assigned == "Diabetes":
            age = int(random.gauss(52, 11))
            fasting_glucose = round(random.gauss(165, 35), 1) # elevated glucose
            hba1c = round(random.gauss(7.8, 1.2), 1)           # elevated HbA1c
            bmi = round(random.gauss(31.5, 4.5), 1)           # high BMI
            systolic_bp = round(random.gauss(134, 14), 1)
            diastolic_bp = round(random.gauss(84, 9), 1)
            cholesterol = round(random.gauss(220, 35), 1)
            heart_rate = int(random.gauss(78, 9))
            serum_creatinine = round(random.gauss(1.05, 0.25), 2)
            alt_liver_enzyme = round(random.gauss(32, 12), 1)
            physical_activity = round(max(0, random.gauss(1.8, 1.2)), 1)
            chest_pain = random.choices([0, 1], weights=[0.8, 0.2])[0]
            fatigue = random.choices([4, 5, 6, 7, 8], weights=[0.1, 0.25, 0.35, 0.2, 0.1])[0]

        elif assigned == "Heart Disease":
            age = int(random.gauss(59, 9))
            cholesterol = round(random.gauss(258, 38), 1)     # high cholesterol
            systolic_bp = round(random.gauss(146, 16), 1)     # elevated BP
            diastolic_bp = round(random.gauss(92, 10), 1)
            fasting_glucose = round(random.gauss(110, 22), 1)
            hba1c = round(random.gauss(5.8, 0.6), 1)
            bmi = round(random.gauss(28.8, 3.8), 1)
            heart_rate = int(random.gauss(84, 12))
            serum_creatinine = round(random.gauss(1.1, 0.2), 2)
            alt_liver_enzyme = round(random.gauss(28, 10), 1)
            physical_activity = round(max(0, random.gauss(1.2, 1.0)), 1)
            chest_pain = random.choices([1, 2], weights=[0.4, 0.6])[0] # angina / pain
            fatigue = random.choices([5, 6, 7, 8, 9], weights=[0.15, 0.25, 0.3, 0.2, 0.1])[0]

        elif assigned == "Hypertension":
            age = int(random.gauss(54, 10))
            systolic_bp = round(random.gauss(158, 15), 1)     # stage 1 & 2 hypertension
            diastolic_bp = round(random.gauss(100, 9), 1)
            cholesterol = round(random.gauss(215, 30), 1)
            fasting_glucose = round(random.gauss(102, 15), 1)
            hba1c = round(random.gauss(5.5, 0.4), 1)
            bmi = round(random.gauss(29.2, 3.5), 1)
            heart_rate = int(random.gauss(82, 10))
            serum_creatinine = round(random.gauss(1.0, 0.2), 2)
            alt_liver_enzyme = round(random.gauss(26, 9), 1)
            physical_activity = round(max(0, random.gauss(1.5, 1.2)), 1)
            chest_pain = random.choices([0, 1], weights=[0.75, 0.25])[0]
            fatigue = random.choices([3, 4, 5, 6], weights=[0.2, 0.35, 0.3, 0.15])[0]

        elif assigned == "Kidney Disease":
            age = int(random.gauss(57, 11))
            serum_creatinine = round(random.gauss(2.8, 0.8), 2) # elevated creatinine (>1.5)
            systolic_bp = round(random.gauss(144, 16), 1)
            diastolic_bp = round(random.gauss(90, 10), 1)
            fasting_glucose = round(random.gauss(115, 25), 1)
            hba1c = round(random.gauss(6.0, 0.8), 1)
            cholesterol = round(random.gauss(205, 30), 1)
            bmi = round(random.gauss(27.4, 4.0), 1)
            heart_rate = int(random.gauss(77, 9))
            alt_liver_enzyme = round(random.gauss(29, 10), 1)
            physical_activity = round(max(0, random.gauss(1.0, 0.8)), 1)
            chest_pain = random.choices([0, 1], weights=[0.8, 0.2])[0]
            fatigue = random.choices([6, 7, 8, 9, 10], weights=[0.1, 0.25, 0.35, 0.2, 0.1])[0]

        elif assigned == "Liver Disease":
            age = int(random.gauss(49, 12))
            alt_liver_enzyme = round(random.gauss(98, 28), 1)  # high ALT / transaminases
            cholesterol = round(random.gauss(230, 40), 1)
            fasting_glucose = round(random.gauss(112, 22), 1)
            hba1c = round(random.gauss(5.7, 0.6), 1)
            bmi = round(random.gauss(29.5, 4.2), 1)
            systolic_bp = round(random.gauss(130, 12), 1)
            diastolic_bp = round(random.gauss(82, 8), 1)
            heart_rate = int(random.gauss(75, 8))
            serum_creatinine = round(random.gauss(0.9, 0.2), 2)
            physical_activity = round(max(0, random.gauss(1.3, 1.0)), 1)
            chest_pain = 0
            fatigue = random.choices([5, 6, 7, 8, 9], weights=[0.15, 0.3, 0.3, 0.15, 0.1])[0]

        # Apply physiological constraints
        age = int(clamp(age, 18, 85))
        systolic_bp = round(clamp(systolic_bp, 90, 200), 1)
        diastolic_bp = round(clamp(diastolic_bp, 55, 120), 1)
        cholesterol = round(clamp(cholesterol, 120, 360), 1)
        fasting_glucose = round(clamp(fasting_glucose, 65, 320), 1)
        bmi = round(clamp(bmi, 16.0, 46.0), 1)
        heart_rate = int(clamp(heart_rate, 50, 125))
        serum_creatinine = round(clamp(serum_creatinine, 0.4, 6.0), 2)
        alt_liver_enzyme = round(clamp(alt_liver_enzyme, 10, 180), 1)
        hba1c = round(clamp(hba1c, 4.2, 13.0), 1)
        physical_activity = round(clamp(physical_activity, 0, 14), 1)
        fatigue = int(clamp(fatigue, 1, 10))

        rows.append([
            age, gender, systolic_bp, diastolic_bp, cholesterol,
            fasting_glucose, bmi, heart_rate, serum_creatinine,
            alt_liver_enzyme, hba1c, smoking, physical_activity,
            family_history, chest_pain, fatigue, assigned
        ])

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated multi-disease dataset: {filepath} ({len(rows)} samples)")

def generate_specialized_diabetes_dataset(filepath, num_samples=1800):
    """
    Specialized Diabetes Risk Dataset (aligned with Pima/CDC clinical indicators).
    Features:
      - Pregnancies (0 if Male, 0-12 if Female)
      - Glucose (mg/dL)
      - BloodPressure (diastolic mmHg)
      - SkinThickness (mm)
      - Insulin (mu U/ml)
      - BMI (kg/m^2)
      - DiabetesPedigreeFunction (0.08 - 2.42)
      - Age (years)
      - Outcome (0: No Diabetes, 1: Diabetes)
    """
    headers = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    rows = []

    for _ in range(num_samples):
        outcome = random.choices([0, 1], weights=[0.65, 0.35])[0]
        age = int(random.gauss(32, 10) if outcome == 0 else random.gauss(46, 11))
        age = int(clamp(age, 21, 80))
        is_female = random.random() < 0.6
        pregnancies = int(clamp(random.gauss(2, 2) if outcome == 0 else random.gauss(4.5, 2.5), 0, 14)) if is_female else 0

        if outcome == 0:
            glucose = round(clamp(random.gauss(108, 18), 70, 150), 1)
            bp = round(clamp(random.gauss(70, 10), 50, 105), 1)
            skin = round(clamp(random.gauss(24, 7), 10, 50), 1)
            insulin = round(clamp(random.gauss(80, 40), 15, 240), 1)
            bmi = round(clamp(random.gauss(26.5, 4.2), 18.0, 42.0), 1)
            dpf = round(clamp(random.gauss(0.38, 0.20), 0.08, 1.8), 3)
        else:
            glucose = round(clamp(random.gauss(155, 28), 115, 280), 1)
            bp = round(clamp(random.gauss(78, 12), 55, 120), 1)
            skin = round(clamp(random.gauss(32, 9), 15, 60), 1)
            insulin = round(clamp(random.gauss(185, 80), 30, 480), 1)
            bmi = round(clamp(random.gauss(34.8, 5.5), 23.0, 50.0), 1)
            dpf = round(clamp(random.gauss(0.68, 0.32), 0.15, 2.4), 3)

        rows.append([pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, outcome])

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated specialized diabetes dataset: {filepath} ({len(rows)} samples)")

def generate_specialized_heart_dataset(filepath, num_samples=1800):
    """
    Specialized Heart Disease Dataset (aligned with Cleveland/UCI standard format).
    Features:
      - Age, Sex (1=Male, 0=Female), ChestPainType (0: typical angina, 1: atypical, 2: non-anginal, 3: asymptomatic)
      - RestingBP, Cholesterol, FastingBS (0 or 1), RestingECG (0: normal, 1: ST-T wave, 2: LVH)
      - MaxHR, ExerciseAngina (0 or 1), Oldpeak (ST depression), ST_Slope (0: upsloping, 1: flat, 2: downsloping)
      - HeartDisease (0: Normal, 1: Disease)
    """
    headers = [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "HeartDisease"
    ]
    rows = []

    for _ in range(num_samples):
        outcome = random.choices([0, 1], weights=[0.52, 0.48])[0]
        sex = random.choices([1, 0], weights=[0.68, 0.32])[0]
        age = int(clamp(random.gauss(50, 9) if outcome == 0 else random.gauss(57, 8), 28, 80))

        if outcome == 0:
            cpt = random.choices([1, 2, 0, 3], weights=[0.35, 0.35, 0.20, 0.10])[0]
            rbp = round(clamp(random.gauss(126, 14), 94, 180), 1)
            chol = round(clamp(random.gauss(218, 38), 125, 340), 1)
            fbs = 1 if random.random() < 0.12 else 0
            recg = random.choices([0, 1, 2], weights=[0.65, 0.25, 0.10])[0]
            maxhr = int(clamp(random.gauss(156, 18), 90, 202))
            exang = 1 if random.random() < 0.18 else 0
            oldpeak = round(clamp(random.gauss(0.4, 0.5), 0.0, 3.5), 1)
            sts = random.choices([0, 1, 2], weights=[0.60, 0.35, 0.05])[0]
        else:
            cpt = random.choices([3, 0, 2, 1], weights=[0.55, 0.25, 0.12, 0.08])[0]
            rbp = round(clamp(random.gauss(138, 18), 100, 200), 1)
            chol = round(clamp(random.gauss(262, 45), 150, 420), 1)
            fbs = 1 if random.random() < 0.28 else 0
            recg = random.choices([0, 1, 2], weights=[0.35, 0.40, 0.25])[0]
            maxhr = int(clamp(random.gauss(128, 20), 70, 180))
            exang = 1 if random.random() < 0.62 else 0
            oldpeak = round(clamp(random.gauss(1.8, 1.1), 0.0, 5.0), 1)
            sts = random.choices([1, 2, 0], weights=[0.55, 0.35, 0.10])[0]

        rows.append([age, sex, cpt, rbp, chol, fbs, recg, maxhr, exang, oldpeak, sts, outcome])

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated specialized heart disease dataset: {filepath} ({len(rows)} samples)")

def generate_specialized_kidney_dataset(filepath, num_samples=1600):
    """
    Specialized Chronic Kidney Disease (CKD) Dataset.
    Features:
      - Age, BloodPressure, SpecificGravity, Albumin (0-5), Sugar (0-5),
      - BloodUrea, SerumCreatinine, Sodium, Potassium, Hemoglobin, CKD_Disease (0: Normal, 1: CKD)
    """
    headers = [
        "Age", "BloodPressure", "SpecificGravity", "Albumin", "Sugar",
        "BloodUrea", "SerumCreatinine", "Sodium", "Potassium", "Hemoglobin", "CKD_Disease"
    ]
    rows = []

    for _ in range(num_samples):
        outcome = random.choices([0, 1], weights=[0.58, 0.42])[0]
        age = int(clamp(random.gauss(48, 14) if outcome == 0 else random.gauss(58, 12), 18, 85))

        if outcome == 0:
            bp = round(clamp(random.gauss(74, 8), 60, 100), 1)
            sg = random.choice([1.020, 1.025, 1.015])
            al = random.choices([0, 1], weights=[0.92, 0.08])[0]
            su = random.choices([0, 1], weights=[0.94, 0.06])[0]
            bu = round(clamp(random.gauss(32, 9), 14, 55), 1)
            sc = round(clamp(random.gauss(0.88, 0.18), 0.5, 1.3), 2)
            sod = round(clamp(random.gauss(140, 3.5), 132, 148), 1)
            pot = round(clamp(random.gauss(4.3, 0.4), 3.4, 5.2), 2)
            hemo = round(clamp(random.gauss(14.8, 1.4), 11.5, 17.5), 1)
        else:
            bp = round(clamp(random.gauss(86, 12), 70, 120), 1)
            sg = random.choice([1.010, 1.015, 1.005])
            al = random.choices([1, 2, 3, 4], weights=[0.35, 0.35, 0.20, 0.10])[0]
            su = random.choices([0, 1, 2, 3], weights=[0.60, 0.20, 0.12, 0.08])[0]
            bu = round(clamp(random.gauss(72, 32), 35, 190), 1)
            sc = round(clamp(random.gauss(3.1, 1.6), 1.4, 9.5), 2)
            sod = round(clamp(random.gauss(132, 5.5), 115, 142), 1)
            pot = round(clamp(random.gauss(4.8, 0.8), 3.2, 7.0), 2)
            hemo = round(clamp(random.gauss(10.2, 1.8), 6.0, 13.0), 1)

        rows.append([age, bp, sg, al, su, bu, sc, sod, pot, hemo, outcome])

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Generated specialized kidney disease dataset: {filepath} ({len(rows)} samples)")

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    multi_csv = os.path.join(data_dir, "early_disease_screening.csv")
    diabetes_csv = os.path.join(data_dir, "diabetes_prediction.csv")
    heart_csv = os.path.join(data_dir, "heart_disease_prediction.csv")
    kidney_csv = os.path.join(data_dir, "kidney_disease_prediction.csv")

    generate_multi_disease_dataset(multi_csv, num_samples=3500)
    generate_specialized_diabetes_dataset(diabetes_csv, num_samples=1800)
    generate_specialized_heart_dataset(heart_csv, num_samples=1800)
    generate_specialized_kidney_dataset(kidney_csv, num_samples=1600)
    print("All datasets generated successfully!")
