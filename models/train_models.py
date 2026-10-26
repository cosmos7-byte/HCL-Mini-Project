"""
Early Disease Prediction - Machine Learning Model Training Engine
Trains, cross-validates, evaluates, and exports ML models:
1. Multi-Disease Classifier (Healthy, Diabetes, Heart Disease, Hypertension, Kidney Disease, Liver Disease)
2. Specialized Diabetes Classifier
3. Specialized Heart Disease Classifier
4. Specialized Chronic Kidney Disease (CKD) Classifier
Generates full test metrics, confusion matrices, and feature importance rankings.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVED_DIR = os.path.join(BASE_DIR, "models", "saved")
STATIC_REPORTS_DIR = os.path.join(BASE_DIR, "static", "reports")

os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(STATIC_REPORTS_DIR, exist_ok=True)

metrics_summary = {}

def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold')

    plt.ylabel('True Clinical Diagnosis', fontsize=10, fontweight='bold')
    plt.xlabel('AI Predicted Diagnosis', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_feature_importance(importances, feature_names, title, save_path):
    plt.figure(figsize=(9, 5))
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] for i in indices]

    bars = plt.barh(range(len(top_features)), top_importances[::-1], color='#2563eb', edgecolor='#1e40af')
    plt.yticks(range(len(top_features)), top_features[::-1], fontsize=9)
    plt.xlabel('Relative Feature Importance (Gini)', fontsize=10, fontweight='bold')
    plt.title(title, fontsize=12, fontweight='bold', pad=10)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def train_multi_disease_model():
    print("=" * 60)
    print("1. TRAINING MULTI-DISEASE SCREENING MODEL")
    print("=" * 60)
    csv_path = os.path.join(DATA_DIR, "early_disease_screening.csv")
    df = pd.read_csv(csv_path)

    feature_cols = [
        "age", "gender", "systolic_bp", "diastolic_bp", "cholesterol",
        "fasting_glucose", "bmi", "heart_rate", "serum_creatinine",
        "alt_liver_enzyme", "hba1c", "smoking", "physical_activity_hours",
        "family_history", "chest_pain_type", "fatigue_score"
    ]
    target_col = "target_disease"

    X = df[feature_cols]
    y = df[target_col]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidate_models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=14, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, max_depth=4, learning_rate=0.1, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42)
    }

    best_name = None
    best_model = None
    best_acc = 0
    candidate_scores = {}

    for name, model in candidate_models.items():
        model.fit(X_train_scaled, y_train)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        test_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        candidate_scores[name] = {
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(test_acc), 4)
        }
        print(f"  [{name}] CV Acc: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} | Test Acc: {test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_name = name
            best_model = model

    print(f"\n=> Selected Champion Model: {best_name} (Test Accuracy: {best_acc * 100:.2f}%)")

    # Evaluate best model
    best_pred = best_model.predict(X_test_scaled)
    prec = precision_score(y_test, best_pred, average="weighted")
    rec = recall_score(y_test, best_pred, average="weighted")
    f1 = f1_score(y_test, best_pred, average="weighted")
    cm = confusion_matrix(y_test, best_pred)
    clf_report = classification_report(y_test, best_pred, target_names=classes, output_dict=True)

    # Save artifacts
    joblib.dump(best_model, os.path.join(SAVED_DIR, "multi_disease_model.joblib"))
    joblib.dump(scaler, os.path.join(SAVED_DIR, "multi_disease_scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(SAVED_DIR, "multi_disease_encoder.joblib"))

    # Feature importances / Feature impacts
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        fi = np.mean(np.abs(best_model.coef_), axis=0)
        fi = fi / np.sum(fi)
    else:
        fi = None

    if fi is not None:
        plot_feature_importance(fi, feature_cols, f"Key Clinical Biomarkers ({best_name})", os.path.join(STATIC_REPORTS_DIR, "multi_disease_fi.png"))
        top_indices = np.argsort(fi)[::-1]
        top_factors = [{"feature": feature_cols[i], "importance": round(float(fi[i]), 4)} for i in top_indices]
    else:
        top_factors = []

    # Confusion matrix plot
    plot_confusion_matrix(cm, classes, f"Confusion Matrix: {best_name} Screening", os.path.join(STATIC_REPORTS_DIR, "multi_disease_cm.png"))

    metrics_summary["multi_disease"] = {
        "model_type": best_name,
        "classes": classes,
        "features": feature_cols,
        "accuracy": round(best_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "candidate_comparison": candidate_scores,
        "classification_report": clf_report,
        "top_features": top_factors
    }

def train_specialized_models():
    # 2. DIABETES SPECIALIZED MODEL
    print("\n" + "=" * 60)
    print("2. TRAINING SPECIALIZED DIABETES MODEL")
    print("=" * 60)
    df_dia = pd.read_csv(os.path.join(DATA_DIR, "diabetes_prediction.csv"))
    dia_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X_dia = df_dia[dia_features]
    y_dia = df_dia["Outcome"]

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dia, y_dia, test_size=0.2, random_state=42, stratify=y_dia)
    scaler_dia = StandardScaler()
    X_tr_sc_d = scaler_dia.fit_transform(X_train_d)
    X_te_sc_d = scaler_dia.transform(X_test_d)

    dia_rf = RandomForestClassifier(n_estimators=140, max_depth=10, random_state=42)
    dia_rf.fit(X_tr_sc_d, y_train_d)
    dia_pred = dia_rf.predict(X_te_sc_d)
    dia_acc = accuracy_score(y_test_d, dia_pred)
    dia_f1 = f1_score(y_test_d, dia_pred)
    print(f"  Diabetes Model Test Accuracy: {dia_acc:.4f} | F1: {dia_f1:.4f}")

    joblib.dump(dia_rf, os.path.join(SAVED_DIR, "diabetes_model.joblib"))
    joblib.dump(scaler_dia, os.path.join(SAVED_DIR, "diabetes_scaler.joblib"))

    cm_dia = confusion_matrix(y_test_d, dia_pred)
    plot_confusion_matrix(cm_dia, ["Non-Diabetic", "Diabetic"], "Diabetes Prediction Matrix", os.path.join(STATIC_REPORTS_DIR, "diabetes_cm.png"))
    plot_feature_importance(dia_rf.feature_importances_, dia_features, "Diabetes Diagnostic Factors", os.path.join(STATIC_REPORTS_DIR, "diabetes_fi.png"))

    metrics_summary["diabetes"] = {
        "features": dia_features,
        "accuracy": round(dia_acc, 4),
        "f1_score": round(dia_f1, 4),
        "precision": round(precision_score(y_test_d, dia_pred), 4),
        "recall": round(recall_score(y_test_d, dia_pred), 4)
    }

    # 3. HEART DISEASE SPECIALIZED MODEL
    print("\n" + "=" * 60)
    print("3. TRAINING SPECIALIZED HEART DISEASE MODEL")
    print("=" * 60)
    df_hd = pd.read_csv(os.path.join(DATA_DIR, "heart_disease_prediction.csv"))
    hd_features = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
    X_hd = df_hd[hd_features]
    y_hd = df_hd["HeartDisease"]

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_hd, y_hd, test_size=0.2, random_state=42, stratify=y_hd)
    scaler_hd = StandardScaler()
    X_tr_sc_h = scaler_hd.fit_transform(X_train_h)
    X_te_sc_h = scaler_hd.transform(X_test_h)

    hd_gb = GradientBoostingClassifier(n_estimators=130, max_depth=4, learning_rate=0.08, random_state=42)
    hd_gb.fit(X_tr_sc_h, y_train_h)
    hd_pred = hd_gb.predict(X_te_sc_h)
    hd_acc = accuracy_score(y_test_h, hd_pred)
    hd_f1 = f1_score(y_test_h, hd_pred)
    print(f"  Heart Disease Model Test Accuracy: {hd_acc:.4f} | F1: {hd_f1:.4f}")

    joblib.dump(hd_gb, os.path.join(SAVED_DIR, "heart_model.joblib"))
    joblib.dump(scaler_hd, os.path.join(SAVED_DIR, "heart_scaler.joblib"))

    cm_hd = confusion_matrix(y_test_h, hd_pred)
    plot_confusion_matrix(cm_hd, ["Normal / Low Risk", "Heart Disease"], "Heart Disease Prediction Matrix", os.path.join(STATIC_REPORTS_DIR, "heart_cm.png"))
    plot_feature_importance(hd_gb.feature_importances_, hd_features, "Heart Risk Diagnostic Factors", os.path.join(STATIC_REPORTS_DIR, "heart_fi.png"))

    metrics_summary["heart"] = {
        "features": hd_features,
        "accuracy": round(hd_acc, 4),
        "f1_score": round(hd_f1, 4),
        "precision": round(precision_score(y_test_h, hd_pred), 4),
        "recall": round(recall_score(y_test_h, hd_pred), 4)
    }

    # 4. CHRONIC KIDNEY DISEASE SPECIALIZED MODEL
    print("\n" + "=" * 60)
    print("4. TRAINING SPECIALIZED KIDNEY DISEASE MODEL")
    print("=" * 60)
    df_ckd = pd.read_csv(os.path.join(DATA_DIR, "kidney_disease_prediction.csv"))
    ckd_features = ["Age", "BloodPressure", "SpecificGravity", "Albumin", "Sugar", "BloodUrea", "SerumCreatinine", "Sodium", "Potassium", "Hemoglobin"]
    X_ckd = df_ckd[ckd_features]
    y_ckd = df_ckd["CKD_Disease"]

    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_ckd, y_ckd, test_size=0.2, random_state=42, stratify=y_ckd)
    scaler_ckd = StandardScaler()
    X_tr_sc_k = scaler_ckd.fit_transform(X_train_k)
    X_te_sc_k = scaler_ckd.transform(X_test_k)

    ckd_rf = RandomForestClassifier(n_estimators=130, max_depth=8, random_state=42)
    ckd_rf.fit(X_tr_sc_k, y_train_k)
    ckd_pred = ckd_rf.predict(X_te_sc_k)
    ckd_acc = accuracy_score(y_test_k, ckd_pred)
    ckd_f1 = f1_score(y_test_k, ckd_pred)
    print(f"  Kidney Disease Model Test Accuracy: {ckd_acc:.4f} | F1: {ckd_f1:.4f}")

    joblib.dump(ckd_rf, os.path.join(SAVED_DIR, "kidney_model.joblib"))
    joblib.dump(scaler_ckd, os.path.join(SAVED_DIR, "kidney_scaler.joblib"))

    cm_ckd = confusion_matrix(y_test_k, ckd_pred)
    plot_confusion_matrix(cm_ckd, ["Normal Kidney", "Early CKD"], "Kidney Disease Prediction Matrix", os.path.join(STATIC_REPORTS_DIR, "kidney_cm.png"))
    plot_feature_importance(ckd_rf.feature_importances_, ckd_features, "Kidney Function Biomarkers", os.path.join(STATIC_REPORTS_DIR, "kidney_fi.png"))

    metrics_summary["kidney"] = {
        "features": ckd_features,
        "accuracy": round(ckd_acc, 4),
        "f1_score": round(ckd_f1, 4),
        "precision": round(precision_score(y_test_k, ckd_pred), 4),
        "recall": round(recall_score(y_test_k, ckd_pred), 4)
    }

if __name__ == "__main__":
    train_multi_disease_model()
    train_specialized_models()

    summary_file = os.path.join(SAVED_DIR, "model_metrics.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED, EVALUATED & SERIALIZED SUCCESSFULLY!")
    print(f"Metrics saved to: {summary_file}")
    print("=" * 60)
