/**
 * MediPredict AI - Client Application Logic
 * Handles tab navigation, preset patient loading, API prediction calls, and visualization rendering.
 */

// Sample patient profiles for instant testing
const sampleProfiles = {
    healthy: {
        age: 28, gender: 0, bmi: 22.4, systolic_bp: 116, diastolic_bp: 74,
        heart_rate: 68, fasting_glucose: 86, hba1c: 5.0, cholesterol: 170,
        serum_creatinine: 0.85, alt_liver_enzyme: 22, smoking: 0,
        physical_activity_hours: 5.0, family_history: 0, chest_pain_type: 0, fatigue_score: 2
    },
    diabetes: {
        age: 54, gender: 1, bmi: 33.5, systolic_bp: 138, diastolic_bp: 86,
        heart_rate: 79, fasting_glucose: 182, hba1c: 8.4, cholesterol: 228,
        serum_creatinine: 1.05, alt_liver_enzyme: 34, smoking: 0,
        physical_activity_hours: 1.0, family_history: 1, chest_pain_type: 0, fatigue_score: 7
    },
    cardiac: {
        age: 61, gender: 1, bmi: 29.8, systolic_bp: 154, diastolic_bp: 96,
        heart_rate: 88, fasting_glucose: 112, hba1c: 5.8, cholesterol: 275,
        serum_creatinine: 1.15, alt_liver_enzyme: 29, smoking: 1,
        physical_activity_hours: 0.5, family_history: 1, chest_pain_type: 2, fatigue_score: 8
    },
    hypertension: {
        age: 56, gender: 0, bmi: 30.1, systolic_bp: 164, diastolic_bp: 102,
        heart_rate: 82, fasting_glucose: 104, hba1c: 5.6, cholesterol: 218,
        serum_creatinine: 1.0, alt_liver_enzyme: 26, smoking: 0,
        physical_activity_hours: 1.5, family_history: 1, chest_pain_type: 0, fatigue_score: 5
    },
    kidney: {
        age: 59, gender: 1, bmi: 27.6, systolic_bp: 146, diastolic_bp: 92,
        heart_rate: 76, fasting_glucose: 118, hba1c: 6.1, cholesterol: 210,
        serum_creatinine: 3.20, alt_liver_enzyme: 30, smoking: 0,
        physical_activity_hours: 1.0, family_history: 0, chest_pain_type: 0, fatigue_score: 8
    },
    liver: {
        age: 51, gender: 1, bmi: 31.0, systolic_bp: 132, diastolic_bp: 84,
        heart_rate: 74, fasting_glucose: 116, hba1c: 5.8, cholesterol: 242,
        serum_creatinine: 0.90, alt_liver_enzyme: 115, smoking: 1,
        physical_activity_hours: 1.0, family_history: 0, chest_pain_type: 0, fatigue_score: 7
    }
};

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initMultiDiseaseForm();
    initSpecializedForms();
    fetchMetrics();
});

// Tab Navigation
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const panels = document.querySelectorAll(".tab-panel");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.getAttribute("data-tab");
            tabBtns.forEach(b => b.classList.remove("active"));
            panels.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(target).classList.add("active");
        });
    });
}

// Load Preset Clinical Profile
function loadSampleProfile(type) {
    const data = sampleProfiles[type];
    if (!data) return;

    document.getElementById("m_age").value = data.age;
    document.getElementById("m_gender").value = data.gender;
    document.getElementById("m_bmi").value = data.bmi;
    document.getElementById("m_systolic_bp").value = data.systolic_bp;
    document.getElementById("m_diastolic_bp").value = data.diastolic_bp;
    document.getElementById("m_heart_rate").value = data.heart_rate;
    document.getElementById("m_fasting_glucose").value = data.fasting_glucose;
    document.getElementById("m_hba1c").value = data.hba1c;
    document.getElementById("m_cholesterol").value = data.cholesterol;
    document.getElementById("m_serum_creatinine").value = data.serum_creatinine;
    document.getElementById("m_alt_liver").value = data.alt_liver_enzyme;
    document.getElementById("m_smoking").value = data.smoking;
    document.getElementById("m_physical_activity").value = data.physical_activity_hours;
    document.getElementById("m_family_history").value = data.family_history;
    document.getElementById("m_chest_pain").value = data.chest_pain_type;
    document.getElementById("m_fatigue").value = data.fatigue_score;
    document.getElementById("fatigue_val").innerText = data.fatigue_score;

    // Trigger prediction automatically
    document.getElementById("multi-disease-form").dispatchEvent(new Event("submit"));
}

// Multi-Disease Form Submission
function initMultiDiseaseForm() {
    const form = document.getElementById("multi-disease-form");
    const submitBtn = document.getElementById("btn-predict-multi");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        submitBtn.disabled = true;
        submitBtn.querySelector(".btn-text").innerText = "Analyzing Health Biomarkers...";

        const payload = {
            age: parseFloat(document.getElementById("m_age").value),
            gender: parseInt(document.getElementById("m_gender").value),
            bmi: parseFloat(document.getElementById("m_bmi").value),
            systolic_bp: parseFloat(document.getElementById("m_systolic_bp").value),
            diastolic_bp: parseFloat(document.getElementById("m_diastolic_bp").value),
            heart_rate: parseFloat(document.getElementById("m_heart_rate").value),
            fasting_glucose: parseFloat(document.getElementById("m_fasting_glucose").value),
            hba1c: parseFloat(document.getElementById("m_hba1c").value),
            cholesterol: parseFloat(document.getElementById("m_cholesterol").value),
            serum_creatinine: parseFloat(document.getElementById("m_serum_creatinine").value),
            alt_liver_enzyme: parseFloat(document.getElementById("m_alt_liver").value),
            smoking: parseInt(document.getElementById("m_smoking").value),
            physical_activity_hours: parseFloat(document.getElementById("m_physical_activity").value),
            family_history: parseInt(document.getElementById("m_family_history").value),
            chest_pain_type: parseInt(document.getElementById("m_chest_pain").value),
            fatigue_score: parseInt(document.getElementById("m_fatigue").value)
        };

        try {
            const res = await fetch("/api/predict/multi", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (data.success) {
                renderMultiDiseaseResult(data);
            } else {
                alert("Diagnostic analysis error: " + (data.detail || "Server error"));
            }
        } catch (err) {
            console.error(err);
            alert("Connection error. Make sure the backend server is running.");
        } finally {
            submitBtn.disabled = false;
            submitBtn.querySelector(".btn-text").innerText = "Run Early Disease Diagnostic Screening";
        }
    });
}

// Render Results View
function renderMultiDiseaseResult(data) {
    document.getElementById("empty-state").classList.add("hidden");
    const resultBox = document.getElementById("diagnostic-result");
    resultBox.classList.remove("hidden");

    const badge = document.getElementById("diagnosis-badge");
    const diseaseName = document.getElementById("res-disease-name");
    const riskTier = document.getElementById("res-risk-tier");
    const confText = document.getElementById("res-confidence");
    const confBar = document.getElementById("res-confidence-bar");

    diseaseName.innerText = data.predicted_disease;
    riskTier.innerText = data.risk_level;
    confText.innerText = `${data.confidence}%`;
    confBar.style.width = `${data.confidence}%`;

    // Badge styling
    badge.className = "diagnosis-badge";
    if (data.predicted_disease === "Healthy") {
        badge.classList.add("healthy");
        confBar.style.background = "linear-gradient(90deg, #10b981, #059669)";
    } else if (data.risk_level.includes("High")) {
        badge.classList.add("danger");
        confBar.style.background = "linear-gradient(90deg, #ef4444, #dc2626)";
    } else {
        badge.classList.add("warning");
        confBar.style.background = "linear-gradient(90deg, #f59e0b, #d97706)";
    }

    // Probability breakdown
    const probContainer = document.getElementById("res-prob-list");
    probContainer.innerHTML = "";
    const sortedProbs = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([disease, pct]) => {
        const row = document.createElement("div");
        row.className = "prob-row";
        const barColor = disease === data.predicted_disease ? (disease === "Healthy" ? "#10b981" : "#ef4444") : "#94a3b8";

        row.innerHTML = `
            <span class="prob-name">${disease}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${pct}%; background: ${barColor}"></div>
            </div>
            <span class="prob-percent">${pct.toFixed(1)}%</span>
        `;
        probContainer.appendChild(row);
    });

    // Biomarker Alerts
    const bioContainer = document.getElementById("res-biomarkers");
    bioContainer.innerHTML = "";
    if (data.abnormal_biomarkers && data.abnormal_biomarkers.length > 0) {
        data.abnormal_biomarkers.forEach(b => {
            const tag = document.createElement("span");
            tag.className = "biomarker-tag alert";
            tag.innerHTML = `⚠️ <b>${b.name}:</b> ${b.value} (${b.status})`;
            bioContainer.appendChild(tag);
        });
    } else {
        const tag = document.createElement("span");
        tag.className = "biomarker-tag optimal";
        tag.innerHTML = "✓ All routine hemodynamic & metabolic markers within clinical baseline.";
        bioContainer.appendChild(tag);
    }

    // Recommendations
    const recContainer = document.getElementById("res-recommendations");
    recContainer.innerHTML = "";
    data.recommendations.forEach(rec => {
        const li = document.createElement("li");
        li.innerText = rec;
        recContainer.appendChild(li);
    });
}

// Specialized Forms Logic
function initSpecializedForms() {
    // Diabetes
    document.getElementById("form-specialized-diabetes").addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            pregnancies: parseInt(document.getElementById("dia_preg").value),
            glucose: parseFloat(document.getElementById("dia_gluc").value),
            blood_pressure: parseFloat(document.getElementById("dia_bp").value),
            skin_thickness: parseFloat(document.getElementById("dia_skin").value),
            insulin: parseFloat(document.getElementById("dia_ins").value),
            bmi: parseFloat(document.getElementById("dia_bmi").value),
            dpf: parseFloat(document.getElementById("dia_dpf").value),
            age: parseInt(document.getElementById("dia_age").value)
        };

        const res = await fetch("/api/predict/diabetes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            const box = document.getElementById("dia-result");
            box.classList.remove("hidden");
            document.getElementById("dia-res-title").innerText = data.prediction;
            document.getElementById("dia-res-title").style.color = data.is_diabetic ? "#ef4444" : "#10b981";
            document.getElementById("dia-res-prob").innerText = `Diabetic Risk Probability: ${data.probability}% (${data.risk_level})`;
            document.getElementById("dia-res-rec").innerText = data.recommendation;
        }
    });

    // Heart Disease
    document.getElementById("form-specialized-heart").addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            age: parseInt(document.getElementById("hd_age").value),
            sex: parseInt(document.getElementById("hd_sex").value),
            chest_pain_type: parseInt(document.getElementById("hd_cpt").value),
            resting_bp: parseFloat(document.getElementById("hd_rbp").value),
            cholesterol: parseFloat(document.getElementById("hd_chol").value),
            fasting_bs: parseInt(document.getElementById("hd_fbs").value),
            resting_ecg: parseInt(document.getElementById("hd_recg").value),
            max_hr: parseInt(document.getElementById("hd_maxhr").value),
            exercise_angina: parseInt(document.getElementById("hd_exang").value),
            oldpeak: parseFloat(document.getElementById("hd_oldpeak").value),
            st_slope: parseInt(document.getElementById("hd_sts").value)
        };

        const res = await fetch("/api/predict/heart", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            const box = document.getElementById("hd-result");
            box.classList.remove("hidden");
            document.getElementById("hd-res-title").innerText = data.prediction;
            document.getElementById("hd-res-title").style.color = data.has_heart_disease ? "#ef4444" : "#10b981";
            document.getElementById("hd-res-prob").innerText = `Cardiovascular Risk: ${data.probability}% (${data.risk_level})`;
            document.getElementById("hd-res-rec").innerText = data.recommendation;
        }
    });

    // Kidney Disease
    document.getElementById("form-specialized-kidney").addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            age: parseInt(document.getElementById("kd_age").value),
            blood_pressure: parseFloat(document.getElementById("kd_bp").value),
            specific_gravity: parseFloat(document.getElementById("kd_sg").value),
            albumin: parseInt(document.getElementById("kd_al").value),
            sugar: parseInt(document.getElementById("kd_su").value),
            blood_urea: parseFloat(document.getElementById("kd_bu").value),
            serum_creatinine: parseFloat(document.getElementById("kd_sc").value),
            sodium: parseFloat(document.getElementById("kd_sod").value),
            potassium: parseFloat(document.getElementById("kd_pot").value),
            hemoglobin: parseFloat(document.getElementById("kd_hemo").value)
        };

        const res = await fetch("/api/predict/kidney", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            const box = document.getElementById("kd-result");
            box.classList.remove("hidden");
            document.getElementById("kd-res-title").innerText = data.prediction;
            document.getElementById("kd-res-title").style.color = data.has_ckd ? "#ef4444" : "#10b981";
            document.getElementById("kd-res-prob").innerText = `Renal Disease Probability: ${data.probability}% (${data.risk_level})`;
            document.getElementById("kd-res-rec").innerText = data.recommendation;
        }
    });
}

// Fetch Metrics for Analytics Tab
async function fetchMetrics() {
    try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        if (data.multi_disease) {
            const md = data.multi_disease;
            document.getElementById("stat-accuracy").innerText = `${(md.accuracy * 100).toFixed(1)}%`;
            document.getElementById("md-algo").innerText = md.model_type || "Random Forest";
            document.getElementById("md-acc").innerText = `${(md.accuracy * 100).toFixed(1)}%`;
            document.getElementById("md-prec").innerText = `${(md.precision * 100).toFixed(1)}%`;
            document.getElementById("md-rec").innerText = `${(md.recall * 100).toFixed(1)}%`;
            document.getElementById("md-f1").innerText = `${(md.f1_score * 100).toFixed(1)}%`;
        }
        if (data.diabetes) {
            const dia = data.diabetes;
            document.getElementById("dia-acc").innerText = `${(dia.accuracy * 100).toFixed(1)}%`;
            document.getElementById("dia-prec").innerText = `${(dia.precision * 100).toFixed(1)}%`;
            document.getElementById("dia-rec").innerText = `${(dia.recall * 100).toFixed(1)}%`;
            document.getElementById("dia-f1").innerText = `${(dia.f1_score * 100).toFixed(1)}%`;
        }
        if (data.heart) {
            const hd = data.heart;
            document.getElementById("hd-acc").innerText = `${(hd.accuracy * 100).toFixed(1)}%`;
            document.getElementById("hd-prec").innerText = `${(hd.precision * 100).toFixed(1)}%`;
            document.getElementById("hd-rec").innerText = `${(hd.recall * 100).toFixed(1)}%`;
            document.getElementById("hd-f1").innerText = `${(hd.f1_score * 100).toFixed(1)}%`;
        }
        if (data.kidney) {
            const kd = data.kidney;
            document.getElementById("kd-acc").innerText = `${(kd.accuracy * 100).toFixed(1)}%`;
            document.getElementById("kd-prec").innerText = `${(kd.precision * 100).toFixed(1)}%`;
            document.getElementById("kd-rec").innerText = `${(kd.recall * 100).toFixed(1)}%`;
            document.getElementById("kd-f1").innerText = `${(kd.f1_score * 100).toFixed(1)}%`;
        }
    } catch (e) {
        console.warn("Could not fetch metrics yet:", e);
    }
}
