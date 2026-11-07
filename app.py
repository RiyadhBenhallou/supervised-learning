import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern card UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #E63946;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #457B9D;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1D3557;
        margin-bottom: 10px;
    }
    .risk-high {
        background-color: #FFE5E5;
        border-left: 6px solid #D90429;
        padding: 20px;
        border-radius: 10px;
        color: #2B2D42;
    }
    .risk-low {
        background-color: #E8F8F5;
        border-left: 6px solid #2EC4B6;
        padding: 20px;
        border-radius: 10px;
        color: #2B2D42;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. LOAD MODEL WITH CACHING
# ==========================================
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "heart_disease_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please verify model export.")
        return None
    return joblib.load(model_path)

model = load_model()

# ==========================================
# 3. SIDEBAR: METRICS & PRESETS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=70)
    st.title("CardioCare ML")
    st.caption("Supervised Learning Clinical Decision Support")

    st.markdown("---")
    st.subheader("📊 Model Performance")
    st.markdown(
        """
        - **Algorithm:** Tuned `LogisticRegression`
        - **Test Accuracy:** **88.52%**
        - **ROC-AUC Score:** **0.93**
        - **5-Fold Cross-Val Recall:** **92.16%**
        - **5-Fold Cross-Val F1:** **86.73%**
        """
    )

    st.markdown("---")
    st.subheader("⚡ Quick Test Profiles")
    preset = st.radio(
        "Load sample profile:",
        options=["Custom Patient", "Healthy Profile (Low Risk)", "High Risk Profile", "Dataset Patient 1"],
        index=0,
    )

    st.markdown("---")
    st.info(
        "💡 **Disclaimer:** This tool is for educational & demonstration purposes only and should not replace professional medical diagnosis."
    )

# ==========================================
# 4. PRESET VALUES SELECTION
# ==========================================
# Default values based on preset selection
if preset == "Healthy Profile (Low Risk)":
    def_age, def_sex = 38, "Female (0)"
    def_cp = "0: Typical Angina"
    def_trestbps, def_chol, def_fbs = 115, 175, "No (< 120 mg/dl)"
    def_restecg = "0: Normal"
    def_thalach = 178
    def_exang = "No (0)"
    def_oldpeak = 0.2
    def_slope = "0: Upsloping"
    def_ca = 0
    def_thal = "1: Normal"
elif preset == "High Risk Profile":
    def_age, def_sex = 64, "Male (1)"
    def_cp = "2: Non-Anginal Pain"
    def_trestbps, def_chol, def_fbs = 160, 286, "Yes (> 120 mg/dl)"
    def_restecg = "1: ST-T Wave Abnormality"
    def_thalach = 108
    def_exang = "Yes (1)"
    def_oldpeak = 2.8
    def_slope = "2: Downsloping"
    def_ca = 2
    def_thal = "3: Reversible Defect"
elif preset == "Dataset Patient 1":
    def_age, def_sex = 63, "Male (1)"
    def_cp = "3: Asymptomatic"
    def_trestbps, def_chol, def_fbs = 145, 233, "Yes (> 120 mg/dl)"
    def_restecg = "0: Normal"
    def_thalach = 150
    def_exang = "No (0)"
    def_oldpeak = 2.3
    def_slope = "0: Upsloping"
    def_ca = 0
    def_thal = "1: Normal"
else:  # Custom Patient (Cleveland dataset medians)
    def_age, def_sex = 55, "Male (1)"
    def_cp = "1: Atypical Angina"
    def_trestbps, def_chol, def_fbs = 130, 240, "No (< 120 mg/dl)"
    def_restecg = "1: ST-T Wave Abnormality"
    def_thalach = 153
    def_exang = "No (0)"
    def_oldpeak = 0.8
    def_slope = "1: Flat"
    def_ca = 0
    def_thal = "2: Fixed Defect"

# ==========================================
# 5. MAIN HEADER
# ==========================================
st.markdown('<div class="main-title">❤️ Heart Disease Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter patient clinical metrics to evaluate heart disease probability in real time using our tuned machine learning pipeline.</div>',
    unsafe_allow_html=True,
)

# ==========================================
# 6. PATIENT INPUT FORM
# ==========================================
with st.container():
    col1, col2, col3 = st.columns(3)

    # Column 1: Demographics & Vitals
    with col1:
        st.markdown("### 👤 Demographics & Vitals")
        age = st.slider("Age (years)", min_value=20, max_value=85, value=def_age, help="Age of patient")
        sex = st.radio("Sex", options=["Female (0)", "Male (1)"], index=0 if "Female" in def_sex else 1, horizontal=True)
        trestbps = st.slider(
            "Resting Blood Pressure (`trestbps`) [mm Hg]",
            min_value=80,
            max_value=200,
            value=def_trestbps,
            help="Blood pressure upon hospital admission (normal < 120 mm Hg)",
        )
        chol = st.slider(
            "Serum Cholesterol (`chol`) [mg/dl]",
            min_value=120,
            max_value=570,
            value=def_chol,
            help="Normal desirable cholesterol is below 200 mg/dl",
        )
        fbs = st.radio(
            "Fasting Blood Sugar > 120 mg/dl (`fbs`)",
            options=["No (< 120 mg/dl)", "Yes (> 120 mg/dl)"],
            index=0 if "No" in def_fbs else 1,
            horizontal=True,
        )

    # Column 2: Cardiac & Exercise Symptoms
    with col2:
        st.markdown("### 🫀 Cardiac Symptoms")
        cp_options = [
            "0: Typical Angina",
            "1: Atypical Angina",
            "2: Non-Anginal Pain",
            "3: Asymptomatic",
        ]
        cp_idx = [i for i, opt in enumerate(cp_options) if opt.startswith(def_cp[:1])][0]
        cp = st.selectbox(
            "Chest Pain Type (`cp`)",
            options=cp_options,
            index=cp_idx,
            help="Non-anginal and atypical pain showed higher risk in this cohort",
        )
        thalach = st.slider(
            "Max Heart Rate Achieved (`thalach`) [bpm]",
            min_value=60,
            max_value=220,
            value=def_thalach,
            help="Maximum heart rate recorded during peak cardiovascular exercise",
        )
        exang = st.radio(
            "Exercise Induced Angina (`exang`)",
            options=["No (0)", "Yes (1)"],
            index=0 if "No" in def_exang else 1,
            horizontal=True,
            help="Chest pain triggered during exercise test",
        )
        oldpeak = st.slider(
            "ST Depression (`oldpeak`) [mm]",
            min_value=0.0,
            max_value=6.2,
            value=float(def_oldpeak),
            step=0.1,
            help="ST depression induced by exercise relative to rest",
        )
        slope_options = ["0: Upsloping", "1: Flat", "2: Downsloping"]
        slope_idx = [i for i, opt in enumerate(slope_options) if opt.startswith(def_slope[:1])][0]
        slope = st.selectbox(
            "Peak Exercise ST Slope (`slope`)",
            options=slope_options,
            index=slope_idx,
            help="Slope of peak exercise ST segment",
        )

    # Column 3: Diagnostic Tests & Scans
    with col3:
        st.markdown("### 🔬 Clinical & Scan Tests")
        restecg_options = [
            "0: Normal",
            "1: ST-T Wave Abnormality",
            "2: Left Ventricular Hypertrophy",
        ]
        restecg_idx = [i for i, opt in enumerate(restecg_options) if opt.startswith(def_restecg[:1])][0]
        restecg = st.selectbox("Resting ECG (`restecg`)", options=restecg_options, index=restecg_idx)

        ca = st.slider(
            "Major Vessels Colored by Fluoroscopy (`ca`)",
            min_value=0,
            max_value=4,
            value=def_ca,
            help="Number of major vessels (0-3) with clear blood flow under fluoroscopy",
        )

        thal_options = [
            "1: Normal",
            "2: Fixed Defect",
            "3: Reversible Defect",
            "0: Unknown / Other",
        ]
        thal_idx = [i for i, opt in enumerate(thal_options) if opt.startswith(def_thal[:1])][0]
        thal = st.selectbox(
            "Thallium Stress Result (`thal`)",
            options=thal_options,
            index=thal_idx,
            help="Perfusion scan showing blood flow to cardiac muscle tissue",
        )

# ==========================================
# 7. PARSE INPUT DATA FOR MODEL
# ==========================================
input_data = pd.DataFrame([
    {
        "age": int(age),
        "sex": 1 if "Male" in sex else 0,
        "cp": int(cp[0]),
        "trestbps": int(trestbps),
        "chol": int(chol),
        "fbs": 1 if "Yes" in fbs else 0,
        "restecg": int(restecg[0]),
        "thalach": int(thalach),
        "exang": 1 if "Yes" in exang else 0,
        "oldpeak": float(oldpeak),
        "slope": int(slope[0]),
        "ca": int(ca),
        "thal": int(thal[0]),
    }
])

# ==========================================
# 8. PREDICTION & RESULTS DISPLAY
# ==========================================
st.markdown("---")
col_btn, col_fill = st.columns([1, 4])
with col_btn:
    predict_clicked = st.button("🔮 Calculate Risk", type="primary", use_container_width=True)

if predict_clicked or preset != "Custom Patient":
    if model is not None:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        prob_healthy = probabilities[0]
        prob_disease = probabilities[1]

        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="risk-high">
                        <h2 style="color: #D90429; margin-top: 0;">🚨 High Risk: Heart Disease Indicated</h2>
                        <p style="font-size: 1.15rem;">The model predicts a <b>{prob_disease:.1%}</b> probability of heart disease presence based on the clinical biomarkers provided.</p>
                        <hr style="border-color: #F8D7DA;"/>
                        <b>Clinical Recommendation:</b> Suggest prompt review by a cardiologist, including confirmatory echocardiography and full cardiovascular risk assessment.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-low">
                        <h2 style="color: #0F9D58; margin-top: 0;">🟢 Low Risk: Normal Cardiac Markers</h2>
                        <p style="font-size: 1.15rem;">The model predicts an <b>{prob_healthy:.1%}</b> probability of no significant heart disease based on current parameters.</p>
                        <hr style="border-color: #D4EDDA;"/>
                        <b>Clinical Recommendation:</b> Continue standard preventive wellness guidelines, regular lipid profile screenings, and cardiovascular exercise.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.markdown("### 📊 Probability Breakdown")
            st.write(f"**Heart Disease (Target = 1):** `{prob_disease:.2%}`")
            st.progress(float(prob_disease))

            st.write(f"**No Disease (Target = 0):** `{prob_healthy:.2%}`")
            st.progress(float(prob_healthy))

            st.caption(f"Decision Boundary Threshold: 50.0% | Model Confidence: {max(prob_disease, prob_healthy):.1%}")

# ==========================================
# 9. EXPLANATIONS & MODEL INSIGHTS TABS
# ==========================================
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📋 Current Patient Record", "🔍 Feature Importance & Model Explanations", "📈 Training Performance Plots"])

with tab1:
    st.subheader("Input Vector Passed to Model")
    st.dataframe(input_data, use_container_width=True)

with tab2:
    st.subheader("Key Clinical Drivers (Logistic Regression Weights)")
    st.markdown(
        """
        Our model calculates log-odds using learned coefficients. Here are the most significant features influencing predictions:
        - **Risk Drivers (+):**
          - `cp` (+0.66): Non-anginal or atypical chest pain significantly increases the probability of disease.
          - `slope` (+0.45): Upsloping ST changes correlate with positive diagnoses.
          - `restecg` (+0.32): ST-T wave abnormalities increase suspicion.
        - **Protective / Negative Drivers (-):**
          - `ca` (-0.64): More major vessels visible under fluoroscopy indicates healthier coronary circulation.
          - `exang` (-0.60): Absence of exercise-induced angina lowers the estimated risk.
          - `oldpeak` (-0.57): Lower ST depression points away from disease.
        """
    )
    if os.path.exists("plots/11_feature_importance.png"):
        st.image("plots/11_feature_importance.png", caption="Model Feature Importance (Logistic Regression Coefficients)", use_container_width=True)

with tab3:
    st.subheader("Evaluation Visualizations from Training Pipeline")
    plot_cols = st.columns(3)
    with plot_cols[0]:
        if os.path.exists("plots/06_model_comparison.png"):
            st.image("plots/06_model_comparison.png", caption="Baseline Model Accuracy Comparison")
    with plot_cols[1]:
        if os.path.exists("plots/08_roc_curve.png"):
            st.image("plots/08_roc_curve.png", caption="Receiver Operating Characteristic (ROC-AUC = 0.93)")
    with plot_cols[2]:
        if os.path.exists("plots/09_confusion_matrix.png"):
            st.image("plots/09_confusion_matrix.png", caption="Confusion Matrix on Holdout Test Set")

