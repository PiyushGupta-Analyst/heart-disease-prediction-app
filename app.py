import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Heart Health AI",
    page_icon="❤️",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
}
.prediction-card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0 6px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load("heart_disease_model_KNN.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_assets()
except:
    st.error("❌ Model files not found. Please upload all .pkl files.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=100)
    st.title("About")
    st.info("""
    This AI system predicts the likelihood of heart disease using a **KNN model**.

    ⚠️ This is not a medical diagnosis tool.
    Always consult a doctor.
    """)

# --- HEADER ---
st.markdown("""
# ❤️ Heart Disease Diagnostic Assistant  
### AI-Powered Clinical Risk Assessment
""")

st.divider()

# --- INPUT UI ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Profile")
    age = st.slider("Age", 1, 110, 45)
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    chest_pain = st.selectbox("Chest Pain Type", ["ASY", "NAP", "ATA", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 600, 200)

with col2:
    st.subheader("🧪 Clinical Metrics")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1],
                              format_func=lambda x: "Yes" if x else "No")
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exercise_angina = st.radio("Exercise Induced Angina?", ["N", "Y"], horizontal=True)
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.divider()

# --- PREDICTION ---
if st.button("🔍 Generate Diagnostic Report", type="primary"):

    # === INPUT VALIDATION ===
    if resting_bp <= 0:
        st.error("❌ Resting Blood Pressure must be greater than 0")
        st.stop()
    if cholesterol < 0:
        st.error("❌ Cholesterol cannot be negative")
        st.stop()
    if max_hr < 60 or max_hr > 220:
        st.error("❌ Max Heart Rate must be between 60-220")
        st.stop()
    if oldpeak < 0:
        st.error("❌ Oldpeak cannot be negative")
        st.stop()

    # Step 1: Initialize ALL columns to 0 (SAFE METHOD)
    input_dict = {col: 0 for col in expected_columns}

    # Step 2: Fill numerical values
    input_dict.update({
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak
    })

    # Step 3: One-hot encoding (with fallback)
    try:
        input_dict[f"Sex_{sex}"] = 1
        input_dict[f"ChestPainType_{chest_pain}"] = 1
        input_dict[f"RestingECG_{resting_ecg}"] = 1
        input_dict[f"ExerciseAngina_{exercise_angina}"] = 1
        input_dict[f"ST_Slope_{st_slope}"] = 1
    except KeyError as e:
        st.error(f"❌ Column mismatch: {e}. Check model column names.")
        st.stop()

    # Step 4: Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_columns]

    # Step 5: Scale
    scaled_input = scaler.transform(input_df)

    # Step 6: Predict
    prediction = model.predict(scaled_input)[0]

    # FIXED: Proper KNN probability calculation
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(scaled_input)[0][1]
    else:
        # KNN distance-based confidence (normalized 0-1)
        distances, _ = model.kneighbors(scaled_input, n_neighbors=model.n_neighbors)
        # Convert distance to similarity (closer = higher confidence)
        max_distance = np.max(distances)
        prob = 1.0 - (np.mean(distances[0]) / max_distance) if max_distance > 0 else 0.5
        prob = max(0.01, min(0.99, prob))  # Clamp between 0.01-0.99

    # === RESULTS UI ===
    st.subheader("🧾 Diagnostic Report")
    st.caption(f"📅 Report generated: {datetime.datetime.now().strftime('%d %B %Y, %I:%M %p')}")  

    
    # Main results
    res1, res2 = st.columns([1, 2])

    with res1:
        if prediction == 1:
            st.error("### 🔴 HIGH RISK")
        else:
            st.success("### 🟢 LOW RISK")

        st.metric("Model Confidence", f"{prob*100:.1f}%")

    with res2:
        st.write("### 🧠 AI Assessment")
        progress_bar = st.progress(min(prob, 1.0))
        st.success(f"Risk Probability: **{prob*100:.1f}%**")
        
        if prob > 0.75:
            st.warning("⚠️ **Strong indication** of heart disease detected.")
        elif prob > 0.5:
            st.warning("⚠️ **Moderate risk** detected.")
        else:
            st.info("✅ **Low likelihood** of heart disease.")

    st.divider()

    # === FEATURE INSIGHTS ===
    st.subheader("📊 Key Risk Factors")
    
    # Dynamic feature importance based on input values
    risk_factors = {
        "Age": age / 100,
        "Cholesterol": min(cholesterol / 400, 1.0),
        "Chest Pain Type": 0.8 if chest_pain == "ASY" else 0.3,
        "Max Heart Rate": max(0, (220 - max_hr) / 160),
        "Exercise Angina": 0.9 if exercise_angina == "Y" else 0.1
    }
    
    st.bar_chart(risk_factors)
    
    # Show top 3 risk factors
    sorted_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
    st.write("**Top Risk Contributors:**")
    for factor, score in sorted_factors[:3]:
        st.write(f"• **{factor}** ({score*100:.0f}%)")

    st.divider()

    # === RECOMMENDATIONS ===
    st.subheader("🩺 Actionable Recommendations")
    
    if prediction == 1 or prob > 0.6:
        with st.container():
            st.error("""
            **🚨 URGENT: Seek Medical Attention**
            
            **Immediate Steps:**
            1. Contact cardiologist within 24-48 hours
            2. Avoid strenuous activity
            3. Monitor symptoms (chest pain, shortness of breath)
            
            **Recommended Tests:**
            - ECG / Stress Test
            - Echocardiogram  
            - Lipid Profile
            - Possible Angiography
            """)
    else:
        with st.container():
            st.success("""
            **✅ Good News - Low Risk**
            
            **Preventive Measures:**
            - 150+ min/week moderate exercise
            - Heart-healthy diet (Mediterranean)
            - Annual cardiac checkup
            - Manage cholesterol & blood pressure
            """)

    st.info("⚠️ **This is NOT a medical diagnosis.** Always consult a healthcare professional.")