import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction by Het",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------------------------
# Load Model, Scaler & Columns
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Logistic_Regression_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, expected_columns = load_artifacts()

# -------------------------------------------------
# App Title
# -------------------------------------------------
st.title("❤️ Heart Disease Prediction by Het")
st.markdown("Provide the following details to check your heart disease risk:")

# -------------------------------------------------
# User Input
# -------------------------------------------------
st.subheader("🧍 Personal Information")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])

st.subheader("🩺 Clinical Information")
col1, col2 = st.columns(2)

with col1:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

with col2:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Predict Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        f'Sex_{sex}': 1,
        f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Model prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # -------------------------------------------------
    # Results
    # -------------------------------------------------
    st.subheader("📊 Prediction Result")
    st.info(f"**Risk Probability:** {probability * 100:.2f}%")

    if probability >= 0.5:
        st.error("⚠️ High Risk of Heart Disease")
    elif probability >= 0.3:
        st.warning("🟠 Moderate Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    # -------------------------------------------------
    # Explanation (Updated based on Feature Importance)
    # -------------------------------------------------
    st.subheader("🧠 Key Observations")

    if st_slope in ["Flat", "Down"]:
        st.write("- ⚠️ Flat or Downward ST Slope detected (Strong risk indicator)")
    if chest_pain == "ASY":
        st.write("- ⚠️ Asymptomatic (ASY) chest pain reported")
    if exercise_angina == "Y":
        st.write("- ⚠️ Exercise-induced angina present")
    if oldpeak > 1.0:
        st.write("- ⚠️ Elevated ST depression (Oldpeak) detected")
    if sex == "M":
        st.write("- ℹ️ Male patients show a statistically higher baseline risk in this dataset")

# -------------------------------------------------
# Disclaimer
# -------------------------------------------------
st.caption(
    "⚠️ This application is for educational purposes only and is not a medical diagnosis."
)