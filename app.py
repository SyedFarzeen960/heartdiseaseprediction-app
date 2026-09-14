import streamlit as st
import pandas as pd
import joblib

# Load trained model, scaler, and expected columns
model = joblib.load("KNN.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# App title
st.title("Heart Disease Prediction App")

st.markdown(
    "Provide the following details to predict heart disease. "
    "If you are unsure about any medical information, please consult a healthcare professional."
)

st.info(
    "⚠️ This app is for educational purposes only and is not a medical diagnosis tool."
)

# User inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

st.caption(
    "Sex: Select M for male or F for female."
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

st.caption(
    "ATA = Atypical Angina: chest discomfort that does not follow the typical pattern "
    "of heart-related chest pain.\n\n"
    "NAP = Non-Anginal Pain: chest pain that is less likely to be caused by the heart.\n\n"
    "TA = Typical Angina: chest discomfort that follows a pattern commonly associated "
    "with reduced blood flow to the heart.\n\n"
    "ASY = Asymptomatic: no typical chest pain symptoms."
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    80,
    200,
    120
)

st.caption(
    "Resting blood pressure is the pressure in your arteries while you are resting. "
    "It is measured in mm Hg."
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    100,
    600,
    200
)

st.caption(
    "Cholesterol is a substance in your blood. Higher levels of certain types of "
    "cholesterol can be associated with increased cardiovascular risk."
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

st.caption(
    "Fasting blood sugar is your blood glucose level after fasting. "
    "0 = No, fasting blood sugar is not above 120 mg/dL. "
    "1 = Yes, it is above 120 mg/dL."
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

st.caption(
    "ECG (electrocardiogram) records the electrical activity of the heart.\n\n"
    "Normal = no abnormality indicated.\n\n"
    "ST = an ST-segment abnormality was observed.\n\n"
    "LVH = Left Ventricular Hypertrophy, meaning the muscular wall of the "
    "heart's main pumping chamber is thicker than normal."
)

max_hr = st.slider(
    "Max Heart Rate",
    60,
    220,
    150
)

st.caption(
    "Maximum heart rate is the highest heart rate recorded during an exercise or "
    "stress test. It is measured in beats per minute (BPM)."
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

st.caption(
    "Exercise-induced angina means chest discomfort that occurs during physical "
    "activity or exercise.\n\n"
    "Y = Yes\n\n"
    "N = No"
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0,
    6.0,
    1.0
)

st.caption(
    "Oldpeak represents the amount of ST-segment depression observed during "
    "exercise compared with the resting ECG. It is usually obtained from an "
    "exercise stress test."
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

st.caption(
    "ST Slope describes the direction of the ST segment during exercise.\n\n"
    "Up = Upsloping\n\n"
    "Flat = Flat\n\n"
    "Down = Downsloping"
)

# Prediction
if st.button("Predict"):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,

        "ChestPainType_" + chest_pain: 1,

        "RestingECG_" + resting_ecg: 1,

        "ExerciseAngina_" + exercise_angina: 1,

        "ST_Slope_" + st_slope: 1
    }

    # Convert input into DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns with 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in exactly the same order as training
    input_df = input_df[expected_columns]

    # Use the already-trained scaler
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Display result
    if prediction[0] == 1:
        st.error(
            "High Risk of Heart Disease. Please consult a doctor."
        )
    else:
        st.success(
            "Low Risk of Heart Disease. Keep up the healthy lifestyle!"
        )

