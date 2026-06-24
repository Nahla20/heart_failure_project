import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About")
st.sidebar.write("""
This app predicts heart disease risk using Machine Learning.

### Features Used:
- Age
- Blood Pressure
- Cholesterol
- Heart Rate
- ECG & others

### Model:
- Random Forest
""")

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "artifacts/models/heart_model.pkl")
scaler_path = os.path.join(BASE_DIR, "artifacts/models/scaler.pkl")
columns_path = os.path.join(BASE_DIR, "artifacts/models/columns.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(columns_path)
except Exception as e:
    st.error("❌ Error loading model files")
    st.write(e)
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
# ❤️ Heart Disease Prediction Dashboard
### 🩺 AI-Powered Health Risk Analysis
""")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Patient Information")
    Age = st.slider("Age", 20, 100, 40)
    Sex = st.selectbox("Sex", ["M", "F"])
    ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    RestingBP = st.slider("Resting Blood Pressure", 80, 200, 120)
    Cholesterol = st.slider("Cholesterol", 100, 400, 200)
    FastingBS_option = st.selectbox(
    "Fasting Blood Sugar",
    ["Normal", "High"],
    help="Normal: ≤ 120 mg/dL | High: > 120 mg/dL"
)

# Convert to model input
    FastingBS = 1 if FastingBS_option == "High" else 0

with col2:
    st.subheader("📊 Medical Details")
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.slider("Max Heart Rate", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise Angina", ["Y", "N"])
    Oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------- HISTORY ----------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ---------------- PREDICT ----------------
if st.button("🔍 Predict Heart Disease"):

    if RestingBP == 0 or Cholesterol == 0:
        st.warning("⚠️ Please enter valid medical values")
    else:
        try:
            with st.spinner("Analyzing patient data..."):

                input_dict = {
                    'Age': Age,
                    'Sex': Sex,
                    'ChestPainType': ChestPainType,
                    'RestingBP': RestingBP,
                    'Cholesterol': Cholesterol,
                    'FastingBS': FastingBS,
                    'RestingECG': RestingECG,
                    'MaxHR': MaxHR,
                    'ExerciseAngina': ExerciseAngina,
                    'Oldpeak': Oldpeak,
                    'ST_Slope': ST_Slope
                }

                input_df = pd.DataFrame([input_dict])

                # Feature Engineering
                input_df['AgeGroup'] = pd.cut(
                    input_df['Age'],
                    bins=[20,40,60,100],
                    labels=['Young','Middle','Senior']
                )

                input_df['CholesterolLevel'] = pd.cut(
                    input_df['Cholesterol'],
                    bins=[0,200,240,600],
                    labels=['Normal','Borderline','High']
                )

                # Encoding
                input_df = pd.get_dummies(input_df)

                # Match columns
                input_df = input_df.reindex(columns=columns, fill_value=0)

                # Scaling
                num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
                input_df[num_cols] = scaler.transform(input_df[num_cols])

                # Prediction
                prediction = model.predict(input_df)
                prob = model.predict_proba(input_df)[0][1]

                st.session_state.history.append(prob)

            # ---------------- RESULT ----------------
            st.markdown("---")
            st.subheader("📌 Prediction Result")

            st.metric("Risk Probability", f"{prob*100:.2f}%")
            st.progress(float(prob))

            # Risk Zone
            if prob < 0.4:
                st.success("🟢 Safe Zone")
            elif prob < 0.7:
                st.warning("🟠 Caution Zone")
            else:
                st.error("🔴 Danger Zone")

            # Confidence
            if prob > 0.8:
                st.write("📊 Model Confidence: Very High")
            elif prob > 0.6:
                st.write("📊 Model Confidence: High")
            elif prob > 0.4:
                st.write("📊 Model Confidence: Medium")
            else:
                st.write("📊 Model Confidence: Low")

            # Risk Summary
            st.markdown(f"""
            ### 🧾 Risk Summary
            - **Probability:** {prob*100:.2f}%
            - **Risk Level:** {"High" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"}
            """)

            # ---------------- SUMMARY ----------------
            st.subheader("🧾 Patient Summary")
            st.dataframe(input_df.style.highlight_max(axis=1))

            # ---------------- DOWNLOAD ----------------
            result_df = pd.DataFrame({
                "Risk Probability": [prob]
            })

            st.download_button(
                label="📥 Download Result",
                data=result_df.to_csv(index=False),
                file_name="prediction.csv",
                mime="text/csv"
            )

            # ---------------- ABNORMAL ALERTS ----------------
            st.subheader("⚠️ Abnormal Indicators")

            if Cholesterol > 240:
                st.warning("High Cholesterol")

            if RestingBP > 140:
                st.warning("High Blood Pressure")

            if MaxHR < 100:
                st.warning("Low Heart Rate")

            # ---------------- HEALTH TIPS ----------------
            if prediction[0] == 1:
                st.info("💡 Advice: Maintain healthy diet, exercise regularly, avoid stress, and consult a doctor.")
            else:
                st.info("💡 Great! Continue maintaining a healthy lifestyle.")

            # ---------------- VISUALIZATION ----------------
            st.subheader("📊 Patient Health Overview")

            chart_data = pd.DataFrame({
                'Feature': ['Age', 'BP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
                'Value': [Age, RestingBP, Cholesterol, MaxHR, Oldpeak]
            })

            st.bar_chart(chart_data.set_index('Feature'))

            # ---------------- COMPARISON ----------------
            st.subheader("📊 Health Comparison")

            normal_values = [50, 120, 200, 150, 1.0]

            compare_df = pd.DataFrame({
                'Feature': ['Age', 'BP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
                'Your Value': [Age, RestingBP, Cholesterol, MaxHR, Oldpeak],
                'Normal': normal_values
            })

            st.bar_chart(compare_df.set_index('Feature'))

            # ---------------- HISTORY ----------------
            st.subheader("📈 Prediction History")
            st.line_chart(st.session_state.history)

            history_df = pd.DataFrame({
                "Prediction Probability": st.session_state.history
            })

            st.subheader("📋 Prediction History Table")
            st.dataframe(history_df)

            # ---------------- EXPLANATION ----------------
            st.subheader("🧠 Why this prediction?")
            st.write("""
            The prediction is mainly influenced by:
            - Cholesterol level
            - Oldpeak (ECG depression)
            - Max Heart Rate
            """)

        except Exception as e:
            st.error("❌ Prediction Error")
            st.write(e)

# ---------------- DISCLAIMER ----------------
st.markdown("---")
st.caption("⚠️ This is a machine learning prediction and not a medical diagnosis. Please consult a doctor.")