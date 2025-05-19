import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

# Set Front End Page
st.set_page_config(page_title="Stock Predictor", layout="centered")
image = Image.open("chart_icon.png")
st.image(image, width=80)


# Title
st.markdown("""
    <h1 style='color:#004080;'>📈 Stock Trend Predictor <span style='font-size:22px;'>(TSLA vs NVDA)</span></h1>
    <p style='font-size:16px;'>Predict whether the stock price will go <b style='color:green;'>UP</b> 📈 or <b style='color:red;
            '>DOWN</b> 📉 based on recent 5-day returns using XGBoost or LSTM.</p>
""", unsafe_allow_html=True)

# User Inputs
company = st.selectbox("Select a Company", ["TSLA", "NVDA"])
model_type = st.selectbox("Choose a Model", ["XGBoost", "LSTM"])
input_text = st.text_input("Enter 5-day returns separated by commas (e.g., 0.01,0.02,-0.01,0.00,0.03)")

# Predict Button
if st.button("📊 Predict", type="primary"):
    try:
        returns = [float(i.strip()) for i in input_text.split(",")]
        if len(returns) != 5:
            st.error("❌ Please enter exactly 5 return values.")
        else:
            input_data = np.array(returns)
            scaler = joblib.load(f"{company.lower()}_scaler.pkl")

            if model_type == "XGBoost":
                scaled = scaler.transform(input_data.reshape(-1, 1)).reshape(1, -1)
                model = joblib.load(f"{company.lower()}_xgb_model.pkl")
                prediction = model.predict(scaled)[0]
            else:
                scaled = scaler.transform(input_data.reshape(-1, 1)).reshape(1, 5, 1)
                model = load_model(f"{company.lower()}_lstm_model.h5")
                prediction = int(model.predict(scaled)[0][0] > 0.5)

            result = "📈 Up" if prediction == 1 else "📉 Down"
            st.success(f"✅ {company} Prediction: {result}")
    except Exception as e:
        st.error("⚠️ Please enter valid comma-separated numbers.")

# Set Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0; bottom: 0; width: 100%;
    background-color: #4CAF50; color: white;
    text-align: center; padding: 10px;
}
.footer a {
    color: white; text-decoration: none; margin: 0 10px;
}
</style>
<div class="footer">
    Developed by Tesleem Oduola - ©2025<br>
    <a href="https://x.com/oduolates" target="_blank">Twitter</a>
    <a href="https://tesleemoduola.github.io" target="_blank">Portfolio</a>        
    <a href="https://www.linkedin.com/in/tesleemaderemioduola/" target="_blank">LinkedIn</a>
    <a href="https://github.com/Tesleemoduola" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)