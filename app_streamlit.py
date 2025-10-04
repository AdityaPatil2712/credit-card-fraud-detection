import streamlit as st
import joblib
import pandas as pd
from predict import predict_single, load_model

st.set_page_config(page_title="Credit Card Fraud Detection Demo")
st.title("💳 Credit Card Fraud Detection")

@st.cache_resource
def load_resources():
    data = joblib.load("models/model.joblib")
    return data["model"], data["scaler"]

model, scaler = load_resources()

st.write("Enter feature values for a transaction:")

Time = st.number_input("Time", value=0.0)
Amount = st.number_input("Amount", value=100.0)

sample = {"Time": Time, "Amount": Amount}
for i in range(1, 29):
    sample[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

if st.button("Predict"):
    pred, prob = predict_single(sample, model, scaler)
    if pred == 1:
        st.error(f"🚨 Predicted FRAUD (probability {prob:.3f})")
    else:
        st.success(f"✅ Predicted LEGITIMATE (fraud probability {prob:.3f})")
