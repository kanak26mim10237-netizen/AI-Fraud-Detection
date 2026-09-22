import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI Fraud Detection", page_icon="💳")

st.title("💳 AI Fraud Detection System")
st.write("Upload transaction data to detect fraudulent transactions.")

# Load trained model
try:
    model = joblib.load("fraud_model.pkl")
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("fraud_model.pkl not found. Please train the model first.")
    st.stop()

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Transaction Data")
    st.dataframe(data.head())

    if "Class" in data.columns:

        X = data.drop("Class", axis=1)

        predictions = model.predict(X)

        data["Prediction"] = predictions

        st.subheader("Fraud Detection Results")
        st.dataframe(data)

        fraud = (predictions == 1).sum()
        normal = (predictions == 0).sum()

        st.write("### Results")
        st.write("🚨 Fraud Transactions:", fraud)
        st.write("✅ Normal Transactions:", normal)

    else:
        st.error("CSV file must contain a 'Class' column.")
