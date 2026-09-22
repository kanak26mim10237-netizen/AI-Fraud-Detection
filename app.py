import streamlit as st
import pandas as pd
import joblib

st.title("💳 AI Fraud Detection System")

st.write("Enter transaction details to check whether a transaction may be fraudulent.")

amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
transaction_type = st.selectbox(
    "Transaction Type",
    ["Payment", "Transfer", "Cash Out", "Deposit"]
)

if st.button("Check Transaction"):
    try:
        model = joblib.load("fraud_model.pkl")

        data = pd.DataFrame({
            "amount": [amount],
            "transaction_type": [transaction_type]
        })

        prediction = model.predict(data)[0]

        if prediction == 1:
            st.error("⚠️ Potential Fraud Detected")
        else:
            st.success("✅ Transaction Appears Normal")

    except FileNotFoundError:
        st.warning("Please train the model first.")
