# AI-Based Fraud Detection System

## Project Description

This project is designed to detect fraudulent transactions using Artificial Intelligence and Machine Learning.

The system analyzes transaction data and uses a trained machine learning model to classify transactions as fraudulent or legitimate.

## Features

- Fraud transaction detection
- Machine Learning based prediction
- Transaction data analysis
- Fraud and legitimate transaction classification
- CSV-based transaction data upload
- Prediction result display

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Model

The project uses a Random Forest Classifier for fraud detection.

The trained model is saved as:

`fraud_model.pkl`

The model generates predictions where:

- `1` = Fraud
- `0` = Normal/Legitimate

## Dataset

The transaction dataset is stored in:

`data/fraud.csv`

The target column used for classification is:

`Class`

The remaining columns are used as input features for prediction.

## Project Structure

```text
AI-Fraud-Detection/
│
├── app.py
├── train_model.py
├── fraud_model.pkl
├── requirements.txt
├── README.md
│
└── data/
    └── fraud.csv
```
## How to Run the Project

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
streamlit run app.py
### 2. Run the Streamlit Application

```bash
streamlit run app.py

