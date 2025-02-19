# health_assistant.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import subprocess
import sys

# Function to install dependencies within the script
def install_dependencies():
    required_packages = ["streamlit", "pandas", "numpy", "scikit-learn", "joblib"]
    for package in required_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_dependencies()

# Load dataset (Dummy dataset for symptom-disease mapping)
data = {
    "Fever": [1, 0, 1, 0, 1],
    "Cough": [1, 1, 0, 0, 1],
    "Fatigue": [1, 1, 1, 0, 0],
    "Headache": [0, 1, 0, 1, 0],
    "Disease": ["Flu", "Cold", "Malaria", "Migraine", "COVID-19"]
}

df = pd.DataFrame(data)

# Preprocessing
X = df.drop(columns=["Disease"])
y = df["Disease"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train a simple ML model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "health_model.pkl")

# Load the model
model = joblib.load("health_model.pkl")

# Streamlit UI
st.title("AI-Powered Health Assistant")
st.write("Select your symptoms and get a possible disease prediction.")

# User input
symptoms = ["Fever", "Cough", "Fatigue", "Headache"]
user_input = [st.checkbox(symptom) for symptom in symptoms]

# Predict button
if st.button("Predict"):
    input_data = np.array([user_input]).astype(int)
    prediction = model.predict(input_data)
    predicted_disease = encoder.inverse_transform(prediction)
    st.success(f"Possible disease: {predicted_disease[0]}")

st.write("Disclaimer: This tool is for educational purposes only. Consult a doctor for medical advice.")
