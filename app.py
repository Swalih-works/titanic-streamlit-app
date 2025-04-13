import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details to predict their survival probability:")

# Input fields
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.radio("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
Fare = st.number_input("Fare Paid", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode inputs
Sex_encoded = 0 if Sex == "male" else 1
Embarked_map = {"S": 0, "C": 1, "Q": 2}
Embarked_encoded = Embarked_map[Embarked]

# Prepare input
features = np.array([[Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]
    st.success(f"Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
    st.info(f"Survival Probability: {proba:.2%}")
