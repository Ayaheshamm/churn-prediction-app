import streamlit as st
import pandas as pd
import joblib

st.title("Guess The Churn")

st.write("Predict whether a customer will churn using Naive Bayes")

model = joblib.load("Churn_model.pkl")

age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure (months)", 0, 120)
sex = st.selectbox("Sex", ["Male", "Female"])

if sex == "Male":
    sex = 1
else:
    sex = 0

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age":[age],
        "Tenure":[tenure],
        "Sex":[sex]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer Will Churn")
    else:
        st.success("Customer Will Stay")
