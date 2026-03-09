import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
df = pd.read_excel('churn_dataset.xlsx')

# Preprocessing
label_map_churn = {'No': 0, 'Yes': 1}
label_map_sex   = {'Male': 1, 'Female': 0}
df["Churn"] = df["Churn"].map(label_map_churn)
df["Sex"]   = df["Sex"].map(label_map_sex)

# Train model
features = ["Age", "Tenure", "Sex"]
x = df[features]
y = df["Churn"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Guess The Churn 🤔")
st.write(f"Model Accuracy: **{acc:.2%}**")

st.header("Predict Churn for a New Customer")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
sex = st.selectbox("Sex", ["Male", "Female"])

sex_encoded = 1 if sex == "Male" else 0

if st.button("Predict"):
    input_data = pd.DataFrame([[age, tenure, sex_encoded]], columns=features)
    prediction = model.predict(input_data)[0]
    result = "Will Churn ❌" if prediction == 1 else "Will Not Churn ✅"
    st.subheader(f"Prediction: {result}")

