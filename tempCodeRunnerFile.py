# GENDER -> 1 FEMALE  0 MALE
# SENIORCITIZEN -> 1 YES -> 0 NO
# SCLAER IS EXPORTED AS SCALER.PKL
# MODEL IS EXPORTED AS CHURN_MODEL.PKL
# ORDER OF THE X->'GENDER', 'TENURE','MONTHLYCHARGES'



import streamlit as st
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")

st.divider()

st.write("Enter the following details to predict whether the customer will churn or not:")

st.divider()

tenure = st.number_input("Enter the tenure (in months):", min_value=0, max_value=130, value=12)

monthly_charges = st.number_input("Enter the monthly charges:", min_value=0.0, max_value=200.0, value=70.0)   

gender = st.selectbox("enter the gender", options=["Male", "Female"])

st.divider()

predictbutton = st.button("Predict Churn")

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    x = [gender_selected, tenure, monthly_charges]

    x1 = np.array(x)

    x_array = scaler.transform([x1])

    prediction = model.predict(x_array)[0]

    predicted = "yes" if prediction == 1 else "No"

    st.write(f"The predicted result is: {predicted}")

else:
    st.write("please enter the values and use predict button")
