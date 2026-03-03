import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="agmsk86/Tourism_Package_Prediction_mohan", filename="best_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Travel Package Purchase Prediction
st.title("Travel Package Purchase Prediction App")
st.write("""
Predict whether a customer will purchase the travel package.
""")

# User Inputs
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
duration = st.sidebar.slider("Duration Of Pitch", 5, 40, 15)
occupation = st.sidebar.selectbox("Occupation", encoders['Occupation'].classes_)
gender = st.sidebar.selectbox("Gender", encoders['Gender'].classes_)
num_visit = st.sidebar.slider("Number Of Person Visiting", 1, 5, 2)
num_follow = st.sidebar.slider("Number Of Followups", 1, 10, 3)
product = st.sidebar.selectbox("Product Pitched", encoders['ProductPitched'].classes_)
property_star = st.sidebar.selectbox("Preferred Property Star", [3, 4, 5])
marital = st.sidebar.selectbox("Marital Status", encoders['MaritalStatus'].classes_)
trips = st.sidebar.slider("Number Of Trips", 0, 10, 2)
passport = st.sidebar.selectbox("Passport", [0, 1])
pitch_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car = st.sidebar.selectbox("Own Car", [0, 1])
children = st.sidebar.slider("Children Visiting", 0, 5, 0)
designation = st.sidebar.selectbox("Designation", encoders['Designation'].classes_)
income = st.sidebar.slider("Monthly Income", 10000, 50000, 20000)
contact = st.sidebar.selectbox("Type Of Contact", encoders['TypeofContact'].classes_)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': [age],
    'TypeofContact': [type_contact],
    'CityTier': [city_tier],
    'DurationOfPitch': [duration],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [num_visit],
    'NumberOfFollowups': [num_follow],
    'ProductPitched': [product],
    'PreferredPropertyStar': [property_star],
    'MaritalStatus': [marital],
    'NumberOfTrips': [trips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_score],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [children],
    'Designation': [designation],
    'MonthlyIncome': [income]
}])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Customer is likely to PURCHASE the travel package" if prediction == 1 else "Customer is NOT likely to PURCHASE the travel package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
