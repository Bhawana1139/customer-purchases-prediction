import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="Bhawana12345/customer-purchases-prediction",
    filename="best_customer-purchases-prediction_model_v1.joblib",
    repo_type="space"
)
model = joblib.load(model_path)

# Streamlit UI for ProdTaken Prediction
st.title("ProdTaken Prediction App")
st.write("""
This application predicts the **ProdTaken** based on personal and lifestyle details.
Please enter the required information below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=61, value=30, step=1)
Gender = st.selectbox("Gender", ["male", "female"])
TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("CityTier", ["Tier 1", "Tier 2", "Tier 3"])
DurationOfPitch = st.number_input("DurationOfPitch",min_value=5, max_value=127, value=30, step=1)
Occupation = st.selectbox("Occupation", ["Salaried" , "Small Business","Large Business","Free Lancer"])                              
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=22, value=10, step=1)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=2, step=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=3, step=1)
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe"])
MaritalStatus= st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
Passport= st.selectbox("Passport", ["Yes", "No"])
OwnCar= st.selectbox("OwnCar", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=2, step=1)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000.0, max_value=50000.0,value=10000.0, step=100.0)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=1, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'Gender': Gender,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'NumberOfFollowups': NumberOfFollowups

}])

# Prediction
if st.button("ProdTaken"):
    prediction = model.predict(input_data)[0]
    st.subheader("ProdTaken Result:")
    st.success(f"Estimated ProdTaken: **${prediction:,.2f}**")
