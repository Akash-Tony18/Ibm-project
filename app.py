import streamlit as st
import pandas as pd
import joblib

# Load preprocessor and model
preprocessor = joblib.load("salary_prediction_preprocessor.pkl")
model = joblib.load("salary_prediction_model.pkl")

st.title("Salary Category Predictor 💼")

# User Input Form
with st.form("user_input_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'State-gov', 'Federal-gov', 'Local-gov'])
    fnlwgt = st.number_input("Fnlwgt", min_value=10000, value=50000)
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college'])
    education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced'])
    occupation = st.selectbox("Occupation", ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty'])
    relationship = st.selectbox("Relationship", ['Not-in-family', 'Husband', 'Wife'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Cuba'])

    submit = st.form_submit_button("Predict Salary")

# Process and Predict
if submit:
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

    try:
        processed = preprocessor.transform(input_data)
        prediction = model.predict(processed)
        st.success(f"Predicted Salary Category: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
