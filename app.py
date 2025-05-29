import streamlit as st
import pickle
import pandas as pd
import numpy as np

linear_model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
logistic_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))


car = pd.read_csv('Cleaned Car.csv')

companies = sorted(car['company'].unique())
car_models_dict = {
    company: sorted(car[car['company'] == company]['name'].unique())
    for company in companies
}
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()


st.set_page_config(page_title="Car Price & Value App", layout="centered")
st.title("Car Value & Price App")
st.markdown(" This app **predicts your car's price** and tells whether it is considered **high value** in the market.")


company = st.selectbox("Select the company", ['Select Company'] + companies)

car_model = None
if company != 'Select Company':
    car_model = st.selectbox("Select the model", car_models_dict[company])

year = st.selectbox("Select Year of Purchase", years)
fuel_type = st.selectbox("Select the Fuel Type", fuel_types)
kilo_driven = st.text_input("Enter the Number of Kilometres the car has travelled", placeholder="e.g., 25000")

if st.button("Evaluate Car"):
    if company == 'Select Company' or not car_model or not kilo_driven.isdigit():
        st.error(" Please fill out all fields correctly.")
    else:
        input_df = pd.DataFrame(
            [[car_model, company, int(year), int(kilo_driven), fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )

        # Linear Prediction
        predicted_price = linear_model.predict(input_df)[0]
        st.subheader(" Estimated Price:")
        st.success(f"Rs. {np.round(predicted_price, 2)}")

        # Logistic Classification
        value_class = logistic_model.predict(input_df)[0]
        st.subheader(" Value Status:")
        if value_class == 1:
            st.success(" This car is considered **High Value**.")
        else:
            st.warning("️ This car is considered **Not High Value**.")
