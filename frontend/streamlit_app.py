# Streamlit for frontend

import streamlit as st
import requests

st.title("🏠 House Price Prediction")

longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
housing_median_age = st.number_input("Housing Median Age")
total_rooms = st.number_input("Total Rooms")
total_bedrooms = st.number_input("Total Bedrooms")
population = st.number_input("Population")
households = st.number_input("Households")
median_income = st.number_input("Median Income")
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"]
)

if st.button("Predict"):

    data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",   # <-- fixed
        json=data
    )

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Predicted Price: ${prediction:,.2f}")
    else:
        st.error("Something went wrong with API")