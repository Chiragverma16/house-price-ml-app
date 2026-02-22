import streamlit as st
import pandas as pd
from src.predict_pipeline import PredictPipeline, CustomData

st.set_page_config(page_title="California Housing Predictor")

st.title("🏠 California Housing Price Prediction")
st.write("Choose how you want to provide input.")

col1, col2 = st.columns(2)

user_input = st.radio("Select Input Type", ["Basic", "CSV Upload"])


# BASIC INPUT (Single Prediction)
if user_input == "Basic":

    with col1:
        longitude = st.number_input("Longitude", value=-122.23)
        latitude = st.number_input("Latitude", value=37.88)
        housing_median_age = st.number_input("Housing Median Age", min_value=1, value=41)
        total_rooms = st.number_input("Total Rooms", min_value=1, value=880)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=129)
        population = st.number_input("Population", min_value=1, value=322)
        households = st.number_input("Households", min_value=1, value=126)
        median_income = st.number_input("Median Income", min_value=0.0, value=8.3252)

        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["<1H OCEAN", "INLAND", "ISLAND", "NEAR OCEAN", "NEAR BAY"]
        )

    if st.button("Predict House Price"):
        data = CustomData(
            longitude=longitude,
            latitude=latitude,
            housing_median_age=housing_median_age,
            total_rooms=total_rooms,
            total_bedrooms=total_bedrooms,
            population=population,
            households=households,
            median_income=median_income,
            ocean_proximity=ocean_proximity
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.success(f"💰 Predicted House Price: ${results[0]:,.2f}")


# CSV UPLOAD (Batch Prediction)
elif user_input == "CSV Upload":

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df)

        if st.button("Run Batch Prediction"):

            with st.spinner("Making predictions..."):
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(df)

                df["Prediction"] = results

            st.success("Batch prediction completed!")
            st.dataframe(df)