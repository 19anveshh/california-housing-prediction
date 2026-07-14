import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained pipeline
model = joblib.load("models/california_housing_pipeline.joblib")

st.title("California Housing Price Predictor")
st.write("Enter the details below to predict the median house value.")

# Input fields for each original feature
MedInc = st.number_input("Median Income (in $10,000s)", min_value=0.0, max_value=20.0, value=3.5)
HouseAge = st.number_input("House Age (years)", min_value=0.0, max_value=100.0, value=25.0)
AveRooms = st.number_input("Average Rooms per Household", min_value=1.0, max_value=20.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms per Household", min_value=0.0, max_value=10.0, value=1.0)
Population = st.number_input("Population", min_value=0.0, max_value=40000.0, value=1000.0)
AveOccup = st.number_input("Average Occupancy per Household", min_value=0.5, max_value=20.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.0)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-118.0)

if st.button("Predict Price"):
    # Recreate engineered features same way as training
    RoomsPerHousehold = AveRooms / AveOccup
    BedroomsPerRoom = AveBedrms / AveRooms
    PopulationPerHousehold = Population / AveOccup

    input_df = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
        "RoomsPerHousehold": RoomsPerHousehold,
        "BedroomsPerRoom": BedroomsPerRoom,
        "PopulationPerHousehold": PopulationPerHousehold
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Median House Value: ${prediction * 100000:,.2f}")