# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle
import gdown

# Set up the page layout
st.set_page_config(page_title="Vehicle Price Prediction App", layout="wide")

# Download the Random Forest model
model_url = "https://drive.google.com/uc?id=1p3nR1L-1lAO5mYl_1BVdTbNGNgp2aeEV"
gdown.download(model_url, "rf_model.pkl", quiet=False)

# Load the model and label encoders
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

make_model_df = pd.read_csv("make_model_pairs.csv")

# Create two tabs
tab1, tab2 = st.tabs(["Welcome", "Predict"])

# --- Tab 1: Welcome ---
with tab1:
    st.title("Welcome to the Vehicle Price Prediction App!")
    st.write("""
        This app helps Go Auto estimate optimal vehicle prices using machine learning. 
        It supports data-driven pricing strategies by analyzing vehicle characteristics such as make, model, mileage, and year.
    """)
    st.subheader("Problem")
    st.write("""
        Develop a regression model that predicts the optimal price of a vehicle based on its features 
        such as year, make, model, and mileage. The goal is to help Go Auto dealerships price vehicles competitively 
        and maximize sales performance.
    """)

# --- Tab 2: Predict ---
with tab2:
    st.title("Vehicle Price Prediction")
    st.markdown("Estimate vehicle price using a trained Random Forest model.")

    make = st.selectbox("Make", sorted(make_model_df["make"].unique()))
    filtered_models = make_model_df[make_model_df["make"] == make]["model"].unique()
    if len(filtered_models) == 0:
        filtered_models = label_encoders["model"].classes_

    model_name = st.selectbox("Model", sorted(filtered_models))
    model_year = st.slider("Model Year", 2010, 2024, 2017)
    mileage = st.number_input("Mileage (in KM)", value=50000)
    fuel_type = st.selectbox("Fuel Type", label_encoders["fuel_type_from_vin"].classes_)
    transmission = st.selectbox("Transmission", label_encoders["transmission_from_vin"].classes_)
    stock_type = st.selectbox("Stock Type", label_encoders["stock_type"].classes_)

    if st.button("Predict Price"):
        input_df = pd.DataFrame({
            "make": [make],
            "model": [model_name],
            "model_year": [model_year],
            "mileage": [mileage],
            "fuel_type_from_vin": [fuel_type],
            "transmission_from_vin": [transmission],
            "stock_type": [stock_type]
        })
        for col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Price: ${prediction:,.2f}")
