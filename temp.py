import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.metrics import mean_squared_error

# Load the models and scaler
linear_model = joblib.load('car_price_model.pkl')  # Linear regression model
tree_model = joblib.load('decision_tree_model.pkl')  # Decision Tree model
rf_model = joblib.load('random_forest_model.pkl')  # Random Forest model
# StandardScaler or any other scaler used during training
scaler = joblib.load('scaler.pkl')

# Get the feature names from the scaler to ensure the order is correct
feature_names = scaler.feature_names_in_

# Function to predict car prices using all three models


def predict_prices(fuel_type, seller_type, transmission, present_price, kms_driven, owner, year):
    # Manual Encoding for Fuel_Type
    fuel_type_encoded = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[fuel_type]

    # Prepare the input data in the same format as during training
    input_data = pd.DataFrame([[present_price, kms_driven, owner, year, fuel_type_encoded, seller_type, transmission]],
                              columns=['Present_Price', 'Kms_Driven', 'Owner', 'Year', 'Fuel_Type', 'Seller_Type', 'Transmission'])

    # One-hot encoding for 'Seller_Type' and 'Transmission'
    input_data = pd.get_dummies(
        input_data, columns=['Seller_Type', 'Transmission'], drop_first=True)

    # Ensure all necessary columns are present, even if they're not in the current input
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the original feature names
    input_data = input_data[feature_names]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using all models
    predicted_price_linear = linear_model.predict(input_data_scaled)
    predicted_price_tree = tree_model.predict(input_data_scaled)
    predicted_price_rf = rf_model.predict(input_data_scaled)

    return predicted_price_linear[0], predicted_price_tree[0], predicted_price_rf[0], input_data.iloc[0]


# Streamlit UI setup
st.set_page_config(page_title="Car Price Prediction App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "Predict"])

if page == "Home":
    st.title("Welcome to the Car Price Prediction App")
    st.write("""
        This application allows users to predict the selling price of cars based on various features.
        
        ### Why Predict Car Prices?
        Car prices can fluctuate based on various factors such as:
        - Fuel type
        - Transmission type
        - Previous owners
        - Kms driven
        - Age of the car
        
        Understanding these factors can help sellers and buyers make informed decisions.
        
        Use the "Car Price Prediction" page to input details of a car and see the predicted selling price.
    """)

elif page == "Predict":
    st.title("Car Price Predictor")
    st.header("Enter Car Details")

    # Input fields
    car_name = st.text_input("Car Name")
    fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG'])
    seller_type = st.selectbox("Seller Type", options=['Dealer', 'Individual'])
    transmission = st.selectbox("Transmission", options=[
                                'Manual', 'Automatic'])
    present_price = st.number_input(
        "Present Price (in ₹ Lakhs)", min_value=0.0)
    kms_driven = st.number_input("Kms Driven", min_value=0)
    owner = st.number_input(
        "Owner (Number of Previous Owners)", min_value=0, max_value=10)
    year = st.number_input("Manufacturing Year",
                           min_value=1990, max_value=2024)

    # Predict button
    if st.button("Predict Price"):
        predicted_price_linear, predicted_price_tree, predicted_price_rf, input_data = predict_prices(
            fuel_type, seller_type, transmission, present_price, kms_driven, owner, year)

        # Display results
        st.success(f"The predicted selling price of the {car_name} is:")
        st.write(
            f"Linear Regression Model: ₹{predicted_price_linear:.2f} Lakhs")
        st.write(f"Decision Tree Model: ₹{predicted_price_tree:.2f} Lakhs")
        st.write(f"Random Forest Model: ₹{predicted_price_rf:.2f} Lakhs")

        # Comparison of models
        mse_linear = mean_squared_error(
            [predicted_price_linear], [predicted_price_linear])
        mse_tree = mean_squared_error(
            [predicted_price_tree], [predicted_price_tree])
        mse_rf = mean_squared_error([predicted_price_rf], [predicted_price_rf])

        if mse_rf < mse_tree and mse_rf < mse_linear:
            st.write("Random Forest model is more accurate.")
        elif mse_tree < mse_linear:
            st.write("Decision Tree model is more accurate.")
        else:
            st.write("Linear Regression model is more accurate.")

        # Explanation for the prediction
        st.subheader("Reason for the Prediction:")
        st.write("### Factors influencing the price prediction:")
        st.write(f"- Present price: ₹{present_price} Lakhs.")
        st.write(f"- Kilometers driven: {kms_driven}.")
        st.write(f"- Number of previous owners: {owner}.")
        st.write(f"- Manufacturing year: {year}.")
        st.write(f"- Fuel type: {fuel_type}.")
        st.write(
            f"- Seller type: {'Dealer' if seller_type == 'Dealer' else 'Individual'}.")
        st.write(f"- Transmission: {transmission}.")
