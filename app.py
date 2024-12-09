import streamlit as st
import tensorflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# Load the pre-trained LSTM model (ensure the model is saved in your directory)
model = load_model('improved_lstm_stock_model.h5')

# Load the scaler (use the same scaler you used for training)
# scaler = load_scaler('scaler.pkl')

# Function to predict stock prices
def predict_stock_price(model, data, sequence_length=100):
    """
    Predict stock prices using a trained LSTM model.

    Parameters:
    - model: The trained LSTM model.
    - data: The historical stock price data (in the original scale).
    - sequence_length: The number of previous days used for prediction (default is 100).

    Returns:
    - predicted_prices: The predicted stock prices.
    - actual_prices: The actual stock prices corresponding to the prediction period.
    """
    # Scale the data using the same scaler that was used during training
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare input sequences for prediction
    X_input = []
    for i in range(len(scaled_data) - sequence_length):
        X_input.append(scaled_data[i:i + sequence_length, 0])

    X_input = np.array(X_input)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

    # Make predictions
    predictions = model.predict(X_input)

    # Inverse transform the predictions back to original scale
    predicted_prices = scaler.inverse_transform(predictions)

    # Actual prices (use the data after the sequence_length for comparison)
    actual_prices = data[sequence_length:]

    return predicted_prices, actual_prices


# Streamlit App Interface
def main():
    st.title("Stock Price Prediction Using LSTM")
    st.markdown("This app predicts stock prices based on historical data using an LSTM model.")

    # File upload for the user to upload their data
    uploaded_file = st.file_uploader("Upload Stock Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded data
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the data:", df.head())

        # Select the stock price column (assumed 'Close' column for stock price)
        stock_data = df['close'].values

        # Show options for users to choose sequence length
        sequence_length = st.slider("Select Sequence Length", 50, 200, 100)

        # Load the pre-trained model (commented out for now)
        # model = load_model('path_to_your_model.h5')

        # Make predictions using the model
        predicted_prices, actual_prices = predict_stock_price(model, stock_data, sequence_length)

        # Plot the predictions
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(actual_prices, label="Actual Prices", color='blue')
        ax.plot(predicted_prices, label="Predicted Prices", color='red')
        ax.set_title("Stock Price Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Display the predicted prices
        st.write("Predicted Stock Prices:", predicted_prices.flatten())


if __name__ == "__main__":
    main()
