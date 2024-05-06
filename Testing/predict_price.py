import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Fetch stock data
def fetch_data(stock_symbol, start_date, end_date):
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Preprocess data
def preprocess_data(df, feature_col='Close', look_back=60):
    df = df[['Date', feature_col]].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_col].values.reshape(-1,1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshaping for LSTM
    return X, y, scaler, df['Date'][look_back:]

# Build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot the results
def plot_results(test_dates, real_data, predicted_data):
    plt.figure(figsize=(14, 5))
    plt.plot(test_dates, real_data, color='red', label='Real Stock Price')
    plt.plot(test_dates, predicted_data, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main function
def main():
    stock_symbol = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2020-01-01'
    df = fetch_data(stock_symbol, start_date, end_date)
    
    if df is not None:
        X, y, scaler, dates = preprocess_data(df)
        
        # Split data into train and test
        split_percent = 0.80
        split = int(split_percent * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        test_dates = dates[split:]

        # Build and train the model
        model = build_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

        # Predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Evaluate the model
        rmse = np.sqrt(mean_squared_error(real_prices, predictions))
        print(f"Root Mean Squared Error: {rmse}")

        # Visualization
        plot_results(test_dates, real_prices, predictions)

if __name__ == "__main__":
    main()
