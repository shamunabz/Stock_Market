# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pandas_datareader import data as pdr
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# # Data Collection
# def fetch_data(stock_symbol, start_date, end_date):
#     df = pdr.get_data_yahoo(stock_symbol, start=start_date, end=end_date)
#     df.reset_index(inplace=True)
#     return df

# # Data Preprocessing
# def preprocess_data(df, feature_cols, target_col):
#     df = df.dropna()
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(df[feature_cols])
    
#     # Creating dataset for LSTM
#     X, y = [], []
#     look_back = 60  # Number of previous days to consider for predicting the next day's price
#     for i in range(look_back, len(scaled_data)):
#         X.append(scaled_data[i-look_back:i])
#         y.append(scaled_data[i, feature_cols.index(target_col)])
#     X, y = np.array(X), np.array(y)
#     return X, y, scaler

# # Build the LSTM Model
# def build_model(input_shape):
#     model = Sequential([
#         LSTM(50, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(50, return_sequences=True),
#         Dropout(0.2),
#         LSTM(50),
#         Dropout(0.2),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Main execution function
# def main():
#     stock_symbol = 'AAPL'
#     start_date = '2010-01-01'
#     end_date = '2020-01-01'
    
#     df = fetch_data(stock_symbol, start_date, end_date)
#     feature_cols = ['Close', 'Volume']  # Features to consider
#     target_col = 'Close'  # Target variable
    
#     X, y, scaler = preprocess_data(df, feature_cols, target_col)
    
#     # Splitting data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model = build_model((X_train.shape[1], X_train.shape[2]))
#     model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
#     # Predicting and plotting the results
#     predicted_stock_price = model.predict(X_test)
#     predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
#     real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(real_stock_price, color='red', label='Real Stock Price')
#     plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
#     plt.title(f'{stock_symbol} Stock Price Prediction')
#     plt.xlabel('Time')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.show()

# if __name__ == "__main__":
#     main()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import time

def fetch_data(stock_symbol, start_date, end_date, attempts=3):
    for attempt in range(attempts):
        try:
            df = pdr.get_data_yahoo(stock_symbol, start=start_date, end=end_date)
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # wait a couple of seconds before retrying
    print(f"All attempts to fetch data failed.")
    return None

def preprocess_data(df, feature_cols, target_col, look_back=60):
    df = df.dropna()
    training_data = df[df['Date'] < '2018-01-01']  # Example split date
    test_data = df[df['Date'] >= '2018-01-01']

    scaler = MinMaxScaler(feature_range=(0, 1))
    training_scaled = scaler.fit_transform(training_data[feature_cols])

    X, y = [], []
    for i in range(look_back, len(training_scaled)):
        X.append(training_scaled[i-look_back:i])
        y.append(training_scaled[i, feature_cols.index(target_col)])
    X, y = np.array(X), np.array(y)
    
    # Preparing test data
    total_data = pd.concat((training_data[feature_cols], test_data[feature_cols]), axis=0)
    inputs = total_data[len(total_data) - len(test_data) - look_back:]
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(look_back, len(inputs)):
        X_test.append(inputs[i-look_back:i])
    X_test = np.array(X_test)

    return X, y, X_test, scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    stock_symbol = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2020-01-01'
    
    df = fetch_data(stock_symbol, start_date, end_date)
    if df is None:
        return
    
    feature_cols = ['Close', 'Volume']
    target_col = 'Close'
    
    X_train, y_train, X_test, scaler = preprocess_data(df, feature_cols, target_col)
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    # Assuming 'Date' column is available for plotting
    test_dates = df['Date'].iloc[-len(predicted_stock_price):]
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_dates, df['Close'].iloc[-len(predicted_stock_price):], color='red', label='Real Stock Price')
    plt.plot(test_dates, predicted_stock_price, color='blue', label='Predicted Stock Price')
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
