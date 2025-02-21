import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys
import json
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def yahoo_finance(ticker, start_date, end_date):
    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    # Save data to CSV file
    data.to_csv(f'{ticker}_{start_date}_{end_date}.csv')
    return data

def price_plot(data):
    # Plot the closing price of the stock
    plt.figure(figsize=(10, 4))
    plt.plot(data['Low'], label='Low')
    plt.plot(data['High'], label='High')
    plt.plot(data['Close'], label='Close')
    plt.title('Close-Volume Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def create_lstm_model(data):
    # Prepare data for LSTM model
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create training and testing datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(dataset, time_step=1):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Shift train predictions for plotting
    train_predict_plot = np.empty_like(scaled_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

    # Shift test predictions for plotting
    test_predict_plot = np.empty_like(scaled_data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

    # Plot predictions
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data['Close'], label='Actual Price')
    plt.plot(data.index, train_predict_plot, label='Train Predict')
    plt.plot(data.index, test_predict_plot, label='Test Predict')
    plt.title('LSTM Model Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    # Predict future prices
    future_steps = 180  # Number of days to predict (6 months)
    last_data = scaled_data[-time_step:]
    future_predictions = []

    for _ in range(future_steps):
        last_data = last_data.reshape((1, time_step, 1))
        future_pred = model.predict(last_data)
        future_predictions.append(future_pred[0, 0])
        last_data = np.append(last_data[:, 1:, :], future_pred.reshape((1, 1, 1)), axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Plot future predictions
    future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, closed='right')
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data['Close'], label='Actual Price')
    plt.plot(future_dates, future_predictions, label='Future Predictions')
    plt.title('LSTM Model Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

data = yahoo_finance('AAPL', '2024-01-01', '2025-02-05')
price_plot(data)
create_lstm_model(data)