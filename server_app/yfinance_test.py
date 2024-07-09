import yfinance as yf
import pandas as pd
import ta
import numpy as np
import tensorflow as tf
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from data.config import *

def get_historical_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def add_technical_indicators(df: pd.DataFrame):
    return ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

def lstm_model_predict(
        df,
        underlyingname,
        sequence_length=40,
        test_size=40,
        batch_size=32,
        epochs=50
):
    # Use all features except 'Close' for prediction
    features = df.columns.drop('Close')
    X = df[features]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(X)

    # Prepare the data for the LSTM model
    x_data, y_data = [], []

    for i in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[i - sequence_length:i])
        y_data.append(df['Close'].iloc[i])

    x_data, y_data = np.array(x_data), np.array(y_data).reshape(-1, 1)

    # Normalize the target variable separately
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    y_data = target_scaler.fit_transform(y_data)

    # Split the data into training and testing sets
    train_size = len(x_data) - test_size
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]

    # Calculate number of units based on data size (example heuristic)
    num_units = int(np.sqrt(len(x_train[0])))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=num_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    #model.add(Dropout(0.2))
    model.add(LSTM(units=num_units, return_sequences=False))
    #model.add(Dropout(0.2))
    model.add(Dense(units=num_units // 2, activation='relu'))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    # Predict the stock prices
    predictions = model.predict(x_test)
    predictions = target_scaler.inverse_transform(predictions)

    # Calculate error metrics
    mse = mean_squared_error(target_scaler.inverse_transform(y_test), predictions)
    mae = mean_absolute_error(target_scaler.inverse_transform(y_test), predictions)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')

    # Prepare the validation data for plotting
    train = df[['Close']].iloc[:train_size + sequence_length]
    valid = df[['Close']].iloc[train_size + sequence_length:]

    valid['Predictions'] = np.nan
    valid['Predictions'].iloc[:len(predictions)] = predictions.reshape(-1)

    modelname = "LSTM"
    filename = f"data/plots/{modelname}_{underlyingname}.png"

    # Plot the results
    plt.figure(figsize=(16, 8))
    plt.title(f'Model: {modelname} Underlying {underlyingname}')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train.index, train['Close'], label='Train')
    plt.plot(valid.index, valid['Close'], label='Validation')
    plt.plot(valid.index, valid['Predictions'], label='Predictions')
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.text(valid.index[-1], valid['Predictions'].iloc[-1], f'MSE: {mse:.2f}\nMAE: {mae:.2f}',
             fontsize=12, verticalalignment='bottom')
    plt.savefig(filename)
    plt.show()

    return predictions
def hist_gradient_boosting_regressor_model_predict(df, test_size, underlyingname):
    # Use all features except 'Close' for prediction
    features = df.columns.drop('Close')
    X = df[features]
    y = df['Close']

    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # train
    est = HistGradientBoostingRegressor(random_state=42)
    est.fit(X_train, y_train)

    # predict
    predictions = est.predict(X_test)

    # Calculate error metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')

    # Prepare the validation data for plotting
    total_rows = len(df)
    train = df[['Close']].iloc[:total_rows - test_size]
    valid = df[['Close']].iloc[total_rows - test_size:]

    valid['Predictions'] = np.nan
    valid['Predictions'].iloc[:len(predictions)] = predictions.reshape(-1)

    modelname = "hist gradient boosting regressor"
    filename = f"data/plots/{modelname}_{underlyingname}.png"

    # Plot the results
    plt.figure(figsize=(16, 8))
    plt.title(f'Model: {modelname} Underlying {underlyingname}')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train.index, train['Close'], label='Train')
    plt.plot(valid.index, valid['Close'], label='Validation')
    plt.plot(valid.index, valid['Predictions'], label='Predictions')
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.text(valid.index[-1], valid['Predictions'].iloc[-1], f'MSE: {mse:.2f}\nMAE: {mae:.2f}',
             fontsize=12, verticalalignment='bottom')
    plt.savefig(filename)
    plt.show()

    return predictions

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == '__main__':
    ticker = 'MSFT'
    start_date = '2020-01-01'
    end_date = '2024-07-01'

    set_seeds(42)

    # all symbols
    for ticker in SYMBOLS:
        df = get_historical_data(ticker, start_date, end_date)
        df = add_technical_indicators(df)
        #lstm_predictions = lstm_model_predict(df, underlyingname=ticker)
        hist_gradient_boosting_regressor_predictions = hist_gradient_boosting_regressor_model_predict(df, test_size=50,
                                                                                                      underlyingname=ticker)

    # one symbol
    # df = get_historical_data(ticker, start_date, end_date)
    # df = add_technical_indicators(df)
    # #lstm_predictions = lstm_model_predict(df, underlyingname=ticker)
    # hist_gradient_boosting_regressor_predictions = hist_gradient_boosting_regressor_model_predict(df, test_size=50, underlyingname=ticker)


    print("done")
