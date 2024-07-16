import os
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from data.config import *
from sklearn.linear_model import LinearRegression

def get_historical_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def add_technical_indicators(df: pd.DataFrame):
    return ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)


def sklearn_model_predict(est, df, test_size, underlyingname, modelname):
    # 'High', 'Low', 'Close', "Adj Close" are unknown at forecasting
    # 'others_cr' seems to represent more or less the Close Price
    # "Close" is predicted
    features = df.columns.drop(['High', 'Low', 'Close', "Adj Close", "others_cr"])
    X = df[features]
    y = df['Close']

    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # train
    est.fit(X_train, y_train)

    # predict
    # predictions = est.predict(X_test)

    predictions = []

    for index, row in X_test.iterrows():
        # prediction
        current_row = row.values.reshape(1, -1)
        prediction = est.predict(current_row)[0]
        predictions.append(prediction)

        # adjust technical indicators

        pass

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

    # filename for the plot
    filename = f"../data/plots/{modelname}/{underlyingname}.png"

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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

    return mse, mae

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

if __name__ == '__main__':
    ticker = 'MSFT'
    start_date = '2020-01-01'
    end_date = '2024-07-01'

    set_seeds(42)

    # DataFrame to store results
    results_df = pd.DataFrame(columns=['Ticker', 'Model', 'MSE', 'MAE'])

    # all symbols
    for ticker in SYMBOLS:
        print(f"Ticker: {ticker}")

        df = get_historical_data(ticker, start_date, end_date)
        df = add_technical_indicators(df)

        # # HistGradientBoostingRegressor
        # est = HistGradientBoostingRegressor(random_state=42)
        # modelname = "HistGradientBoostingRegressor"
        # mse, mae = sklearn_model_predict(est, df, test_size=50, underlyingname=ticker, modelname=modelname)
        # results_df = pd.concat([results_df, pd.DataFrame({'Ticker': [ticker], 'Model': [modelname], 'MSE': [mse], 'MAE': [mae]})], ignore_index=True)
        #
        # # RandomForestRegressor
        # est = RandomForestRegressor(random_state=42, n_jobs=-1)
        # modelname = "RandomForestRegressor"
        # mse, mae = sklearn_model_predict(est, df, test_size=50, underlyingname=ticker, modelname=modelname)
        # results_df = pd.concat([results_df, pd.DataFrame({'Ticker': [ticker], 'Model': [modelname], 'MSE': [mse], 'MAE': [mae]})], ignore_index=True)

        # LinearRegression
        est = LinearRegression(n_jobs=-1)
        modelname = "LinearRegression"
        mse, mae = sklearn_model_predict(est, df, test_size=50, underlyingname=ticker, modelname=modelname)
        results_df = pd.concat([results_df, pd.DataFrame({'Ticker': [ticker], 'Model': [modelname], 'MSE': [mse], 'MAE': [mae]})], ignore_index=True)


        # Save results to disk
        results_filename = "../data/model_performance_results.csv"
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        print(f"store results in {results_filename} after ticker symbol: {ticker}")
        results_df.to_csv(results_filename, index=False)

    print("done")
