import os
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from data.config import *
from sklearn.linear_model import LinearRegression

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def get_historical_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def add_technical_indicators(df: pd.DataFrame):
    return ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

def data_preprocessing_train_test_split(df: pd.DataFrame, test_size: int):
    # 'others_cr' seems to represent more or less the Close Price
    # "trend_sma_fast" sometimes gets nan -> ignore this
    # "trend_sma_slow" sometimes gets nan -> ignore this
    # "Close" is predicted
    features = df.columns.drop(['Close', "Adj Close", "others_cr", 'trend_sma_fast', 'trend_sma_slow'])
    X = df[features]
    y = df['Close']

    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test

def sklearn_model_predict(est, df, test_size, underlyingname, modelname):
    X_train, X_test, y_train, y_test = data_preprocessing_train_test_split(df, test_size)

    # train model
    est.fit(X_train, y_train)

    # set the X_test dataframe values to 0. The values must be predicted and calculated based on the previous value
    X_test.iloc[:, :] = 0

    # predict
    predictions = []
    mean_volumn_last_10_days = X_train['Volume'].iloc[-10:].mean()
    last_index = X_test.index[-1]

    # to predict the first day of test data we need the values from the last train day
    # for the technical analysis we need to set Open, High, Low, Close and Volume
    X_test.at[X_test.index[0], 'Open'] = X_train['Open'].iloc[-1]
    X_test.at[X_test.index[0], 'High'] = X_train['Open'].iloc[-1]
    X_test.at[X_test.index[0], 'Low'] = X_train['Open'].iloc[-1]
    X_test.at[X_test.index[0], 'Close'] = X_train['Open'].iloc[-1]
    X_test.at[X_test.index[0], 'Volume'] = X_train['Volume'].iloc[-1]

    for index, row in X_test.iterrows():
        # ensure we go not out of bounds
        if index == last_index:
            break

        # merge train and test data as base for the technical indicators
        temp_df = pd.concat([X_train, df["Close"][:-test_size]], axis=1)
        temp_df = pd.concat([temp_df, X_test[:index]], axis=0)
        temp_df = temp_df[["Open", "High", "Low", "Close", "Volume"]]

        # calculate technical indicators
        temp_df = add_technical_indicators(temp_df)

        # ensure drop not needed columns from the temp dataframe
        temp_df = temp_df[X_train.columns]

        # prediction
        current_row = temp_df.loc[index].to_frame().T
        prediction = est.predict(current_row)
        predictions.append(prediction)

        # find next index
        next_index = X_test.index[X_test.index.get_loc(index) + 1]

        # set predicted close price as open on the next index (=next trading day)
        # todo High Low and volume predcitions in own models just go with fantasy values for now
        X_test.at[next_index, 'Open'] = prediction
        X_test.at[next_index, 'Close'] = prediction
        X_test.at[next_index, 'High'] = prediction * 1.05
        X_test.at[next_index, 'Low'] = prediction * 0.95
        X_test.at[next_index, 'Volume'] = mean_volumn_last_10_days

    # todo last prediction should be handled diffrent but just test this
    predictions.append(prediction)

    # Convert predictions list to a NumPy array
    predictions = np.array(predictions).flatten()

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
    # tf.random.set_seed(seed)

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
        df = add_technical_indicators(df.copy())

        # # HistGradientBoostingRegressor
        # est = HistGradientBoostingRegressor(random_state=42)
        # modelname = "HistGradientBoostingRegressor"
        # mse, mae = sklearn_model_predict(est, df, test_size=50, underlyingname=ticker, modelname=modelname)
        # results_df = pd.concat([results_df, pd.DataFrame({'Ticker': [ticker], 'Model': [modelname], 'MSE': [mse], 'MAE': [mae]})], ignore_index=True)
        #
        # RandomForestRegressor
        est = RandomForestRegressor(random_state=42, n_jobs=-1)
        modelname = "RandomForestRegressor"
        mse, mae = sklearn_model_predict(est, df, test_size=50, underlyingname=ticker, modelname=modelname)
        results_df = pd.concat([results_df, pd.DataFrame({'Ticker': [ticker], 'Model': [modelname], 'MSE': [mse], 'MAE': [mae]})], ignore_index=True)

        # # LinearRegression
        # est = LinearRegression(n_jobs=-1)
        # modelname = "LinearRegression"
        # mse, mae = sklearn_model_predict(est, df, test_size=50, underlyingname=ticker, modelname=modelname)
        # results_df = pd.concat([results_df, pd.DataFrame({'Ticker': [ticker], 'Model': [modelname], 'MSE': [mse], 'MAE': [mae]})], ignore_index=True)

        # Save results to disk
        results_filename = "../data/model_performance_results.csv"
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        print(f"store results in {results_filename} after ticker symbol: {ticker}")
        results_df.to_csv(results_filename, index=False)

    print("done")
