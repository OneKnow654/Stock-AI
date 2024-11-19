import numpy as np
import pandas as pd
import os   
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def prepare_lstm_data(stock_data, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close', 'MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']])

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, :])  # Collect 'time_steps' data points before each target
        y.append(scaled_data[i, 0])  # Closing price is the target

    X, y = np.array(X), np.array(y)
    return X, y, scaler

#time step loader 
def time_step_load(timeframe):
    if timeframe == 'short-term':
        return 60
    elif timeframe == 'mid-term':
        return 90 # or 60
    elif timeframe == 'long-term':
        return 120  # or 180
    else :
        return 60  #default value

def train_and_save_model(stock_data, ticker, timeframe):
    X, y, scaler = prepare_lstm_data(stock_data)
    
    # Define and train the LSTM model (same as before)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)

    # Save the model and scaler with stock-specific names
    model.save(f'models/{ticker}_{timeframe}_lstm_model.h5')
    joblib.dump(scaler, f'models/{ticker}_{timeframe}_scaler.pkl')
    print(f"Model and scaler saved for {ticker} with {timeframe} timeframe.")


def predict_closing_price_with_accuracy(model, stock_data, scaler, time_steps=60, days=5,risk_percentage=0.05):
    # Prepare data for prediction: Scale and reshape
    recent_data = stock_data[['Close', 'MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']].values
    scaled_recent_data = scaler.transform(recent_data)
    X_input = scaled_recent_data[-time_steps:]
    X_input = np.reshape(X_input, (1, X_input.shape[0], X_input.shape[1]))

    # Predict closing price and inverse transform
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform([[predicted_scaled[0][0], 0, 0, 0, 0, 0]])[0][0]

     # Calculate threshold price
    threshold_price = predicted_price * (1 - risk_percentage)

    # Calculate accuracy metrics on recent data
    X_recent, y_recent, _ = prepare_lstm_data(stock_data.tail(days + time_steps), time_steps)
    y_pred_recent = model.predict(X_recent)
    y_pred_recent = scaler.inverse_transform(np.concatenate([y_pred_recent, np.zeros((y_pred_recent.shape[0], 5))], axis=1))[:, 0]
    y_recent = scaler.inverse_transform(np.concatenate([y_recent.reshape(-1, 1), np.zeros((y_recent.shape[0], 5))], axis=1))[:, 0]

    mse = mean_squared_error(y_recent, y_pred_recent)
    mae = mean_absolute_error(y_recent, y_pred_recent)
    mape = np.mean(np.abs((y_recent - y_pred_recent) / y_recent)) * 100

    return predicted_price, threshold_price,mse, mae, mape


def load_model_and_scaler(ticker, timeframe, stock_data=None):
    model_path = f'models/{ticker}_{timeframe}_lstm_model.h5'
    scaler_path = f'models/{ticker}_{timeframe}_scaler.pkl'

    # Check if both model and scaler exist; if not, train a new model
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        if stock_data is None:
            raise ValueError("Stock data is required to train the model")
        print(f"Training model for {ticker} as no saved model was found...")
        train_and_save_model(stock_data, ticker, timeframe)

    # Load the trained model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler





















# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import joblib  # For saving/loading models

# def train_and_save_model(stock_data, timeframe):
#     # Prepare features and target (closing price)
#     X = stock_data[['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']]
#     y = stock_data['Close']

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train model
#     model = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', random_state=42)
#     model.fit(X_train, y_train)

#     # Evaluate model
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     print(f"{timeframe} Model Mean Squared Error: {mse}")

#     # Save model
#     filename = f"models/{timeframe}_model.pkl"
#     joblib.dump(model, filename)
#     print(f"{timeframe} model saved as {filename}")

# def load_model(timeframe):
#     filename = f"models/{timeframe}_model.pkl"
#     try:
#         model = joblib.load(filename)
#         return model
#     except FileNotFoundError:
#         print(f"Model file {filename} not found.")
#         return None
    
# # def predict_closing_price(model, stock_data):
# #     # Prepare input for prediction (last row of data with indicators)
# #     last_row = stock_data.iloc[-1][['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']].values.reshape(1, -1)
# #     predicted_price = model.predict(last_row)
# #     return predicted_price[0]

# def predict_closing_price_with_accuracy(model, stock_data, days=5):
#     # Prepare input for prediction (last row of data with indicators)
#     last_row = stock_data.iloc[-1][['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']].values.reshape(1, -1)
#     predicted_price = model.predict(last_row)[0]
    
#     # Calculate accuracy on the last 'days' days
#     recent_data = stock_data.tail(days)
#     X_recent = recent_data[['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']]
#     y_recent = recent_data['Close']
    
#     y_pred_recent = model.predict(X_recent)
#     mse = mean_squared_error(y_recent, y_pred_recent)
#     mae = mean_absolute_error(y_recent, y_pred_recent)
    
#     # Calculate Mean Absolute Percentage Error (MAPE)
#     mape = np.mean(np.abs((y_recent - y_pred_recent) / y_recent)) * 100

#     return predicted_price, mse, mae, mape




# ------------------------------------------------------



