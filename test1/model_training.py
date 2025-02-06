import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import talib as ta
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def format_term(term):
    """Helper to format term strings for filenames."""
    return term.lower().replace("-", "").replace(" ", "")

def fetch_stock_data(ticker, start, end):
    """Fetch stock data from Yahoo Finance and compute technical indicators."""
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data.reset_index(inplace=True)
    
    if stock_data.empty:
        raise ValueError("Stock data is empty! Adjust date range or check the ticker symbol.")
    
    # Compute technical indicators
    stock_data['MACD'], stock_data['MACD_Signal'], _ = ta.MACD(
        stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    stock_data['RSI'] = ta.RSI(stock_data['Close'], timeperiod=14)
    stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = ta.BBANDS(
        stock_data['Close'], timeperiod=20
    )
    
    stock_data.dropna(inplace=True)
    return stock_data

def train_random_forest(X_train, y_train, term):
    """Train Random Forest model."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    term_formatted = format_term(term)
    joblib.dump(rf, f"random_forest_{term_formatted}.pkl")
    return rf

def train_xgboost(X_train, y_train, term):
    """Train XGBoost model."""
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    term_formatted = format_term(term)
    joblib.dump(xgb, f"xgboost_{term_formatted}.pkl")
    return xgb

def train_lstm(X_train, y_train, term):
    """Train LSTM model using separate scalers for features and target."""
    # Create separate scalers for X and y
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # Scale features and reshape for LSTM [samples, timesteps, features]
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    
    # Scale target values (reshape y to be 2D)
    y_train = y_train.values.reshape(-1, 1)
    y_train_scaled = y_scaler.fit_transform(y_train)
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[1])),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=1)
    
    term_formatted = format_term(term)
    model.save(f"lstm_{term_formatted}.h5")
    joblib.dump(x_scaler, f"lstm_x_scaler_{term_formatted}.pkl")
    joblib.dump(y_scaler, f"lstm_y_scaler_{term_formatted}.pkl")
    return model

def prepare_data(stock_data):
    """Prepare data for model training."""
    required_columns = ['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']
    missing_columns = [col for col in required_columns if col not in stock_data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in stock data: {missing_columns}")
    
    X = stock_data[required_columns]
    y = stock_data['Close']
    
    if len(X) < 10:
        raise ValueError(f"Not enough data for training! Available samples: {len(X)}")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models_for_term(ticker, start_date, end_date, term):
    """Fetch data and train models for a given term (short, mid, long)."""
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    if stock_data.empty:
        raise ValueError(f"No data available for {term} term!")
    
    X_train, _, y_train, _ = prepare_data(stock_data)
    
    print(f"Training models for {term} term...")
    train_random_forest(X_train, y_train, term)
    train_xgboost(X_train, y_train, term)
    train_lstm(X_train, y_train, term)
    
    print(f"Models trained and saved for {term} term!")

if __name__ == "__main__":
    ticker = "tatasteel.NS"
    
    # Train models for three different terms (files are saved with term-specific names)
    #train_models_for_term(ticker, "2023-09-01", "2024-01-01", "Short-Term")  # e.g., 15 weeks
    #train_models_for_term(ticker, "2023-04-01", "2024-01-01", "Mid-Term")     # e.g., 3 months
    train_models_for_term(ticker, "2020-01-01", "2025-01-01", "Long-Term")    # e.g., 5 years
    
    print("All models trained and saved using Yahoo Finance data!")
