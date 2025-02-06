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
    """Fetch stock data from Yahoo Finance and compute basic technical indicators."""
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data.reset_index(inplace=True)
    if stock_data.empty:
        raise ValueError("Stock data is empty! Adjust date range or check the ticker symbol.")
    
    # Compute common technical indicators
    stock_data['MACD'], stock_data['MACD_Signal'], _ = ta.MACD(
        stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    stock_data['RSI'] = ta.RSI(stock_data['Close'], timeperiod=14)
    stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = ta.BBANDS(
        stock_data['Close'], timeperiod=20
    )
    
    return stock_data

def add_midterm_features(stock_data, term):
    """For mid-term, add additional moving average features."""
    if term == "Mid-Term":
        stock_data['SMA_30'] = ta.SMA(stock_data['Close'], timeperiod=30)
        stock_data['EMA_30'] = ta.EMA(stock_data['Close'], timeperiod=30)
    stock_data.dropna(inplace=True)
    return stock_data

def prepare_data(stock_data):
    """Prepare features and target for model training.
       If extra features exist, they are automatically included.
    """
    required_columns = ['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']
    if 'SMA_30' in stock_data.columns:
        required_columns.append('SMA_30')
    if 'EMA_30' in stock_data.columns:
        required_columns.append('EMA_30')
    X = stock_data[required_columns]
    y = stock_data['Close']
    if len(X) < 10:
        raise ValueError("Not enough data for training!")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train, term):
    """Train a Random Forest model with tuned hyperparameters for mid-term."""
    if term == "Mid-Term":
        rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    elif term == "Short-Term":
        rf = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
    else:

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    term_formatted = format_term(term)
    joblib.dump(rf, f"random_forest_{term_formatted}.pkl")
    return rf

def train_xgboost(X_train, y_train, term):
    """Train an XGBoost model with tuned hyperparameters for mid-term."""
    if term == "Mid-Term":
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=10)
    elif term == "Short-Term":
        xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=10)
    else:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    term_formatted = format_term(term)
    joblib.dump(xgb, f"xgboost_{term_formatted}.pkl")
    return xgb

def create_sequences(X, y, time_steps):
    """Convert 2D arrays into 3D sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

def train_lstm(X_train, y_train, term):
    """Train an LSTM model.
       For mid-term predictions, a sliding window (e.g. 90 days) is used.
    """
    # Create separate scalers for features and target
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # Use a longer sequence for mid-term; otherwise, use 1 timestep.
    if term == "Mid-Term": 
        time_steps = 90 
    else:
        time_steps = 1
    
    
    # Scale the training data
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train = y_train.values.reshape(-1, 1)
    y_train_scaled = y_scaler.fit_transform(y_train)
    
    # Create sequences if time_steps > 1
    if time_steps > 1:
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
    else:
        X_train_seq = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        y_train_seq = y_train_scaled
        
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, X_train.shape[1])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=1)
    
    term_formatted = format_term(term)
    model.save(f"lstm_{term_formatted}.h5")
    joblib.dump(x_scaler, f"lstm_x_scaler_{term_formatted}.pkl")
    joblib.dump(y_scaler, f"lstm_y_scaler_{term_formatted}.pkl")
    return model

def train_models_for_term(ticker, start_date, end_date, term):
    """Fetch data, add any term-specific features, and train all models for the given term."""
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = add_midterm_features(stock_data, term)
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
    
    # Train models for different terms.
    train_models_for_term(ticker, "2023-09-01", "2024-01-01", "Short-Term")  # (Leave unchanged)
    train_models_for_term(ticker, "2023-04-01", "2024-01-01", "Mid-Term")     # New tuning for mid-term
    train_models_for_term(ticker, "2019-01-01", "2024-01-01", "Long-Term")    # (Already working)
    
    print("All models trained and saved using Yahoo Finance data!")
