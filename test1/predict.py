import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import tensorflow as tf
import talib as ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def format_term(term):
    """Format term strings for filenames."""
    return term.lower().replace("-", "").replace(" ", "")

def fetch_stock_data(ticker, start, end):
    """Fetch stock data and compute basic technical indicators."""
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data.reset_index(inplace=True)
    stock_data['MACD'], stock_data['MACD_Signal'], _ = ta.MACD(
        stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    stock_data['RSI'] = ta.RSI(stock_data['Close'], timeperiod=14)
    stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = ta.BBANDS(
        stock_data['Close'], timeperiod=20
    )
    stock_data.dropna(inplace=True)
    return stock_data

def load_models(term):
    """Load trained models and scalers for the given term."""
    term_formatted = format_term(term)
    rf = joblib.load(f"random_forest_{term_formatted}.pkl")
    xgb = joblib.load(f"xgboost_{term_formatted}.pkl")
    lstm = tf.keras.models.load_model(f"lstm_{term_formatted}.h5")
    x_scaler = joblib.load(f"lstm_x_scaler_{term_formatted}.pkl")
    y_scaler = joblib.load(f"lstm_y_scaler_{term_formatted}.pkl")
    return rf, xgb, lstm, x_scaler, y_scaler

def create_sequences(X, time_steps):
    """
    Convert a 2D array into sequences for LSTM prediction.
    Uses a loop from 0 to len(X) - time_steps + 1 so that if len(X)==time_steps, one sequence is created.
    """
    Xs = []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:i + time_steps])
    return np.array(Xs)

def make_predictions(models, X_test, term):
    """
    Make predictions using Random Forest, XGBoost, and LSTM.
    For "Mid-Term", LSTM uses a sliding window (sequence length = 90).
      - If X_test has fewer than 90 samples, pad the scaled features (by repeating the first row)
        so that one sequence of length 90 is created, and then use the last sample's prediction
        from RF and XGB for consistency.
    For other terms, each sample is predicted independently.
    """
    rf, xgb, lstm, x_scaler, y_scaler = models
    print("X_test shape:", X_test.shape)
    
    # RF and XGB predict directly on the features
    rf_preds = rf.predict(X_test)
    xgb_preds = xgb.predict(X_test)
    
    if term == "Mid-Term":
        time_steps = 90
        X_test_scaled = x_scaler.transform(X_test)
        if len(X_test) < time_steps:
            # Pad the test data so we have at least one sequence.
            pad_length = time_steps - len(X_test)
            pad_values = np.repeat(X_test_scaled[0:1, :], pad_length, axis=0)
            X_test_padded = np.vstack([pad_values, X_test_scaled])
            # Create one sequence from the padded data.
            X_test_seq = np.expand_dims(X_test_padded, axis=0)  # shape: (1, time_steps, n_features)
            lstm_preds_scaled = lstm.predict(X_test_seq, batch_size=1)
            lstm_preds = y_scaler.inverse_transform(lstm_preds_scaled).flatten()
            # For consistency, use the last sample's prediction for RF and XGB.
            rf_preds = np.array([rf_preds[-1]])
            xgb_preds = np.array([xgb_preds[-1]])
        else:
            # Enough samples exist, so create sequences.
            X_test_seq = create_sequences(x_scaler.transform(X_test), time_steps)
            lstm_preds_scaled = lstm.predict(X_test_seq, batch_size=1)
            lstm_preds = y_scaler.inverse_transform(lstm_preds_scaled).flatten()
    else:
        # For "Short-Term" or "Long-Term", each sample is processed individually.
        X_test_scaled = x_scaler.transform(X_test)
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
        lstm_preds_scaled = lstm.predict(X_test_scaled, batch_size=1)
        lstm_preds = y_scaler.inverse_transform(lstm_preds_scaled).flatten()
    
    return rf_preds, xgb_preds, lstm_preds

def safe_mape(y_true, y_pred, epsilon=1e-8):
    """Compute MAPE while avoiding division by zero."""
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

def evaluate_models(y_test, predictions):
    """Evaluate predictions using MSE, MAE, and MAPE."""
    metrics = {}
    model_names = ["Random Forest", "XGBoost", "LSTM"]
    for name, preds in zip(model_names, predictions):
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mape = safe_mape(np.array(y_test), preds)
        metrics[name] = {"MSE": mse, "MAE": mae, "MAPE": mape}
        print(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    return metrics

def plot_predictions(y_test, predictions):
    """Plot actual vs. predicted stock prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Price", marker='o')
    model_names = ["Random Forest", "XGBoost", "LSTM"]
    for name, preds in zip(model_names, predictions):
        plt.plot(preds, label=f"{name} Prediction", linestyle='--')
    plt.xlabel("Samples")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.title("Actual vs Predicted Stock Prices")
    plt.show()

if __name__ == "__main__":
    # Set the term you want to evaluate: "Short-Term", "Mid-Term", or "Long-Term"
    term = "Short-Term"  # Change as needed.
    ticker = "tatasteel.ns"  # Example Indian stock ticker
    start_date = "2023-09-01" 	
    end_date = "2024-01-01"
    
    # Fetch data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # For Mid-Term predictions, add extra indicators.
    if term == "Mid-Term":
        stock_data['SMA_30'] = ta.SMA(stock_data['Close'], timeperiod=30)
        stock_data['EMA_30'] = ta.EMA(stock_data['Close'], timeperiod=30)
        stock_data.dropna(inplace=True)
        required_columns = ['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'SMA_30', 'EMA_30']
    else:
        required_columns = ['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']
    
    # Prepare features and target.
    X = stock_data[required_columns]
    y = stock_data['Close']
    
    # Create a test split.
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = load_models(term)
    predictions = make_predictions(models, X_test, term)
    
    # Adjust y_test for evaluation:
    if term == "Mid-Term":
        time_steps = 90
        if len(X_test) < time_steps:
            # We padded and produced one prediction: use the last true value.
            y_test_adj = np.array([y_test.values[-1]])
        else:
            # When sequences are created, the first prediction corresponds to index (time_steps - 1)
            y_test_adj = y_test.values[time_steps - 1:]
    else:
        y_test_adj = y_test.values
    
    evaluate_models(y_test_adj, predictions)
    plot_predictions(pd.Series(y_test_adj), predictions)
