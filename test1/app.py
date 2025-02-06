from flask import Flask, request, jsonify # type:ignore
import os
import requests # type:ignore
import numpy as np # type:ignore
import pandas as pd # type:ignore
import joblib # type:ignore
import yfinance as yf # type:ignore
import tensorflow as tf # type:ignore
import talib as ta # type:ignore
from sklearn.model_selection import train_test_split # type:ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type:ignore
from flask_cors import CORS  # type:ignore

# --------------- Flask App Setup ---------------
app = Flask(__name__)
CORS(app)

# --------------- Helper Functions ---------------
def format_term(term: str) -> str:
    return term.lower().replace("-", "").replace(" ", "")

def format_ticker(ticker: str) -> str:
    return ticker.lower().replace(" ", "")

def get_model_filenames(ticker: str, term: str):
    """Return standardized filenames for saving/loading models/scalers."""
    ticker_formatted = format_ticker(ticker)
    term_formatted = format_term(term)
    return {
        "rf": f"random_forest_{ticker_formatted}_{term_formatted}.pkl",
        "xgb": f"xgboost_{ticker_formatted}_{term_formatted}.pkl",
        "lstm": f"lstm_{ticker_formatted}_{term_formatted}.h5",
        "lstm_x_scaler": f"lstm_x_scaler_{ticker_formatted}_{term_formatted}.pkl",
        "lstm_y_scaler": f"lstm_y_scaler_{ticker_formatted}_{term_formatted}.pkl"
    }

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo, compute MACD, RSI, Bollinger Bands."""
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    if data.empty:
        raise ValueError("No data fetched. Check ticker or dates.")
    data['MACD'], data['MACD_Signal'], _ = ta.MACD(
        data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
    data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = ta.BBANDS(data['Close'], timeperiod=20)
    data.dropna(inplace=True)
    return data

def add_midterm_features(stock_data: pd.DataFrame, term: str) -> pd.DataFrame:
    """For 'Mid-Term', add 30-day SMA, EMA. For 'Long-Term' you could add more if desired."""
    if term == "Mid-Term":
        stock_data['SMA_30'] = ta.SMA(stock_data['Close'], timeperiod=30)
        stock_data['EMA_30'] = ta.EMA(stock_data['Close'], timeperiod=30)
        stock_data.dropna(inplace=True)
    # If you want other features for Long-Term, add them here.
    return stock_data

def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int):
    """Create 3D sequences for LSTM. Each sequence has length 'time_steps'."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i : i+time_steps])
        ys.append(y[i+time_steps - 1])
    return np.array(Xs), np.array(ys)

def load_models_for_prediction(ticker: str, term: str):
    """Load previously saved models."""
    filenames = get_model_filenames(ticker, term)
    rf = joblib.load(filenames["rf"])
    xgb = joblib.load(filenames["xgb"])
    lstm = tf.keras.models.load_model(filenames["lstm"])
    x_scaler = joblib.load(filenames["lstm_x_scaler"])
    y_scaler = joblib.load(filenames["lstm_y_scaler"])
    return rf, xgb, lstm, x_scaler, y_scaler

# --------------- Training on Full Data ---------------
def train_models_on_full_data(ticker: str, start_date: str, end_date: str, term: str):
    """
    Trains RandomForest, XGBoost, LSTM on *all* data (no train-test split),
    then saves them for future use.
    """
    from sklearn.ensemble import RandomForestRegressor # type: ignore
    from xgboost import XGBRegressor  # type: ignore
    from tensorflow.keras.models import Sequential # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
    from sklearn.preprocessing import MinMaxScaler # type: ignore

    # 1. Fetch data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = add_midterm_features(stock_data, term)

    # 2. Choose feature columns
    if term == "Mid-Term":
        feature_cols = ['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'SMA_30', 'EMA_30']
    else:
        # Short-Term or Long-Term
        feature_cols = ['MACD', 'RSI', 'Upper_Band', 'Middle_Band', 'Lower_Band']

    X = stock_data[feature_cols]
    y = stock_data['Close']

    # 3. Train RandomForest
    if term == "Mid-Term":
        rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    elif term == "Short-Term":
        rf = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 4. Train XGBoost
    if term == "Mid-Term":
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=10)
    elif term == "Short-Term":
        xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=10)
    else:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05)
    xgb.fit(X, y)

    # 5. Train LSTM
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_reshaped = y.values.reshape(-1, 1)
    y_scaled = y_scaler.fit_transform(y_reshaped)

    if term == "Mid-Term":
        time_steps = 90
    elif term == "Short-Term":
        time_steps = 1
    else:
        # Example: 180 days for "Long-Term"
        time_steps = 180

    if time_steps > len(X_scaled):
        raise ValueError(f"Not enough data ({len(X_scaled)}) for LSTM window of {time_steps} days.")

    if time_steps > 1:
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
        input_shape = (time_steps, X.shape[1])
    else:
        # time_steps = 1
        X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        y_seq = y_scaled
        input_shape = (1, X.shape[1])

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0)

    # 6. Save all
    filenames = get_model_filenames(ticker, term)
    joblib.dump(rf, filenames["rf"])
    joblib.dump(xgb, filenames["xgb"])
    model.save(filenames["lstm"])
    joblib.dump(x_scaler, filenames["lstm_x_scaler"])
    joblib.dump(y_scaler, filenames["lstm_y_scaler"])
    print(f"Models (RF, XGB, LSTM) trained on full data & saved for {ticker}, {term}")

# --------------- Forecast Endpoint ---------------
@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    """
    True multi-day future forecasting beyond end_date.
    - Parameters: ticker, start_date, end_date, term in {Short-Term, Mid-Term, Long-Term}, forecast_days
    - If forecast_days is not provided, defaults based on the term (Short=7, Mid=30, Long=90).
    - Recalculates technical indicators each day so features differ daily.
    - Uses forward-fill instead of dropna to avoid losing the new row.
    """
    try:
        # 1) Parse request
        if request.method == "GET":
            ticker = request.args.get("ticker")
            start_date = request.args.get("start_date")
            end_date = request.args.get("end_date")
            term = request.args.get("term", "Short-Term")
            forecast_days = request.args.get("forecast_days", None)
        else:
            data = request.get_json()
            ticker = data.get("ticker")
            start_date = data.get("start_date")
            end_date = data.get("end_date")
            term = data.get("term", "Short-Term")
            forecast_days = data.get("forecast_days", None)

        if not ticker or not start_date or not end_date:
            return jsonify({"error": "Missing required parameters: ticker, start_date, end_date"}), 400

        # 2) Default forecast days if not specified
        if forecast_days is None:
            if term == "Short-Term":
                forecast_days = 7
            elif term == "Mid-Term":
                forecast_days = 30
            else:  # Long-Term
                forecast_days = 90
        else:
            forecast_days = int(forecast_days)

        # 3) Train on full data if models don't exist
        filenames = get_model_filenames(ticker, term)
        if not all(os.path.exists(f) for f in filenames.values()):
            train_models_on_full_data(ticker, start_date, end_date, term)

        # 4) Load the entire dataset up to end_date
        df = fetch_stock_data(ticker, start_date, end_date)
        if term == "Mid-Term":
            df = add_midterm_features(df, term)
        elif term == "Long-Term":
            # If you have special features for Long-Term, add them here
            pass

        # 5) Load models
        rf, xgb, lstm, x_scaler, y_scaler = load_models_for_prediction(ticker, term)

        # Decide LSTM lookback
        if term == "Short-Term":
            time_steps = 1
        elif term == "Mid-Term":
            time_steps = 90
        else:  # Long-Term
            time_steps = 180

        # 6) Ensure enough data for LSTM
        if len(df) < time_steps:
            return jsonify({
                "error": f"Not enough data ({len(df)}) for {term} LSTM window ({time_steps})."
            }), 400

        # 7) Identify feature columns
        if term == "Mid-Term":
            feature_cols = ["MACD", "RSI", "Upper_Band", "Middle_Band", "Lower_Band", "SMA_30", "EMA_30"]
        else:
            feature_cols = ["MACD", "RSI", "Upper_Band", "Middle_Band", "Lower_Band"]

        # Sort by date to ensure proper sequence
        df.sort_values(by="Date", inplace=True)

        # We'll create a working DataFrame for day-by-day forecasting
        df_for_inference = df.copy()

        forecast_dates = []
        forecast_rf = []
        forecast_xgb = []
        forecast_lstm = []

        current_date = pd.to_datetime(end_date)

        for _ in range(forecast_days):
            next_date = current_date + pd.Timedelta(days=1)
            # (Optionally skip weekends if desired)

            # -------------------------------
            # 1) Prepare the last 'time_steps' rows for LSTM
            # -------------------------------
            last_block = df_for_inference.iloc[-time_steps:].copy()
            X_block_scaled = x_scaler.transform(last_block[feature_cols])
            X_block_scaled = np.expand_dims(X_block_scaled, axis=0)  # shape => (1, time_steps, num_features)
            lstm_pred_scaled = lstm.predict(X_block_scaled)
            lstm_pred = y_scaler.inverse_transform(lstm_pred_scaled)[0, 0]

            # -------------------------------
            # 2) Prepare single row for RF/XGB
            # -------------------------------
            last_row_features = df_for_inference.iloc[-1][feature_cols].values.reshape(1, -1)
            rf_pred = rf.predict(last_row_features)[0]
            xgb_pred = xgb.predict(last_row_features)[0]

            # -------------------------------
            # 3) Store predictions
            # -------------------------------
            forecast_dates.append(str(next_date.date()))
            forecast_rf.append(float(rf_pred))
            forecast_xgb.append(float(xgb_pred))
            forecast_lstm.append(float(lstm_pred))

            # -------------------------------
            # 4) Append a new row with the predicted close
            # -------------------------------
            new_row = {
                "Date": next_date,
                "Open": lstm_pred,
                "High": lstm_pred,
                "Low": lstm_pred,
                "Close": lstm_pred,
            }
            df_for_inference = pd.concat([df_for_inference, pd.DataFrame([new_row])], ignore_index=True)

            # -------------------------------
            # 5) Recompute indicators on the last ~200 rows
            #    (big enough to handle MACD/RSI/Bollinger periods)
            # -------------------------------
            recent_chunk = df_for_inference.iloc[-200:].copy()

            # Recalculate TA-Lib indicators
            recent_chunk["MACD"], recent_chunk["MACD_Signal"], _ = ta.MACD(
                recent_chunk["Close"], fastperiod=12, slowperiod=26, signalperiod=9
            )
            recent_chunk["RSI"] = ta.RSI(recent_chunk["Close"], timeperiod=14)
            recent_chunk["Upper_Band"], recent_chunk["Middle_Band"], recent_chunk["Lower_Band"] = ta.BBANDS(
                recent_chunk["Close"], timeperiod=20
            )

            if term == "Mid-Term":
                recent_chunk["SMA_30"] = ta.SMA(recent_chunk["Close"], timeperiod=30)
                recent_chunk["EMA_30"] = ta.EMA(recent_chunk["Close"], timeperiod=30)

            # Instead of recent_chunk.dropna(), we forward-fill so we don't lose new rows
            recent_chunk.fillna(method="ffill", inplace=True)

            # Update df_for_inference with recalculated indicators
            df_for_inference.update(recent_chunk)

            # Advance the date
            current_date = next_date

        # -------------------------------
        # 6) Build response
        # -------------------------------
        return jsonify({
            "ticker": ticker,
            "term": term,
            "forecast_days": forecast_days,
            "forecast_dates": forecast_dates,
            "predictions": {
                "RandomForest": forecast_rf,
                "XGBoost": forecast_xgb,
                "LSTM": forecast_lstm
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------- Run the Flask App ---------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
