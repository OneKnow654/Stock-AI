from flask import Flask, request, jsonify # type: ignore
from data_fetcher import fetch_historical_data
from indicators import calculate_technical_indicators
#from model_training import train_and_save_model, load_model, predict_closing_price_with_accuracy
from model_training import train_and_save_model, load_model_and_scaler, predict_closing_price_with_accuracy
from sentiment_analysis import *
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Stock Price Prediction API!"

# Endpoint to fetch historical data
@app.route('/data', methods=['GET'])
def get_data():
    ticker = request.args.get('ticker')
    start = request.args.get('start')
    end = request.args.get('end')
    data = fetch_historical_data(ticker, start, end)
    if data is not None:
        return jsonify(data.to_dict(orient="records"))
    else:
        return jsonify({"error": "Data not found"}), 404

# Endpoint to calculate technical indicators
@app.route('/indicators', methods=['POST'])
def indicators():

    data = request.get_json()
    stock_data = pd.DataFrame(data)
    indicators_data = calculate_technical_indicators(stock_data)
    return jsonify(indicators_data.to_dict(orient="records"))

# Endpoint to predict stock price
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     ticker = data['ticker']
#     timeframe = data['timeframe']  # Accept values like 'short-term', 'mid-term', 'long-term'
#     start, end = data['start'], data['end']

#     # Fetch historical data and calculate indicators
#     stock_data = fetch_historical_data(ticker, start, end)
#     if stock_data is None:
#         return jsonify({"error": "Unable to fetch data"}), 404

#     stock_data_with_indicators = calculate_technical_indicators(stock_data)

#     # Get sentiment score for the stock
#     # sentiment_score = get_stock_sentiment(ticker)
#     # stock_data_with_indicators['Sentiment'] = sentiment_score

#     # Train model if it doesn't exist
#     model = load_model(timeframe)  
#     if model is None:
#         print(f"Training {timeframe} model...")
#         train_and_save_model(stock_data_with_indicators, timeframe)
#         model = load_model(timeframe)

#     # Predict closing price and calculate accuracy
#     predicted_price, mse, mae, mape = predict_closing_price_with_accuracy(model, stock_data_with_indicators)

#     return jsonify({
#         "predicted_closing_price": round(predicted_price,2),
#         "sentiment_score": "None",
#         "mean_squared_error": round(mse,2),
#         "mean_absolute_error": round(mae,2),
#         "mean_absolute_percentage_error": round(mape,2)  # MAPE as accuracy percentage
#     })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data['ticker']
    timeframe = data['timeframe']  # Accept values like 'short-term', 'mid-term', 'long-term'
    start, end = data['start'], data['end']

    # Fetch historical data and calculate indicators
    stock_data = fetch_historical_data(ticker, start, end)
    if stock_data is None:
        return jsonify({"error": "Unable to fetch data"}), 404

    stock_data_with_indicators = calculate_technical_indicators(stock_data)

    # Get sentiment score for the stock
    # sentiment_score = get_stock_sentiment(ticker)
    # stock_data_with_indicators['Sentiment'] = sentiment_score

    # Load or train the LSTM model and scaler
    model, scaler = load_model_and_scaler(ticker,timeframe,stock_data_with_indicators)
    if model is None:
        print(f"Training {timeframe} LSTM model...")
        train_and_save_model(stock_data_with_indicators, timeframe)
        model, scaler = load_model_and_scaler(timeframe)

    # Predict closing price and calculate accuracy
    predicted_price, mse, mae, mape = predict_closing_price_with_accuracy(model, stock_data_with_indicators, scaler)

    return jsonify({
        "predicted_closing_price": round(predicted_price,2),
        "sentiment_score":None,
        "mean_squared_error": round(mse,2),
        "mean_absolute_error": round(mae,2),
        "mean_absolute_percentage_error": round(mape,3)
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
