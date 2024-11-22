from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit,join_room, leave_room
import yfinance as yf
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variable to manage WebSocket updates
connected_clients = {}

# Fetch live stock data
def fetch_live_data(ticker, interval=5):
    while ticker in connected_clients:
        try:
            # Fetch real-time stock data from Yahoo Finance
            stock_data = yf.Ticker(ticker).history(period='1d', interval='1m')
            latest_data = stock_data.tail(1).to_dict(orient="records")[0]

            # Broadcast live data to all connected clients for the ticker
            socketio.emit('live_data', {'ticker': ticker, 'data': latest_data}, room=ticker)

            # Wait for the specified interval before fetching again
            time.sleep(interval)
        except Exception as e:
            print(f"Error fetching live data for {ticker}: {e}")
            break

# WebSocket event: Client requests live stock data
@socketio.on('subscribe')
def handle_subscribe(data):
    ticker = data.get('ticker')
    interval = data.get('interval', 1)

    if ticker:
        if ticker not in connected_clients:
            connected_clients[ticker] = []
            threading.Thread(target=fetch_live_data, args=(ticker, interval), daemon=True).start()

        connected_clients[ticker].append(request.sid)
        join_room(ticker)
        emit('subscribed', {'message': f'Subscribed to live data for {ticker}'}, room=request.sid)

# WebSocket event: Client unsubscribes from live data
@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    ticker = data.get('ticker')

    if ticker and ticker in connected_clients:
        leave_room(ticker)
        connected_clients[ticker].remove(request.sid)

        if not connected_clients[ticker]:
            del connected_clients[ticker]

        emit('unsubscribed', {'message': f'Unsubscribed from live data for {ticker}'}, room=request.sid)

# Home route
@app.route('/')
def index():
    return "WebSocket Server for Live Stock Data"

if __name__ == "__main__":
    socketio.run(app, port=5000, debug=True)
