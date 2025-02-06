import talib as ta
import pandas as pd # type: ignore

# def calculate_technical_indicators(stock_data):
#     stock_data['MACD'], stock_data['MACD_Signal'], _ = ta.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
#     stock_data['RSI'] = ta.RSI(stock_data['Close'], timeperiod=14)
#     stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = ta.BBANDS(stock_data['Close'], timeperiod=20)
#     stock_data.dropna(inplace=True)  # Drop NaN values from indicators
#     return stock_data

def calculate_technical_indicators(stock_data):
    import talib as ta
    stock_data['MACD'], stock_data['MACD_Signal'], _ = ta.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data['RSI'] = ta.RSI(stock_data['Close'], timeperiod=14)
    stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = ta.BBANDS(stock_data['Close'], timeperiod=20)
    
    # Add Moving Averages
    stock_data['20_MA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
    
    stock_data.dropna(inplace=True)  # Drop NaN values from indicators
    return stock_data


def calculate_risk_percentage(stock_data):
    atr = ta.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14).iloc[-1]
    current_price = stock_data['Close'].iloc[-1]
    return atr / current_price  # Dynamic risk based on volatility
    