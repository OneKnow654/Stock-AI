import talib as ta
import pandas as pd # type: ignore

def calculate_technical_indicators(stock_data):
    stock_data['MACD'], stock_data['MACD_Signal'], _ = ta.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data['RSI'] = ta.RSI(stock_data['Close'], timeperiod=14)
    stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = ta.BBANDS(stock_data['Close'], timeperiod=20)
    stock_data.dropna(inplace=True)  # Drop NaN values from indicators
    return stock_data
