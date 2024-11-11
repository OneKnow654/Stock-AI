import yfinance as yf

def fetch_historical_data(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        if stock_data.empty:
            return None
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = stock_data['Date'].astype(str)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
