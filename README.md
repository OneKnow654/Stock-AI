| Note : Before run the code make sure u are run in virtual env 
- how to create goto the bottom of this readme 

---

In stock price prediction, **short-term**, **mid-term**, and **long-term** refer to different time horizons over which predictions are made, and each has a specific range based on typical trading and investing strategies:

### 1. **Short-Term** (Days to 1 Week)
   - **Range**: 1 to 7 days ahead.
   - **Typical Use**: Used by day traders or swing traders who aim to profit from small, quick price changes within a few days.
   - **Prediction Frequency**: Daily predictions, often requiring more sensitive models like LSTMs that can capture daily price fluctuations and short-term market momentum.
   - **Indicators**: Momentum indicators (e.g., RSI, MACD) are usually more relevant for short-term predictions.

### 2. **Mid-Term** (1 Week to 1 Month)
   - **Range**: 1 to 4 weeks ahead.
   - **Typical Use**: Used by swing traders and investors who aim to capture broader trends over several weeks, balancing short-term fluctuations with more stable price movement.
   - **Prediction Frequency**: Weekly predictions; may use models that focus on capturing trends while smoothing out daily volatility.
   - **Indicators**: Trend-following indicators (e.g., Moving Averages) are generally suitable for mid-term analysis, helping predict continuation or reversal of trends.

### 3. **Long-Term** (1 Month to 1 Year or More)
   - **Range**: 1 month to 12 months (or even longer).
   - **Typical Use**: Used by long-term investors or position traders who focus on the fundamental value of the stock and expect larger trends to play out over months or years.
   - **Prediction Frequency**: Monthly predictions; long-term models often incorporate broader economic indicators and ignore minor fluctuations.
   - **Indicators**: Longer-term moving averages (e.g., 50-day, 200-day), sentiment indicators, and fundamental data (earnings, P/E ratio) can be useful.

### Summary Table

| Term       | Range                  | Prediction Interval | Use Case                         | Recommended Indicators     |
|------------|------------------------|---------------------|----------------------------------|----------------------------|
| **Short-Term** | 1 to 7 days           | Daily               | Day trading, short-term trades  | RSI, MACD, short MAs       |
| **Mid-Term**   | 1 week to 1 month     | Weekly             | Swing trading                   | Moving Averages, trend lines |
| **Long-Term**  | 1 month to 1 year+    | Monthly            | Position trading, long-term hold | 50-day/200-day MAs, fundamentals |

Each timeframe captures different price movements and may benefit from different models and indicators. In your project, training specific models for each timeframe and customizing features for each range can improve accuracy and relevance. Let me know if you'd like further details on any specific timeframe!



# Test area
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
        "ticker": "ADSL.NS",
        "start": "2024-07-01",
        "end": "2024-11-13",
        "timeframe": "short-term"
    }'
```


1. Dynamic Risk Percentage
Instead of using a fixed risk percentage, dynamically calculate it based on market conditions, stock volatility, or user preferences.

Implementation:
Volatility-Based Risk: Use indicators like Average True Range (ATR) or standard deviation of recent prices to determine the risk percentage.
Risk Percentage


ATR
Current Price
Risk Percentage= 
Current Price
ATR
​
 
User-Defined Levels: Allow users to choose between low, medium, or high risk, and map these to predefined risk percentages (e.g., 2%, 5%, 10%).





### **1. Using `venv` (Python 3 built-in module):**
#### **Step-by-Step:**
1. **Navigate to your project folder:**
   ```bash
   cd /path/to/your/project
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv env_name
   ```
   Replace `env_name` with the desired name for your virtual environment (e.g., `venv`).

3. **Activate the virtual environment:**
   - **On macOS/Linux:**
     ```bash
     source env_name/bin/activate
     ```
   - **On Windows (Command Prompt):**
     ```cmd
     env_name\Scripts\activate
     ```
   - **On Windows (PowerShell):**
     ```powershell
     .\env_name\Scripts\Activate.ps1
     ```

4. **Install dependencies within the environment:**
   ```bash
   pip install <package_name>
   ```

5. **Deactivate the virtual environment (when done):**
   ```bash
   deactivate
   ```
