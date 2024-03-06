import yfinance as yf

msft = yf.Ticker("MSFT")
info = msft.info 
hist = msft.history(period="1y")
""" 
Data Period to download - gives the daily values for the time frame, period = 1y -> gives the daily values for the last year
“   1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”

If not using period, Start and End
start = yyyy-mm-dd
end = yyyy-mm-dd

prepost: Include Pre and Post regular market data in results? (Default is False)
    
Interval 
    data interval (1m data is only for available for last 7 days, and data interval <1d for the last 60 days) Valid intervals are:
    “1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”, “1d”, “5d”, “1wk”, “1mo”, “3mo”

Example
    aapl_historical = aapl.history(start="2020-06-02", end="2020-06-07", interval="1m")

To see data from multiple stocks next to each other use download
    data = yf.download("AMZN AAPL GOOG", start="2017-01-01", end="2017-04-30")


"""
#s = msft.history(start="2024-01-01", end="2024-03-05",interval="1mo")
print(hist)

"""data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
print(data.head())
print(5)
"""
