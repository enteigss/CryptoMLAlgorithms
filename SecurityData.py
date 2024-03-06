import yfinance as yf

msft = yf.Ticker("BTC-USD")
info = msft.info 
hist = msft.history(period="1y")
hist = yf.download("BTC-USD", start="2024-01-01", end="2024-03-05", interval="1wk")
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
    hist = yf.download("AAPL BAC", period="5d")

Download 
    Allows you to skip yt.ticker
    hist = yf.download("AAPL", start="2024-01-01", end="2024-03-05")
    group_by='tickers' - for each stock it will first list the 4 catgeories and then go to the next for the next stock, 
    not listing the values for each stock for a given category
    data = yf.download("AMZN AAPL GOOG", start="2017-01-01",end="2017-04-30", group_by='tickers')

Options 
    opt_chain() - inputs are date: yyyy-mm-dd, expiration date
    aapl.options - gives all of the exipiration dates for options
    
    opt = msft.option_chain(date='2024-03-08')
    opt.calls
        gives all of the option data for the different calls on that exp 
    opt.puts
        gives all of the option data for the different puts on that exp 

Printing variables into a string (j is the variable)
    a = f"HI {j} u suck"

Putting Stock data into CSV
    ap = yf.download("AAPL", start="2024-01-01", end="2024-03-05", interval="1d")
    path = "/Users/arcwilbo/desktop/aps.csv"
    ap.to_csv(path)

Putting options data into CSV 
    SPY = yf.Ticker("SPY")
    opt = SPY.option_chain(date="2024-03-08").calls
    path = "/Users/arcwilbo/desktop/spycalls.csv"
    opt.to_csv(path)
"""
print(hist)
