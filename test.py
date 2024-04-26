import yfinance as yf


hist = yf.download("BTC-USD", period="max", interval="1d")
print(hist.loc[:, hist.columns.str.contains('_\d+$')].iloc[-1].values.reshape(1, -1))
#print(hist.index[-1])