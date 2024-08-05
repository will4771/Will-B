import yfinance as yf
import matplotlib.pyplot as plt



data = yf.download('AAPL', start='2024-01-01', end='2024-08-01')
#print(data.head())

ticker = yf.Ticker('AAPL')
#print(ticker.info)

print(ticker.history(period='5d'))  # last 5 days
#print(ticker.dividends)
#print(ticker.splits)

import matplotlib.pyplot as plt

data['Close'].plot(title='Apple Stock Price')
plt.show()

data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

data[['Close', 'MA50', 'MA200']].plot(title='Apple Stock Price with 50 & 200 Day Moving Averages')
plt.show()
