import requests
from bs4 import BeautifulSoup

def get_stock_data(symbol):
	url = f"https://finance.yahoo.com/quote/{symbol}"
	response = requests.get(url)
	soup = BeautifulSoup(response.text, 'html.parser')
	
	data = {}
	data['price'] = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'}).text
	# data['open'] = soup.find('td', {'data-test': 'OPEN-value'}).text
	# data['close'] = soup.find('td', {'data-test': 'PREV_CLOSE-value'}).text
	# data['high'] = soup.find('td', {'data-test': 'DAYS_RANGE-value'}).text.split(' - ')[1]
	# data['low'] = soup.find('td', {'data-test': 'DAYS_RANGE-value'}).text.split(' - ')[0]
	
	return data

if __name__ == "__main__":
	symbol = "AAPL"
	stock_data = get_stock_data(symbol)
	print(f"The current stock data of {symbol} is:")
	print(f"Price: ${stock_data['price']}")
	# print(f"Open: ${stock_data['open']}")
	# print(f"Close: ${stock_data['close']}")
	# print(f"High: ${stock_data['high']}")
	# print(f"Low: ${stock_data['low']}")

import yfinance as yf
start = '2024-4-30'
end = '2024-10-31'
tickers = ['AAPL'] 

for ticker in tickers:
	data = yf.download(ticker, start=start, end=end)
	print(data)

apple = yf.download("AAPL", period="5d")
print(apple)


# import pandas as pd  
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.optimize as sco
# import datetime as dt
# import math
# from datetime import datetime, timedelta
# from pandas_datareader import data as wb
# from sklearn.cluster import KMeans
# np.random.seed(777)


# start = '2019-4-30'
# end = '2019-10-31'
# # N = 90
# # start = datetime.now() - timedelta(days=N)
# # end = dt.datetime.today()



# tickers = ['MMM',
# 'ABT',
# 'ABBV',
# 'ABMD',
# 'AAPL',
# 'XEL',
# 'XRX',
# 'XLNX',
# 'XYL',
# 'YUM',
# 'ZBH',
# 'ZION',
# 'ZTS'] 

# thelen = len(tickers)

# price_data = []
# for ticker in tickers:
#     prices = wb.DataReader(ticker, start = start, end = end, data_source='yahoo')[['Adj Close']]
#     price_data.append(prices.assign(ticker=ticker)[['ticker', 'Adj Close']])

# df = pd.concat(price_data)
# df.dtypes
# df.head()
# df.shape

# pd.set_option('display.max_columns', 500)

# df = df.reset_index()
# df = df.set_index('Date')
# table = df.pivot(columns='ticker')
# # By specifying col[1] in below list comprehension
# # You can select the stock names under multi-level column
# table.columns = [col[1] for col in table.columns]
# table.head()

# plt.figure(figsize=(14, 7))
# for c in table.columns.values:
#     plt.plot(table.index, table[c], lw=3, alpha=0.8,label=c)
# plt.legend(loc='upper left', fontsize=12)
# plt.ylabel('price in $')
# plt.show()