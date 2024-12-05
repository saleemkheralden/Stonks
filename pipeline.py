from dotenv import load_dotenv, find_dotenv
import os
from models import train_model, LSTM
from utils import action, reformat_data
import requests
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import google.generativeai as genai
import json

class Agent:
	def __init__(self, stock='AAPL', interval='5min', API_KEY='demo', **kwargs):
		self.regressor = Regressor(stock, interval, API_KEY, **kwargs)
		self.gemini = Gemini(stock, interval, API_KEY, **kwargs)

	
	def __call__(self):
		gemini_output = self.gemini.query()
		reg_output = self.regressor.pred()
		return gemini_output, reg_output





class Gemini:
	def __init__(self, stock='AAPL', interval='5min', API_KEY=None, **kwargs):
		_ = load_dotenv(find_dotenv())
		genai.configure(api_key=os.environ['GEMINI_API_KEY'])
		self.model = genai.GenerativeModel('gemini-pro')
		self.url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock}&apikey={API_KEY}'

	
	def query(self):
		r = requests.get(self.url)
		stock_dict = r.json()
		q = f"""Your task is to run a sentiment analysis on the dictionary that's delimited by triple backticks (```) \
this dictionary contains information about a stock in the stock market, you need to process the dictionary \
You need to return only a dictionary that contains whether to Buy or Sell and a score between 0 and 1 of how confident you are with your answer.
the keys of the dictionary should be, action, conf.

Don't add anything to the dictionary that's not mentioned above.
Don't wrap the dictionary in any delimiters.

dictionary:
```{stock_dict}```
"""
		ret = self.model.generate_content(q).text
		ret = ret.replace("'", "\"")
		return json.loads(ret)


class Regressor:
	def __init__(self, stock='AAPL', interval='5min', API_KEY=None, **kwargs):
		self.stock = stock

		self.api_key = API_KEY

		self.outputsize = kwargs.get('outputsize', 'compact')
		self.symbol = kwargs.get('symbol', 'AAPL')
		self.function = kwargs.get('function', 'TIME_SERIES_DAILY')
		self.par = 'Daily' if self.function.__contains__('DAILY') else '5min'
		self.WINDOW = kwargs.get('WINDOW', 60)
		self.TRAIN_LOOKBACK = kwargs.get('TRAIN_LOOKBACK', 5) * 365  # TODO: CHANGE 365 IF NOT DAILY
		
		self.url = lambda function, symbol, outputsize: f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={self.api_key}'
		self.model = LSTM(output_size=5, num_layers=2)
		self.ds = kwargs.get('ds')
		self.ds_flag = self.ds is not None

	def get_train_set(self, slide=0):
		data_mat = self.api_call(self.function, self.symbol, 'full')
		data_mat = data_mat[-self.TRAIN_LOOKBACK:]

		# sc = StandardScaler()
		sc = MinMaxScaler(feature_range=(0, 1))
		data_mat_scaled = sc.fit_transform(data_mat)
		return reformat_data(data_mat_scaled, self.WINDOW)


	def train(self, **kwargs):
		X_train, y_train = self.get_train_set()
		train_model(self.model, X_train, y_train, **kwargs)

	def pred(self, **kwargs):
		data_mat_api = self.api_call(kwargs)
		X, _ = reformat_data(data_mat_api, self.WINDOW)
		return self.model(X).detach().numpy()

	def api_call(self, function=None, symbol=None, outputsize=None, **kwargs):
		if self.ds_flag:
			return self.ds[kwargs.get('ds_type', 'train')]

		function = self.function if function is None else function
		symbol = self.symbol if symbol is None else symbol
		outputsize = self.outputsize if outputsize is None else outputsize

		r = requests.get(self.url(function, symbol, outputsize))
		data = r.json()
		print(f">>>> {data}")

		prices_dict = data[f'Time Series ({self.par})']

		keys = list(prices_dict.keys())[::-1]
		columns = prices_dict[keys[0]].keys()

		data_mat_api = list(map(lambda x: [float(x[e]) for e in columns], prices_dict.values()))[::-1]
		data_mat_api = torch.Tensor(data_mat_api)

		# data_mat_api = data_mat_api[:self.WINDOW]
		return data_mat_api










































