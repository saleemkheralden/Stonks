from dotenv import load_dotenv, find_dotenv
import os
from models import train_model, STOCK_EMBEDDER
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
		q_param = stock_dict['feed'][0]
		q = f"""Your task is to run a sentiment analysis on the dictionary that's delimited by triple backticks (```) \
this dictionary contains information about a stock in the stock market, you need to process the dictionary \
You need to return only a dictionary that contains whether to Buy or Sell and a score between 0 and 1 of how confident you are with your answer.
the keys of the dictionary should be, action, conf.

Don't add anything to the dictionary that's not mentioned above.
Don't wrap the dictionary in any delimiters.

dictionary:
```{q_param}```
"""
		ret = self.model.generate_content(q).text
		print(ret)
		ret = ret.replace("'", "\"")
		return json.loads(ret)


class Regressor:
	def __init__(self, 
			  symbol='AAPL',
			  interval='5min',
			  API_KEY='demo',
			  input_size=15,
			  hidden_size=16,
			  output_size=5,
			  **kwargs):
		"""Constructor for the Regressor class.

		Args:
			symbol (str, optional): Stock symbol. Defaults to 'AAPL'.
			interval (str, optional): The interval of the data. Defaults to '5min'.
			API_KEY (str, optional): Defaults to 'demo'.
			input_size (int): regressor input size. Defaults to 15 (5 features + 5 MA50 + 5 MA200).
			hidden_size (int): regressor hidden size. Defaults to 16.
			output_size (int): regressor output size. Defaults to 5.
			WINDOW (int, optional): Length of the sequence. Defaults to 60.
			TRAIN_LOOKBACK (int, optional): How much into the past (in years) should the Regressor consider while training. Defaults 5 years.
			ds (dict, optional): Pass ds parameter if you want the Regressor to use custom dataset, should be {'train': train_ds, <other_keyword>: X_ds}, not the API calls, mostly used for debugging.
		"""
		self.symbol = symbol

		self.api_key = API_KEY

		self.outputsize = kwargs.get('outputsize', 'compact')
		self.api_function = kwargs.get('function', 'TIME_SERIES_DAILY')
		self.par = 'Daily' if self.api_function.__contains__('DAILY') else '5min'
		self.WINDOW = kwargs.get('WINDOW', 60)
		self.TRAIN_LOOKBACK = kwargs.get('TRAIN_LOOKBACK', 5) * 365  # TODO: CHANGE 365 IF NOT DAILY
		
		self.url = lambda api_function, symbol, outputsize: f'https://www.alphavantage.co/query?function={api_function}&symbol={symbol}&outputsize={outputsize}&apikey={self.api_key}'
		print(self.url(self.api_function, self.symbol, self.outputsize))
		self.model = STOCK_EMBEDDER(
			input_size=input_size,
			hidden_size=hidden_size, 
			output_size=output_size,
			num_layers=kwargs.get('num_layers', 2),
			dropout=kwargs.get('dropout', .3),
			)
		self.ds = kwargs.get('ds')
		self.ds_flag = self.ds is not None
		self.debug = kwargs.get('debug', False)

		if self.debug: print(f"DEBUG> ds_flag={self.ds_flag}")

	def get_train_set(self, slide=0):
		"""Gets the train set from the API call (if ds parameter is None), 
		trims the train set such that it starts from <TRAIN_LOOKBACK> years ago.
		the data is scaled using MinMaxScaler, then formatted into sequences of <WINDOW> lengths, 
		such that the final shape is (ds.shape[0] - WINDOW, WINDOW, ds.shape[1] (num_features))

		Args:
			slide (int, optional): _description_. Defaults to 0.

		Returns:
			_type_: _description_
		"""
		data_mat = self.ds if self.ds_flag else self.api_call(self.api_function, self.symbol, 'full')
		data_mat = data_mat[-self.TRAIN_LOOKBACK:]

		# sc = StandardScaler()
		sc = MinMaxScaler(feature_range=(0, 1))
		data_mat_scaled = sc.fit_transform(data_mat)
		return reformat_data(data_mat_scaled, self.WINDOW)

	def train(self, **kwargs):
		X_train, y_train = self.get_train_set()
		train_model(self.model, X_train, y_train, **kwargs)

	def pred(self, **kwargs):
		"""Returns prediction on the API call dataset, if on parameter is passed then the pred is calculated on self.ds[on]. 

		Returns:
			_type_: _description_
		"""

		data_mat_api = self.api_call(kwargs=kwargs)
		X, _ = reformat_data(data_mat_api, self.WINDOW)
		return self.model(X).detach().numpy()

	def api_call(self, api_function=None, symbol=None, outputsize=None, **kwargs):
		if self.ds_flag:
			if self.debug: print(f"DEBUG> NO API WAS CALLED, api_call function returned self.ds['{kwargs.get('on', 'train')}']")
			return self.ds[kwargs.get('on', 'train')]
		
		# TODO: expand api matrix with moving averages 50 and 200.

		api_function = self.api_function if api_function is None else api_function
		symbol = self.symbol if symbol is None else symbol
		outputsize = self.outputsize if outputsize is None else outputsize

		r = requests.get(self.url(api_function, symbol, outputsize))
		data = r.json()
		# print(f">>>> {data}")

		prices_dict = data[f'Time Series ({self.par})']

		keys = list(prices_dict.keys())[::-1]
		columns = prices_dict[keys[0]].keys()

		data_mat_api = list(map(lambda x: [float(x[e]) for e in columns], prices_dict.values()))[::-1]
		data_mat_api = torch.Tensor(data_mat_api)

		# data_mat_api = data_mat_api[:self.WINDOW]
		return data_mat_api










































