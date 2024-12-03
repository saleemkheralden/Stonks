from dotenv import load_dotenv, find_dotenv
import os
from models import train_model, LSTM
from utils import action
import requests
import torch
import numpy as np

class agent:
	def __init__(self, stock='AAPL', interval='5min', API_KEY=None, **kwargs):
		self.stock = stock

		self.api_key = API_KEY
		if API_KEY is None:
			_ = load_dotenv(find_dotenv())
			self.api_key = os.environ['ALPHA_API_KEY']

		outputsize = kwargs.get('outputsize', 'compact')
		symbol = kwargs.get('symbol', 'AAPL')
		function = kwargs.get('function', 'TIME_SERIES_DAILY')
		self.par = 'Daily' if function.__contains__('DAILY') else '5min'
		self.WINDOW = kwargs.get('WINDOW', 60)
		
		self.url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={self.api_key}'
		self.model = LSTM(output_size=5, num_layers=2)

	def ma(self, ds, w):
		return np.array([
			np.concatenate((
				e[:w - 1], 
				np.convolve(e, np.ones(w), 'valid') / w)) 
				for e in ds.T]).T

	def get_train_set(self):
		raise NotImplementedError()

	def train(self, **kwargs):
		X_train, y_train = self.get_train_set()
		train_model(self.model, X_train, y_train, **kwargs)

	def pred(self):
		data_mat_api = self.api_call()
		return self.model(data_mat_api)

	def api_call(self):
		r = requests.get(self.url)
		data = r.json()
		prices_dict = data[f'Time Series ({self.par})']

		keys = list(prices_dict.keys())[::-1]
		columns = prices_dict[keys[0]].keys()

		data_mat_api = list(map(lambda x: [float(x[e]) for e in columns], prices_dict.values()))[::-1]
		data_mat_api = torch.Tensor(data_mat_api)

		data_mat_api = data_mat_api[:self.WINDOW]
		return data_mat_api










































