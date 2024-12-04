from dotenv import load_dotenv, find_dotenv
import os
from models import train_model, LSTM
from utils import action
import requests
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class agent:
	def __init__(self, stock='AAPL', interval='5min', API_KEY=None, **kwargs):
		self.stock = stock

		self.api_key = API_KEY
		if API_KEY is None:
			_ = load_dotenv(find_dotenv())
			self.api_key = os.environ['ALPHA_API_KEY']

		self.outputsize = kwargs.get('outputsize', 'compact')
		self.symbol = kwargs.get('symbol', 'AAPL')
		self.function = kwargs.get('function', 'TIME_SERIES_DAILY')
		self.par = 'Daily' if self.function.__contains__('DAILY') else '5min'
		self.WINDOW = kwargs.get('WINDOW', 60)
		self.TRAIN_LOOKBACK = kwargs.get('TRAIN_LOOKBACK', 5) * 365  # TODO: CHANGE 365 IF NOT DAILY
		
		self.url = lambda function, symbol, outputsize: f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={self.api_key}'
		self.model = LSTM(output_size=5, num_layers=2)

	def ma(self, ds, w):
		return np.array([
			np.concatenate((
				e[:w - 1], 
				np.convolve(e, np.ones(w), 'valid') / w)) 
				for e in ds.T]).T
	
	def reformat_mat(self, mat):
		X_seq = []
		y_seq = []

		for i in range(self.WINDOW, mat.shape[0]):
			X_seq.append(mat[i-self.WINDOW:i, :])
			y_seq.append(mat[i, :5])

		X_seq = np.array(X_seq)
		y_seq = np.array(y_seq)

		X_seq = torch.Tensor(X_seq)
		y_seq = torch.Tensor(y_seq)

		return X_seq, y_seq

	def get_train_set(self, slide=10):
		data_mat = self.api_call(self.function, self.symbol, 'full')
		data_mat = data_mat[-self.TRAIN_LOOKBACK - slide: -slide]

		# sc = StandardScaler()
		sc = MinMaxScaler(feature_range=(0, 1))
		data_mat_scaled = sc.fit_transform(data_mat)
		return self.reformat_mat(data_mat_scaled)


	def train(self, **kwargs):
		X_train, y_train = self.get_train_set()
		train_model(self.model, X_train, y_train, **kwargs)

	def pred(self):
		data_mat_api = self.api_call()
		X, _ = self.reformat_mat(data_mat_api)
		return self.model(X).detach().numpy()

	def api_call(self, function=None, symbol=None, outputsize=None):
		function = self.function if function is None else function
		symbol = self.symbol if symbol is None else symbol
		outputsize = self.outputsize if outputsize is None else outputsize

		r = requests.get(self.url(function, symbol, outputsize))
		data = r.json()
		# print(data)

		prices_dict = data[f'Time Series ({self.par})']

		keys = list(prices_dict.keys())[::-1]
		columns = prices_dict[keys[0]].keys()

		data_mat_api = list(map(lambda x: [float(x[e]) for e in columns], prices_dict.values()))[::-1]
		data_mat_api = torch.Tensor(data_mat_api)

		# data_mat_api = data_mat_api[:self.WINDOW]
		return data_mat_api










































