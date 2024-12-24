import os
import torch
import requests
import numpy as np
from enum import Enum
import google.generativeai as genai
from dotenv import find_dotenv, load_dotenv


class action(Enum):
	BUY = 0
	SELL = 1
	HOLD = 2


def ma(ds, w):
	return torch.Tensor([
		np.concatenate((
			e[:w - 1], 
			np.convolve(e, np.ones(w), 'valid') / w)) 
			for e in ds.T]).T


def window_format_data(data, WINDOW):
	"""Transforms the data from shape [N, n_features] to [N-WINDOW, WINDOW, n_features]

	Args:
		data (_type_): _description_
		WINDOW (_type_): _description_

	Returns:
		tuple: (X, y)
	"""
	X_seq = []
	y_seq = []

	for i in range(WINDOW, data.shape[0]):
		X_seq.append(data[i-WINDOW:i, :])
		y_seq.append(data[i, :5])

	X_seq = np.array(X_seq)
	y_seq = np.array(y_seq)

	X_seq = torch.Tensor(X_seq)
	y_seq = torch.Tensor(y_seq)

	return X_seq, y_seq

def convert_data(data, inter):
	ts = data[f'Time Series ({inter})']
	ts = [ts[e] for e in sorted(ts)]
	cols = list(ts[0].keys())
	ts = list(map(lambda x: list(map(float, x.values()))[:-1], ts))
	ts = torch.Tensor(ts)
	return ts, cols

def format_data(data, inter, WINDOW=60, mas=[], prod=False):
	ts, cols = convert_data(data, inter)
	ts = torch.hstack((ts, *[ma(ts, e) for e in mas]))

	if prod:
		return ts[-WINDOW:], cols  # i.e. only the last window
	
	return window_format_data(ts, WINDOW), cols 

def train_test_split(X_seq, y_seq, split=.8):

	split = int(y_seq.shape[0] * split)
	X_train, y_train = X_seq[:split], y_seq[:split]
	X_val, y_val = X_seq[split:], y_seq[split:]

	return X_train, y_train, X_val, y_val
	
class API:
	def __init__(self, 
			  url='https://www.alphavantage.co/query',
			  **kwargs):
		_ = load_dotenv(find_dotenv())
		self.api_key = os.getenv('ALPHA_API_KEY')
		self.params =  {
			"function": kwargs.get('function', "TIME_SERIES_DAILY"),
			"symbol": kwargs.get('symbol', "AAPL"),
			# "interval": "5min",
			"outputsize": kwargs.get('outputsize', "compact"),
			"apikey": self.api_key
			}
		self.url = url

	def grab_raw_api_data(self):
		r = requests.get(self.url, params=self.params)
		return r.json()
	
	def grab_api_data(self, inter, window=60, mas=[50, 200], prod=False):
		raw_data = self.grab_raw_api_data()
		return format_data(raw_data, inter, window, mas, prod)






