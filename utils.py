import os
import torch
import numpy as np
from enum import Enum
import google.generativeai as genai
from dotenv import find_dotenv, load_dotenv

class action(Enum):
	BUY = 0
	SELL = 1
	HOLD = 2


def ma(ds, w):
	return np.array([
		np.concatenate((
			e[:w - 1], 
			np.convolve(e, np.ones(w), 'valid') / w)) 
			for e in ds.T]).T


def window_format_data(data, WINDOW):
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

def format_data(data, inter):
	ts = data[f'Time Series ({inter})']
	ts = [ts[e] for e in sorted(ts)]
	cols = list(ts[0].keys())
	ts = list(map(lambda x: list(map(float, x.values()))[:-1], ts))
	ts = torch.Tensor(ts)
	return ts, cols

def train_test_split(X_seq, y_seq, split=.8):

	split = int(y_seq.shape[0] * split)
	X_train, y_train = X_seq[:split], y_seq[:split]
	X_val, y_val = X_seq[split:], y_seq[split:]

	return X_train, y_train, X_val, y_val
	



