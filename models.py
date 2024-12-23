import torch 
from torch import nn
from torch import optim

class Embedder(nn.Module):
	def __init__(self,
			  input_size=12,
			  hidden_size=32,
			  bidirectional=True,
			  num_layers=2,
			  dropout=.3,
			  **kwargs):
		
		super().__init__()
		self.lstm = nn.LSTM(input_size=input_size, 
					  		hidden_size=hidden_size, 
							batch_first=True,
							bidirectional=bidirectional, 
							num_layers=num_layers, 
							dropout=dropout if num_layers > 1 else 0)
		
	def forward(self, x):
		x, _ = self.lstm(x)
		x = x[:, -1, :]
		return x



class Transformer(nn.Module):
	def __init__(self):
		pass

	def forward(self, x):
		return x


class LSTM(nn.Module):
	def __init__(self, input_size=5, hidden_size=16, output_size=1, num_layers=2, dropout=.3):
		super(LSTM, self).__init__()
		bidirectional = True
		self.lstm = nn.LSTM(input_size=input_size, 
					  		hidden_size=hidden_size, 
							batch_first=True,
							bidirectional=bidirectional, 
							num_layers=num_layers, 
							dropout=dropout if num_layers > 1 else 0)
		self.tanh = nn.Tanh()
		self.regression_head = nn.Sequential(nn.Linear((1 + int(bidirectional)) * hidden_size, hidden_size), 
								  		nn.Linear(hidden_size, output_size))
		
	def forward(self, x):
		x = self.embed(x)
		x = self.tanh(x)

		x = self.regression_head(x)
		return x
	
	def embed(self, x):
		x, _ = self.lstm(x)
		
		x = x[:, -1, :]
		return x
		
	

def train_model(model, X_train, y_train, **kwargs):
	model.train()
	lr = kwargs.get('lr', .0001)
	epochs = kwargs.get('epochs', 30)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	for epoch in range(epochs):
		o = model(X_train)
		loss = criterion(o, y_train)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# for x, y in zip(X_train, y_train):
		# 	o = model(x)
		# 	loss = criterion(o, y)

		# 	# print(loss)

		# 	optimizer.zero_grad()
		# 	loss.backward()
		# 	optimizer.step()
		# 	return
		print(f"[{epoch+1}/{epochs}] - loss: {loss.item()}")



	





