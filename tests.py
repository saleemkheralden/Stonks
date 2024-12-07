import matplotlib.pyplot as plt
from pipeline import Agent, Gemini
import pandas as pd
import numpy as np
from utils import ma, reformat_data, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pprint import pprint
from dotenv import find_dotenv, load_dotenv
import os 

_ = load_dotenv(find_dotenv())

agent= Agent(API_KEY=os.environ['ALPHA_API_KEY'])
# print(agent())
# model = Gemini()

data = pd.read_csv('AAPL.csv')
data.set_index('Date', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Adj Close']]

years = 5
s = years * 365

data_mat_np = data.to_numpy()
data_mat_np = data_mat_np[-s:]

data_mat = np.hstack((data_mat_np, 
					  ma(data_mat_np, 50),
					  ma(data_mat_np, 200)))


# sc = StandardScaler()
# sc = MinMaxScaler(feature_range=(0, 1))
# data_mat_scaled = sc.fit_transform(data_mat)
X, y = reformat_data(data_mat, 60)
X_train, _, X_val, _ = train_test_split(X, y)

ds = {
	'train': data_mat,
	'validation': X_val
}

a = Agent(ds=ds, debug=True)
print(a.regressor.pred(on='validation'))
# print(a.regressor.ds_flag)
# print(a.regressor.get_train_set()[0].shape)
# print(X.shape)



# plt.plot(data_mat_np[:, 0])
# plt.plot(ma(data_mat_np, 50)[:, 0])
# plt.plot(ma(data_mat_np, 200)[:, 0])
# plt.show()

# a = agent()

# a.train(lr=0.01, epochs=30)
# pred = a.pred()


# plt.plot(pred)
# plt.show()






