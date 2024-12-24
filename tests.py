from utils import API
from models import Embedder

a = API()
e = Embedder()
x, _ = a.grab_api_data('Daily', 60, prod=True)
print(e.lstm(x))


# import matplotlib.pyplot as plt
# from pipeline import Agent, Gemini, Embedder, LSTM
# import pandas as pd
# import numpy as np
# from utils import ma, window_format_data, format_data, train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from pprint import pprint
# from dotenv import find_dotenv, load_dotenv
# import os 
# import pickle
# from pprint import pprint

# _ = load_dotenv(find_dotenv())
# news_data = None
# with open('NEWS_DATA.pkl', 'rb') as file:
# 	news_data = pickle.load(file)

# data = pd.read_csv('AAPL.csv')
# data.set_index('Date', inplace=True)
# data = data[['Open', 'High', 'Low', 'Close', 'Adj Close']]

# years = 5
# s = years * 365

# data_mat_np = data.to_numpy()
# data_mat_np = data_mat_np[-s:]

# data_mat = np.hstack((data_mat_np, 
# 					  ma(data_mat_np, 50),
# 					  ma(data_mat_np, 200)))


# # sc = StandardScaler()
# # sc = MinMaxScaler(feature_range=(0, 1))
# # data_mat_scaled = sc.fit_transform(data_mat)
# X, y = window_format_data(data_mat, 60)
# _, _, X_val, _ = train_test_split(X, y)

# ds = {
# 	'train': data_mat,
# 	'validation': X_val,
# 	'debug': -1,
# 	'news': news_data,
# }


# a = Embedder(ds=ds, debug=True, news=True)
# # print(a.regressor.pred(on='validation'))
# print(f"ds_flag: {a.ds_flag}")
# print(f"debug: {a.debug}")
# print(f"news_flag: {a.news_flag}")
# news_formatted = a.get_data()
# print(news_formatted)


# # print(x.shape)
# # emb_x = a.embed()
# # print(f"EMBEDDING SHAPE: {emb_x.shape}")
# # print(emb_x)

# # lstm = LSTM(15, 16, 5, 2)
# # print(lstm(x))

# # print(a.regressor.get_train_set()[0].shape)
# # print(X.shape)



# # plt.plot(data_mat_np[:, 0])
# # plt.plot(ma(data_mat_np, 50)[:, 0])
# # plt.plot(ma(data_mat_np, 200)[:, 0])
# # plt.show()

# # a = agent()

# # a.train(lr=0.01, epochs=30)
# # pred = a.pred()


# # plt.plot(pred)
# # plt.show()






