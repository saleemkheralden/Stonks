import gensim.models.doc2vec
import numpy as np
from torch.utils.data import Dataset
from gensim import downloader
import time
import torch
import os
import random


class WordEmbedder:
	def __init__(self, model=None, train=False):
		self.embedding2word = {}
		self.embedders = None
		if isinstance(model, list):
			self.embedders = []
			for m in model:
				self.embedders.append(WordEmbedder(m))

		else:
			s = time.time()
			available_models = ['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', '__testing_word2vec-matrix-synopsis']# list(downloader.info()['models'].keys())
			if not model or model not in available_models:
				error_message = f"No embedding model was chosen or model not supported. Please choose one of the following: {available_models}"
				raise Exception(error_message)

			path = f"{model}_train.d2v"
			if os.path.isfile(path):
				self.embedder = gensim.models.keyedvectors.KeyedVectors.load(path)
			else:
				self.embedder = downloader.load(model)
				self.embedder.save(path)

			self.no_embedding = set()
			self.size_embedding = int(model.split("-")[-1])
			print(f"Loaded: {model} in {time.time() - s: .2f} seconds")


	def embed_word(self, word):
		"""
		Returns tensor which represents the given word, if word not in embedding returns vector of zeros with the same size
		:param word: str word to embed
		:return: tensor
		"""
		if self.embedders:
			return torch.concatenate([embedder.embed_word(word) for embedder in self.embedders], dim=0)
		else:
			embedding = None
			if word in self.embedder:
				embedded = self.embedder[word]
				assert len(embedded) == self.size_embedding
				embedding = torch.from_numpy(np.array(embedded))
			else:
				self.no_embedding.add(word)
				embedding = torch.zeros(self.size_embedding)

			return embedding
		
	def embed(self, x):
		x = x.split()
		print(x)
		return torch.concatenate(tuple((self.embed_word(word).reshape(1, -1) for word in x)))