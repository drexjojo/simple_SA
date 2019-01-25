import torch
import pickle
import unicodedata
import numpy as np
import bcolz
import json
from nltk import word_tokenize
from tqdm import tqdm
from constants import *

class Model_Data:
	def __init__(self, name):
		self.name = name
		self.train_data, self.train_targets, self.valid_data, self.valid_targets = self.read_data()
		self.word2index , self.embeddings = self.build_vocab()
		self.n_words = len(self.word2index.keys())
		self.index2word = {}
		for word,index in self.word2index.items():
			self.index2word[index] = word

	def build_vocab(self):
		
		words = []
		word2idx = {}
		vectors = bcolz.carray(np.zeros(1), rootdir='../models/6B.200.dat', mode='w')
		words.append(PAD_WORD)
		word2idx[PAD_WORD] = PAD
		vect = np.array(([0 for i in range(200)] + [1])).astype(np.float)
		vectors.append(vect)

		idx = 1
		with open("../models/glove.6B.200d.txt", 'rb') as f:
			for l in tqdm(f,desc = "Building Vocab ..."):
				line = l.decode().split()
				word = line[0]
				words.append(word)
				word2idx[word] = idx
				idx += 1
				vect = np.array(line[1:]+[0]).astype(np.float)
				vectors.append(vect)

		vectors = bcolz.carray(vectors[1:].reshape((400001, 201)), rootdir='../models/6B.200.dat', mode='w')
		vectors.flush()
		glove = {w: vectors[word2idx[w]] for w in words}
		return word2idx, glove

	def unicode_to_ascii(self, s):
		return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

	def read_data(self):

		all_pairs = []

		with open(DATA_FILE) as f:
			json_data = json.load(f)

		for i in tqdm(json_data["data"],desc = "Reading training file ..." ) :
			sentence = self.unicode_to_ascii(" ".join(word_tokenize(i[0])))
			if len(sentence.split()) <= MAX_SEQ_LENGTH:
				if i[1] == "positive":
					label = 1
				else :
					label = 0
				all_pairs.append([sentence,label])

		print("Splitting data into train/valid !")
		validation_split = 0.2
		random_seed = 42
		dataset_size = len(all_pairs)
		indices = list(range(dataset_size))
		split = int(np.floor(validation_split * dataset_size))
		np.random.seed(random_seed)
		np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
		
		train_data = []
		valid_data = []
		train_targets = []
		valid_targets = []

		for ind in tqdm(train_indices,desc = "Prep Training data") :
			train_data.append(all_pairs[ind][0])
			train_targets.append(all_pairs[ind][1])

		for ind in tqdm(val_indices,desc = "Prep Valid data") :
			valid_data.append(all_pairs[ind][0])
			valid_targets.append(all_pairs[ind][1])

		return train_data, train_targets, valid_data, valid_targets

def main():
	
	model_data = Model_Data("Sentiment_data")
	print("Saving file !")
	torch.save(model_data, "../models/model_data.pt")

if __name__ == '__main__':
	main()