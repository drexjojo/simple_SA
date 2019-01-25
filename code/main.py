import json
import torch
import pickle
import time
import unicodedata
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from constants import *
from model import *


class Model_Data:
	def __init__(self, name):
		self.name = name
		self.train_data = []
		self.train_targets = []
		self.valid_data = []
		self.valid_targets = []
		self.word2index = {}
		self.embeddings = {}
		self.n_words = 0
		self.index2word = {}
	
class Driver_Data(Dataset):
	def __init__(self,data,targets,word2index):
		self.data = data
		self.targets = targets
		self.word2index = word2index
		if len(self.targets) != len(self.data):
			print("[INFO] -> ERROR in Driver Data !")
			exit(0)

	def __getitem__(self, index):
		input_seq = self.get_sequence(self.data[index])
		for i in range(MAX_SEQ_LENGTH - len(input_seq)):
			input_seq.append(self.word2index[PAD_WORD])
		input_seq = np.array(input_seq)
		input_seq = torch.LongTensor(input_seq).view(-1)
		target_class = torch.tensor(int(self.targets[index]),dtype=torch.float)
		return input_seq, target_class

	def get_sequence(self, sentence):
		indices = []
		for word in sentence.split():
			if word in self.word2index.keys():
				indices.append(self.word2index[word])
			else:
				indices.append(self.word2index['unk'])
		return indices

	def __len__(self):
		return len(self.data)

def print_model_details(model_data):
	print("Model name : ",model_data.name)
	print("n_words : ",model_data.n_words)
	print("len(train_data) : ",len(model_data.train_data))
	print("len(train_targets) : ",len(model_data.train_targets))
	print("len(valid_data) : ",len(model_data.valid_data))
	print("len(valid_targets) : ",len(model_data.valid_targets))
	print("len(word2index) : ",len(model_data.word2index))
	print("len(index2word) : ",len(model_data.index2word))
	print("len(Embedding) : ",len(model_data.embeddings))
	print()
	print("Examples : ")
	print()
	print(model_data.train_data[0])
	print(model_data.train_targets[0])
	print(model_data.valid_data[0])
	print(model_data.valid_targets[0])

def get_performance(rounded_preds,targets):
	count = 0
	for i,pred in enumerate(rounded_preds):
		if pred == targets[i] :
			count += 1
	acc = float(count)/float(BATCH_SIZE)
	return acc*100

def train_epoch(model, training_data, optimizer, criterion, device):
	model.train()
	epoch_loss = 0
	epoch_acc = 0

	for batch in tqdm(training_data, mininterval=2,desc='  - (Training)   ', leave=False):
		sequences,targets = map(lambda x: x.to(device), batch)
		optimizer.zero_grad()

		pred = model(sequences).squeeze(1)
		rounded_preds = torch.round(torch.sigmoid(pred))
		loss = criterion(pred,targets)
		acc = get_performance(rounded_preds, targets)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		epoch_acc += acc

	return epoch_loss / len(training_data), epoch_acc / len(training_data)

def eval_epoch(model, valid_data, optimizer, criterion, device):
	model.eval()
	epoch_loss = 0
	epoch_acc = 0

	with torch.no_grad():
		for batch in tqdm(valid_data, mininterval=2,desc='  - (Validating)   ', leave=False):
			sequences,targets = map(lambda x: x.to(device), batch)
			pred = model(sequences).squeeze(1)
			rounded_preds = torch.round(torch.sigmoid(pred))
			loss = criterion(pred,targets)
			acc = get_performance(rounded_preds, targets)
			epoch_loss += loss.item()
			epoch_acc += acc

	return epoch_loss / len(valid_data), epoch_acc / len(valid_data)

def train(model,training_data, validation_data, optimizer ,device):

	criterion = nn.BCEWithLogitsLoss().to(device)
	max_valid_acc = 0
	for epoch_i in range(EPOCH):
		print('[ Epoch', epoch_i, ']')

		start = time.time()
		train_loss, train_accu = train_epoch(model, training_data, optimizer, criterion,device)
		print('  - (Training)   Loss: {ppl: 8.5f} <-> Accuracy: {accu:3.3f} % <-> '\
		'Time Taken : {elapse:3.3f} min'.format(
		ppl=train_loss, accu=train_accu,
		elapse=(time.time()-start)/60))

		start = time.time()
		valid_loss, valid_accu = eval_epoch(model, validation_data, optimizer, criterion,device)
		print('  - (Validating)   Loss: {ppl: 8.5f} <-> Accuracy: {accu:3.3f} % <-> '\
		'Time Taken : {elapse:3.3f} min'.format(
		ppl=valid_loss, accu=valid_accu,
		elapse=(time.time()-start)/60))
		
		if valid_accu >= max_valid_acc:
			max_valid_acc = valid_accu
			model_state_dict = model.state_dict()
			checkpoint = {'model': model_state_dict,'epoch': epoch_i,'acc':valid_accu,'loss':valid_loss}
			model_name = '../models/best_model.chkpt'
			torch.save(checkpoint, model_name)
			print('[INFO] -> The checkpoint file has been updated.')

def main():
	
	print("[INFO] -> Loading Data ...")
	model_data = torch.load("../models/model_data.pt")
	print("[INFO] -> Done.")

	''' Uncomment to print model_data details '''
	#print_model_details(model_data)

	train_dset = Driver_Data(model_data.train_data,model_data.train_targets,model_data.word2index)
	train_loader = DataLoader(train_dset, batch_size = BATCH_SIZE,shuffle = True, num_workers = 10)
	valid_dset = Driver_Data(model_data.valid_data,model_data.valid_targets,model_data.word2index)
	valid_loader = DataLoader(valid_dset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 10)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("[INFO] -> Using Device : ",device)

	model = LSTM(vocab_size=model_data.n_words,glove=model_data.embeddings,word2index=model_data.word2index).to(device)

	optimizer = optim.Adam(model.parameters())
	train(model, train_loader, valid_loader, optimizer, device)

	print("[INFO] -> Best model")
	print("-----------------------")
	best_model = torch.load("../models/best_model.chkpt")
	print("  -[EPOCH]     : ",best_model["epoch"])
	print("  -[ACCURACY]  : ",best_model["acc"])
	print("  -[LOSS]  : ",best_model["loss"])

if __name__ == '__main__':
	main()