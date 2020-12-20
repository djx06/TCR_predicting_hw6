# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score

class CNN(nn.Module):
	def __init__(self, maxlen, drop_rate=0.5):
		super(CNN, self).__init__()
		self.maxlen = maxlen
		# Define your layers here
		# self.conv1d = nn.Conv1d(maxlen,128,3,padding=1)
		self.layers = nn.Sequential(
			nn.Conv1d(22,128,3),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			nn.MaxPool1d(3,stride=3),
			nn.Conv1d(128,128,3,padding=1),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			nn.MaxPool1d(3,stride=3),
		)
		self.L = nn.Sequential(
		nn.Linear(128*3,128),
		nn.ReLU(inplace=True),
		nn.Linear(128,2)
		)
		
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# the 2-class prediction output is named as "logits"
		# inp = torch.tensor(x, dtype=torch.float32)
		x = x.float()
		x = x.permute(0,2,1)
		# print(x.shape)
		# print(x.shape)
		if len(x.shape) == 2:
			x = x.unsqueeze(0)
		# x = self.conv1d(x)
		# print(x.shape)
		logits = self.layers(x)
		#print(logits.shape)
		logits.permute(0,2,1)
		#print(logits.shape)
		logits = logits.reshape(-1,128*3)
		#print(logits.shape)
		logits = self.L(logits)
		#print(logits.shape)
		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return torch.softmax(logits,dim=1)
		# print(pred)
		# print(y)
		# labels = torch.tensor(y,dtype=torch.int64)
		y = y.long()
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch
		# print(acc)
		# auc = roc_auc_score(pred.int(), y.int())
		auc = 0
		return loss, acc,auc


class LSTM(nn.Module):
	def __init__(self, maxlen, drop_rate=0.5):
		super(LSTM, self).__init__()
		# Define your layers here
		self.dropout = drop_rate
		self.maxlen = maxlen
		self.rnn = nn.LSTM(input_size=64, hidden_size=300, num_layers=1, batch_first=True)
		self.linear = nn.Linear(300,2,bias=True)
		self.h0 = torch.zeros(1,100,300)
		self.c0 = torch.zeros(1,100,300)
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# the 2-class prediction output is named as "logits"
		x = x.float()
		# print(x.shape)
		output,(hn,cn) = self.rnn(x,(self.h0,self.c0))
		# print(output.shape)
		# logits = logits.reshape(100,-1)
		output = output[:,75,:]
		output.squeeze(1)
		logits = self.linear(output)
		# print(logits.shape)
		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		# print(pred.shape)
		if y is None:
			return pred
		y = y.long()
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch
		auc = 0
		return loss, acc,auc