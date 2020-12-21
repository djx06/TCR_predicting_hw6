# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score

class SimpleCNN(nn.Module):
	def __init__(self, maxlen, drop_rate=0.5):
		super(SimpleCNN, self).__init__()
		print("Using Simple CNN")
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
		if len(x.shape) == 2:
			x = x.unsqueeze(0)
		logits = self.layers(x)
		logits.permute(0,2,1)
		logits = logits.reshape(-1,128*3)
		logits = self.L(logits)
		score = torch.softmax(logits,1)[:,1].detach()
		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return torch.softmax(logits,dim=1)
		y = y.long()
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch
		auc = roc_auc_score(y,score)
		return loss, acc,auc

class ParallelCNN(nn.Module):
	def __init__(self, maxlen, drop_rate=0.5):
		super(ParallelCNN, self).__init__()
		print("Using Parallel CNN")
		self.maxlen = maxlen
		self.kernel_sizes = [1,3,5,7,9]
		self.convs = [nn.Sequential(
			nn.Conv1d(22,128,kernel),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			nn.MaxPool1d(3,stride=3),
		) for kernel in self.kernel_sizes]
		
		self.L = nn.Sequential(
			nn.Linear(128*44,128),
			nn.ReLU(inplace=True),
			nn.Linear(128,2)
		)
		self.loss = nn.CrossEntropyLoss()
	def forward(self,x, y=None):
		x = x.float()
		x = x.permute(0,2,1)
		if len(x.shape) == 2:
			x = x.unsqueeze(0)
		logits = [conv(x) for conv in self.convs]
		# for i in logits:
		# 	print("Logit shape:",i.shape)
		logits = torch.cat(logits,dim=2)
		# print("Logit shape:",logits.shape)
		logits.permute(0,2,1)
		logits = logits.reshape(-1,128*44)
		logits = self.L(logits)
		score = torch.softmax(logits,1)[:,1].detach()
		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return torch.softmax(logits,dim=1)
		y = y.long()
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch
		auc = roc_auc_score(y,score)
		return loss, acc,auc

class SiameseCNN(nn.Module):
	def __init__(self, maxlen, drop_rate=0.5):
		super(SiameseCNN, self).__init__()
		print("Using Siamese CNN")
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

		self.interaction = nn.Sequential(
			nn.Linear(128*12,128),
			nn.ReLU(inplace=True),
			nn.Linear(128,2)
		)
		
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y, nx, ny):	
		# the 2-class prediction output is named as "logits"
		# inp = torch.tensor(x, dtype=torch.float32)
		x = x.float()
		x = x.permute(0,2,1)
		similarity = (y == ny).long()
		# print(y)
		# print(ny)
		# print(similarity)
		if len(x.shape) == 2:
			x = x.unsqueeze(0)
		logits = self.layers(x)
		logits.permute(0,2,1)
		feature = logits.reshape(-1,128*3)
		logits = self.L(feature)
		score = torch.softmax(logits,1)[:,1].detach()
		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return torch.softmax(logits,dim=1)
		y = y.long()
		loss = self.loss(logits, y)

		nx = nx.float()
		nx = nx.permute(0,2,1)
		if len(nx.shape) == 2:
			nx = nx.unsqueeze(0)
		nlogits = self.layers(nx)
		nlogits.permute(0,2,1)
		nfeature = nlogits.reshape(-1,128*3)
		nlogits = self.L(nfeature)
		nscore = torch.softmax(nlogits,1)[:,1].detach()
		npred = torch.argmax(nlogits, 1)  # Calculate the prediction result
		if ny is None:
			return torch.softmax(nlogits,dim=1)
		ny = ny.long()
		nloss = self.loss(nlogits, ny)

		feature = torch.cat((feature,feature-nfeature,feature*nfeature,nfeature),-1)
		similarity_output = self.interaction(feature)
		simi_loss = self.loss(similarity_output,similarity)

		total_loss = loss + nloss + simi_loss

		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch
		auc = roc_auc_score(y,score)
		return total_loss, acc,auc
