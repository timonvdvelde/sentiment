import torch
from torchtext import data, datasets
from torchtext.data import Iterator, BucketIterator
import torch.nn as nn
import numpy as np
import bcolz
import pickle
import torch.optim as optim
import torch.nn.functional as F

#load and preprocess data

reviews = data.Field(lower = True, tokenize='spacy')
labels = data.LabelField(sequential = False)

train, test = datasets.IMDB.splits(reviews, labels)
train, val = train.split(split_ratio = 0.9)
reviews.build_vocab(train, vectors='glove.twitter.27B.25d')#add own embedding not possible
labels.build_vocab(train)

#batch size 50 from paper
train_iter, val_iter = data.BucketIterator.splits(
	(train,val),
	batch_size = 50, 
	device = None,
	sort_key = lambda x: len(x.text),
	repeat = False
	)

test_iter = data.Iterator(
	test,
	batch_size = 50,
	device = None,
	sort = False,
	sort_within_batch = False,
	repeat = False
	)

input_dim = len(reviews.vocab)
emb_dim = 25 
output_dim = 1 
num_filters = 100
filter_size = [3,4,5]
dropout = 0.5

class Net(nn.Module):
	def __init__(self, input_dim, emb_dim):
		super(Net, self).__init__()
		self.embedding=nn.Embedding(input_dim, emb_dim)#?
		self.conv1 = nn.Conv2d(1,100,3,emb_dim)
		self.conv2 = nn.Conv2d(1,100,4,emb_dim) 
		self.conv3 = nn.Conv2d(1,100,5,emb_dim)
		self.dropout = nn.Dropout(0.5)
		self.fc = nn.Linear(300,1) 
		
	def forward(self, x):
		x = x.permute(1, 0)
		x = self.embedding(x)
		x = x.unsqueeze(1)
		conv_a = F.relu(self.conv1(x).squeeze(3))
		conv_b = F.relu(self.conv2(x).squeeze(3))
		conv_c = F.relu(self.conv3(x).squeeze(3))
		pool_a = F.max_pool1d(conv_a, conv_a.shape[2]).squeeze(2)
		pool_b = F.max_pool1d(conv_b, conv_b.shape[2]).squeeze(2)
		pool_c = F.max_pool1d(conv_c, conv_c.shape[2]).squeeze(2)
		x = torch.cat((pool_a, pool_b, pool_c), 1)
		x = self.dropout(x)
		x = self.fc(x)
		prob = F.sigmoid(x)
		return prob

model = Net(input_dim, emb_dim)
pretrained_embeddings = reviews.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
model = model.cuda()

def accuracy(pred, y):
	int_pred = torch.round(pred)
	num_correct = (int_pred == y).float()
	acc = num_correct.sum()/len(num_correct)
	return acc

def train(model, iterator, optimizer):
	running_loss = 0
	running_acc = 0
	model.train()
	model.cuda()
	for batch in iterator:
		optimizer.zero_grad()
		outputs = model(batch.text.cuda()).squeeze(1)
		loss = F.binary_cross_entropy(outputs, batch.label.float().cuda())
		acc = accuracy(outputs, batch.label.float().cuda())
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		running_acc += acc.item()

	return running_loss/len(iterator), running_acc/len(iterator)

def valid(model, iterator):
	eval_loss = 0
	eval_acc = 0
	model.eval()
	with torch.no_grad():

		for batch in iterator:
			outputs = model(batch.text.cuda()).squeeze(1)
			loss = F.binary_cross_entropy(outputs, batch.label.float().cuda())
			acc = accuracy(outputs, batch.label.float().cuda())
			eval_loss += loss.item()
			eval_acc += acc.item()
	return eval_loss/len(iterator), eval_acc/len(iterator)

num_epochs = 10

for epoch in range(num_epochs):

    train_loss, train_acc = train(model, train_iter, optimizer)
    valid_loss, valid_acc = valid(model, val_iter)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc*100:.2f}%')

test_loss, test_acc = valid(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc*100:.2f}%')