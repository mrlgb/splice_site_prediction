import numpy as np
import pickle
import ipdb
from tqdm import tqdm
from datetime import datetime

from data_loader import Dataset
from config import get_args, set_dir

from resnet import resnet18, resnet34, resnet50
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import os

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

# set config
time = datetime.now().strftime('%m%d_%H%M%S')

# parse arguments
args = get_args(time)
set_dir(args)

# model parameters
batch_size = 64
lr = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Dataset()

# prepare labels
true_ie_y = np.ones(len(dataset.true_ie_data))
true_ei_y = np.ones(len(dataset.true_ei_data))
false_ie_y = np.zeros(len(dataset.false_ie_data))
false_ei_y = np.zeros(len(dataset.false_ei_data))

# prepare true,false dataset with specified ratio
x = np.concatenate((dataset.true_ie_data, dataset.true_ei_data, dataset.false_ie_data, dataset.false_ei_data), axis=0)
y = np.concatenate((true_ie_y, true_ei_y, false_ie_y, false_ei_y), axis=0)

"""
# 1:1 ratio
x = np.concatenate((dataset.true_ie_data, dataset.true_ei_data, dataset.false_ie_data[:len(dataset.true_ie_data)], dataset.false_ei_data[:len(dataset.true_ei_data)]), axis=0)
y = np.concatenate((true_ie_y, true_ei_y, false_ie_y[:len(dataset.true_ie_data)], false_ei_y[:len(dataset.true_ei_data)]), axis=0)
"""

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
print('training set size: ' + str(len(y_train)))
print('validation set size: ' + str(len(y_valid)))

# convert to pytorch tensor
x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_valid = torch.from_numpy(y_valid).type(torch.LongTensor)

trainset = torch.utils.data.TensorDataset(x_train, y_train)
testset = torch.utils.data.TensorDataset(x_valid, y_valid)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# build model
class Net(nn.Module):
	def __init__(self, layers, norm_layer=None):
		super(Net, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

resnet = None
if args.model == 'resnet18':
	resnet = resnet18(False, True)
elif args.model == 'resnet34':
	resnet = resnet34(False, True)
elif args.model == 'resnet50':
	resnet = resnet50(False, True)
resnet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=0.9)

# train
best = 99
best_state_dict = None
best_optimizer = None

print('Start training')
pbar = tqdm(total=int(args.epoch * len(y_train)/batch_size), desc='training')
for epoch in range(args.epoch):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get inputs; data is a list of [inputs, labels]
		inputs, labels = data[0].unsqueeze(1).to(device), data[1].to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = resnet(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999: # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			if running_loss < best:
				best = running_loss
				best_state_dict = resnet.state_dict()
				best_optimizer = optimizer.state_dict()

			save_checkpoint({
					'epoch': epoch,
					'best_loss': running_loss,
					'state_dict': resnet.state_dict(),
					'optimizer': optimizer.state_dict(),
					}, os.path.join(args.output, 'checkpoint.pth.tar'))
			running_loss = 0.0
		pbar.update(1)
pbar.close()

# save best case
print('saving best case')
save_checkpoint({
		'epoch': epoch,
		'best_loss': best,
		'sate_dict': best_state_dict,
		'optimizer': best_optimizer,
		}, os.path.join(args.output, 'model_best.pth.tar'))

print('Finished Training')

# test
correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		inputs, labels = data[0].unsqueeze(1).to(device), data[1].to(device)
		outputs = resnet(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('total test cases: ' + str(total))
print('Accuracy of the network on the test sequences: %f' % (
			    float(100) * float(correct) / float(total)))

ipdb.set_trace()
print('Finished Execution')
