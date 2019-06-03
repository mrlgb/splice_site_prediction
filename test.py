import torch
import torchvision
import torch.optim as optim

import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

from data_loader import Dataset
from config import get_args
from resnet import resnet18, resnet34, resnet50

from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

time = datetime.now().strftime('%m%d_%H%M%S')
args = get_args(time)

batch_size = 64
lr = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Dataset()

true_ie_y = np.ones(len(dataset.true_ie_data))
true_ei_y = np.ones(len(dataset.true_ei_data))
false_ie_y = np.zeros(len(dataset.false_ie_data))
false_ei_y = np.zeros(len(dataset.false_ei_data))

#x = np.concatenate((dataset.true_ie_data, dataset.true_ei_data, dataset.false_ie_data, dataset.false_ei_data), axis=0)
#y = np.concatenate((true_ie_y, true_ei_y, false_ie_y, false_ei_y), axis=0)

# 1:1 ratio dataset
x = np.concatenate((dataset.true_ie_data, dataset.true_ei_data, dataset.false_ie_data[:len(dataset.true_ie_data)], dataset.false_ei_data[:len(dataset.true_ei_data)]), axis=0)
y = np.concatenate((true_ie_y, true_ei_y, false_ie_y[:len(dataset.true_ie_data)], false_ei_y[:len(dataset.true_ei_data)]), axis=0)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
y_valid = torch.from_numpy(y_valid).type(torch.LongTensor)

testset = torch.utils.data.TensorDataset(x_valid, y_valid)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# load our model
print('loading model')
model = None
if args.model == 'resnet18':
    model = resnet18(False, True)
elif args.model == 'resnet34':
    model = resnet34(False, True)
elif args.model == 'resnet50':
    model = resnet50(False, True)


optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))
start_epoch = checkpoint['epoch']
best_loss = checkpoint['best_loss']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

model.to(device)

# test
correct = 0
total = 0
predicts=[]
with torch.no_grad():
	pbar = tqdm(total=len(y_valid)/batch_size, desc='computing validation set score')
	for data in testloader:
		inputs, labels = data[0].unsqueeze(1).to(device), data[1].to(device)
		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

		predicts += predicted.cpu().data.numpy().tolist()
		pbar.update(1)
	pbar.close()

print('Accuracy of the network on the test sequences: %f' % (100.0 * float(correct) / float(total)))

predicts = np.array(predicts)
f1_labels = np.array([0,1])
print('binary f1 score: %f' % (f1_score(y_valid.data.numpy(), predicts, labels=f1_labels, average='binary')))
print('weighted f1 score: %f' % (f1_score(y_valid.data.numpy(), predicts, labels=f1_labels, average='weighted')))

del model

# DSSP test
correct = 0
total = 0

as_dssp = model_from_json(open(os.path.join(os.getcwd(), 'DSSP', 'AS_model.json')).read())
as_dssp.load_weights(os.path.join(os.getcwd(), 'DSSP', 'DS_model.hdf5'))

ds_dssp = model_from_json(open(os.path.join(os.getcwd(), 'DSSP', 'AS_model.json')).read())
ds_dssp.load_weights(os.path.join(os.getcwd(), 'DSSP', 'DS_model.hdf5'))

# s: pytorch tensor with shape (10,15)
def convert_to_dssp_format(s):
	s = s.flatten()
	s = s.astype(int)

	# decode
	seq_types = []
	input_vecs = []
	seq = ''
	i=0
	j=0
	while i < batch_size:
		while j < 140 * 6:
			token = ''.join(map(str,s[i*140*6+j:i*140*6+j+6]))
			if token=='100000':
				seq += 'A'
			elif token=='010000':
				seq += 'C'
			elif token=='001000':
				seq += 'G'
			elif token=='000100':
				seq += 'T'
			elif token=='000010':
				seq += 'N'
			elif token=='000001':
				seq += 'H'
			j += 6

		if len(seq) != 140:
			print('Failed to convert to DSSP input format')
			import ipdb; ipdb.set_trace()
			exit(1)
		seq_type = ''
		if seq[68:70] == 'AG':
			seq_type = 'AS'
		elif seq[70:72] == 'GT':
			seq_type = 'DS'
		else:
			print('Failed to figure out seq type')
			import ipdb; ipdb.set_trace()
			exit()
		input_vec = dataset.one_hot_dssp(seq)
		
		seq_types.append(seq_type)
		input_vecs.append(input_vec)
		i += 1
	return seq_types, input_vecs

correct=0
total=0
dssp_predicts=[]
with torch.no_grad():
	pbar = tqdm(total=len(y_valid), desc='computing DSSP score')
	for data in testloader:
		# check donor/acceptor
		seq_type, inputs = convert_to_dssp_format(data[0].data.numpy()) # get batch_size of inputs
		labels = data[1].data.numpy()

		for t,x,y in zip(seq_type, inputs, labels): # iterate over batch_size
			dssp = None
			if t == 'AS': # AS case
				dssp = as_dssp
			elif t == 'DS': # DS case
				dssp = ds_dssp
			else:
				print('ERROR: cannot allocate dssp model')
				import ipdb; ipdb.set_trace()
				exit(1)
			predict = dssp.predict(x, batch_size=1, verbose=0)[0,0]
			predicted = 1 if predict>0.5 else 0
			total += 1
			correct += (predicted == y)

			dssp_predicts.append(predicted)
			pbar.update(1)
	pbar.close()

print('Accuracy of the DSSP on the test sequences: %f' % (100.0 * float(correct) / float(total)))

dssp_predicts = np.array(dssp_predicts)
print('binary f1 score: %f' % (f1_score(y_valid.data.numpy(), dssp_predicts, labels=f1_labels, average='binary')))
print('weighted f1 score: %f' % (f1_score(y_valid.data.numpy(), dssp_predicts, labels=f1_labels, average='weighted')))

print('Finished Execution')
