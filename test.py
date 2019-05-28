import torch
import torchvision
import torch.optim as optim

import numpy as np
from datetime import datetime
import os

from data_loader import Dataset
from config import get_args
from resnet import resnet18, resnet34, resnet50

from keras.models import model_from_json
from sklearn.model_selection import train_test_split

time = datetime.now().strftime('%m%d_%H%M%S')
args = get_args(time)

batch_size = 64
lr = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Dataset()

true_ie_y = np.ones(len(dataset.true_ie_data))
true_ei_y = np.ones(len(dataset.true_ei_data))
false_ie_y = np.ones(len(dataset.false_ie_data))
false_ei_y = np.ones(len(dataset.false_ei_data))

x = np.concatenate((dataset.true_ie_data, dataset.true_ei_data, dataset.false_ie_data, dataset.false_ei_data), axis=0)
y = np.concatenate((true_ie_y, true_ei_y, false_ie_y, false_ei_y), axis=0)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor)
y_valid = torch.from_numpy(y_valid).type(torch.LongTensor)
testset = torch.utils.data.TensorDataset(x_valid, y_valid)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

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

# test
correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		inputs, labels = data[0].unsqueeze(1).to(device), data[1].to(device)
		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test sequences: %f' % (100.0 * float(correct) / float(total)))

# DSSP test
correct = 0
total = 0

as_dssp = model_from_json(open(os.path.join(os.path.dirname(__file__), 'AS_model.json')).read())
as_dssp.load_weights(os.path.join(os.path.dirname(__file__), 'DS_model.hdf5'))

ds_dssp = model_from_json(open(os.path.join(os.path.dirname(__file__), 'AS_model.json')).read())
ds_dssp.load_weights(os.path.join(os.path.dirname(__file__), 'DS_model.hdf5'))

ag_vec = dataset.one_hot('AG')
gt_vec = dataset.one_hot('GT')

# s: pytorch tensor with shape (10,15)
def convert_to_dssp_format(s):
	s = s.data.numpy()
	s = s.flatten()

	# decode
	seq = ''
	i=0
	while i < 140:
		token = ''.join(s[i:i+6])
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
	
	if len(seq) != 140:
		print('Failed to conver to DSSP input format')
		exit(1)
	seq_type = None
	if seq[68:70] == 'AG':
		seq_type = 'AS'
	if seq[70:72] == 'GT':
		seq_type == 'DS'
	
	BASE_KEY = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}
	input_vec = np.zeros((1,140,5))
	for i in range(len(seq)):
		try:
			input_vec[0][i][BASE_KEY[seq[i]]] = 1
		except KeyError:
			print('Wrong sequence token for DSSP')
	return seq_type, input_vec

with torch.no_grad():
	for data in testloader:
		# check donor/acceptor
		seq_type, inputs, labels = convert_to_dssp_format(data[0]), data[1].data.numpy()
		if seq_type == 'AS': # AS case
			model = as_dssp
		elif seq_type == 'DS': # DS case
			model = ds_dssp

		predict = model.predict(inputs, batch_size=1, verbose=0)[0,0]
		predicted = 1 if predicted>0.5 else 0
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print('Accuracy of the DSSP on the test sequences: %f' % (100.0 * float(correct) / float(total)))

ipdb.set_trace()
print('Finished Execution')

