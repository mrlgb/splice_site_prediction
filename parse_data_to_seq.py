# -!- coding: utf-8 -*-
from tqdm import tqdm
from data_loader import Dataset
dataset = Dataset()

f = open(dataset.true_donor_file, 'r')
true_donor_data = f.readlines()
f.close()

f = open(dataset.true_acceptor_file, 'r')
true_acceptor_data = f.readlines()
f.close()

f = open(dataset.false_donor_file, 'r')
false_donor_data = f.readlines()
f.close()

f = open(dataset.false_acceptor_file, 'r')
false_acceptor_data = f.readlines()
f.close()

# parse
def parse_fasta(data):
	result = []
	i=4
	pbar = tqdm(total=len(data), desc='parsing seq')
	while i < len(data):
		header = data[i].replace('\n', '')
		tokens = header.split()
		name = tokens[0]
		seq = tokens[-1]
		result.append(seq)
		"""
		if len(tokens[-1]) > self.tmp_max_len:
			self.tmp_max_len = len(tokens[-1])
		"""
		i += 1
		pbar.update(1)
	pbar.close()

	# convert to numpy array
	return result

true_donor_seq = parse_fasta(true_donor_data)
true_acceptor_seq = parse_fasta(true_acceptor_data)
false_donor_seq = parse_fasta(false_donor_data)
false_acceptor_seq = parse_fasta(false_acceptor_data)

# write sequences
"""
with open('ei_true_seq.txt', 'w') as f:
	for e in true_donor_seq:
		f.write(e+'\n')

with open('ie_true_seq.txt', 'w') as f:
	for e in true_acceptor_seq:
		f.write(e+'\n')


with open('ei_false_seq.txt', 'w') as f:
	for e in false_donor_seq:
		f.write(e+'\n')

with open('ie_false_seq.txt', 'w') as f:
	for e in false_acceptor_seq:
		f.write(e+'\n')
"""
# write as fasta format
with open('ei_true_seq.fasta', 'w') as f:
	for e in true_donor_seq:
		f.write('>\n')
		f.write(e+'\n')

with open('ie_true_seq.fasta', 'w') as f:
	for e in true_acceptor_seq:
		f.write('>\n')
		f.write(e+'\n')

with open('ei_false_seq.fasta', 'w') as f:
	for e in false_donor_seq:
		f.write('>\n')
		f.write(e+'\n')

with open('ie_false_seq.fasta', 'w') as f:
	for e in false_acceptor_seq:
		f.write('>\n')
		f.write(e+'\n')
