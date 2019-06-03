import numpy as np

import pickle
import ipdb
from tqdm import tqdm
import os
import random
import re

class Dataset():
	def __init__(self):
		self.true_donor_file = os.path.join(os.getcwd(), 'data', 'EI_true.seq') # exon/intron
		self.true_acceptor_file = os.path.join(os.getcwd(), 'data', 'IE_true.seq') # intron/exon
		self.false_donor_file = os.path.join(os.getcwd(), 'data', 'EI_false.seq') # exon/intron
		self.false_acceptor_file = os.path.join(os.getcwd(), 'data', 'IE_false.seq') # intron/exon

		self.true_ei_pickle_path = os.path.join(os.getcwd(), 'data', 'ei_true_data.pkl')
		self.true_ie_pickle_path = os.path.join(os.getcwd(), 'data', 'ie_true_data.pkl')
		self.false_ei_pickle_path = os.path.join(os.getcwd(), 'data', 'ei_false_data.pkl')
		self.false_ie_pickle_path = os.path.join(os.getcwd(), 'data', 'ie_false_data.pkl')

		self.acid_codes = ['A','C','G','T', 'N', 'H']
		self.idx_dict = {}
		for i, c in enumerate(self.acid_codes):
			self.idx_dict[c] = i
		self.max_len = 140
		self.num_class = 2

		self.dssp_codes = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4}
		self.dssp_vec = np.zeros((1,140,5))

		self.label_sheet = np.eye(self.num_class, dtype='int64')

		self.tmp_max_len = 140

		if not os.path.isfile(self.true_ie_pickle_path):
			print('preprocessing data')
			# read
			f = open(self.true_donor_file, 'r')
			true_donor_data = f.readlines()
			f.close()

			f = open(self.true_acceptor_file, 'r')
			true_acceptor_data = f.readlines()
			f.close()

			f = open(self.false_donor_file, 'r')
			false_donor_data = f.readlines()
			f.close()

			f = open(self.false_acceptor_file, 'r')
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
					seq = self.one_hot(tokens[-1])
					result.append(seq)
					"""
					if len(tokens[-1]) > self.tmp_max_len:
						self.tmp_max_len = len(tokens[-1])
					"""
					i += 1
					pbar.update(1)
				pbar.close()

				# convert to numpy array
				result = np.array(result, dtype=np.float).reshape((-1, 35, 24))
				return result

			true_donor_list = parse_fasta(true_donor_data)
			true_acceptor_list = parse_fasta(true_acceptor_data)
			false_donor_list = parse_fasta(false_donor_data)
			false_acceptor_list = parse_fasta(false_acceptor_data)

			print('true donor data: ' + str(true_donor_list.shape))
			print('true acceptor data: ' + str(true_acceptor_list.shape))
			print('false donor data: ' + str(false_donor_list.shape))
			print('false acceptor data: ' + str(false_acceptor_list.shape))

			#print('max len: ' + str(self.tmp_max_len)) # 140

			# save data
			with open(self.true_ei_pickle_path, 'wb') as f:
				pickle.dump(true_donor_list, f, pickle.HIGHEST_PROTOCOL)

			with open(self.true_ie_pickle_path, 'wb') as f:
				pickle.dump(true_acceptor_list, f, pickle.HIGHEST_PROTOCOL)

			with open(self.false_ei_pickle_path, 'wb') as f:
				pickle.dump(false_donor_list, f, pickle.HIGHEST_PROTOCOL)

			with open(self.false_ie_pickle_path, 'wb') as f:
				pickle.dump(false_acceptor_list, f, pickle.HIGHEST_PROTOCOL)

		# load data
		print('loading data')
		with open(self.true_ei_pickle_path, 'rb') as f:
			self.true_ei_data = pickle.load(f)
		with open(self.true_ie_pickle_path, 'rb') as f:
			self.true_ie_data = pickle.load(f)
		with open(self.false_ei_pickle_path, 'rb') as f:
			self.false_ei_data = pickle.load(f)
		with open(self.false_ie_pickle_path, 'rb') as f:
			self.false_ie_data = pickle.load(f)

	def one_hot(self, s, max_len = None):
		if max_len is None:
			max_len = self.max_len
		s = s.upper()
		str2vec = np.zeros((max_len,len(self.acid_codes)), dtype=np.float)
		max_length = min(len(s), max_len)
		for i in range(max_length):
			c = s[i]
			if c in self.acid_codes:
				str2vec[i][self.idx_dict[c]] = 1
			else:
				ipdb.set_trace()
				print(c)
		return str2vec.flatten()

	def one_hot_dssp(self, s):
		input_vec = self.dssp_vec
		for i in range(len(s)):
			try:
				input_vec[0][i][self.dssp_codes[s[i]]] = 1
			except KeyError:
				print('Wrong sequence token for DSSP')
		return input_vec
