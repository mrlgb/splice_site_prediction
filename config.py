# -*- coding: utf-8 -*-
import argparse
import os

def get_args(time):
	parser = argparse.ArgumentParser(description='Splice Site Prediction')

	# model config
	parser.add_argument('-model', type=str, default='resnet34', choices=['resnet18', 'resnet34', 'resnet50'], help='resnet model selection')

	parser.add_argument('-epoch', type=int, default=2, help='training epochs')

	# directory config
	parser.add_argument('-output', type=str, default=os.path.join('output', time), help='directory to output results')

	return parser.parse_args()
