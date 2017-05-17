import numpy as np 
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import math
import random
import tensorflow as tf
import time
# from utils import *




def train_model(args):
	




def main(args):
	if args.phase == 'train':
		train_model(args)
	elif args.phase == 'test':
		test_model(args)






if __name__ == '__main__':
	print("HEY!")
	args = parse_commandline()
	main(args)
