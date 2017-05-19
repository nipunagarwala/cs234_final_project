import tensorflow as tf 
import numpy as np
import os

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class Config(object):

	def __init__(self):
		pass


class Model(object)

	def __init__(self):
		pass


	def build_model(self):
		pass


	def add_loss_op(self):
		pass

	def add_optimizer_op(self):
		pass

	def add_error_op(self):
		pass

	def add_summary_op(self):
		pass


	def add_feed_dict(self, input_batch, target_batch, seq_batch):
		pass


	def train_one_batch(self, session, input_batch, target_batch, seq_batch):
		pass


	def test_one_batch(self, session):
		pass

	def get_config(self):
		pass





