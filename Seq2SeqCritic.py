import tensorflow as tf
import numpy as np
import os
from models import Config, Model

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class Seq2SeqCriticConfig(Config):

	def __init__(self):
		self.batch_size = 64
		self.lr = 1e-3
		self.l2_lambda = 0.0000001
		self.hidden_size = 256
		self.num_epochs = 50
		self.num_layers = 3
		self.num_classes = 4 # Mean vector of size 4
		self.features_shape = (100,100,3) #TO FIX!!!!
		self.targets_shape = (4,)
		self.init_loc_size = (4,)
		self.max_norm = 10
		self.keep_prob = 0.8
		self.init_state_out_size = 32
		self.cnn_out_shape = 128
		self.variance = 1e-2


class Seq2SeqCritic(Model):

	def __init__(self, features_shape, num_classes, cell_type='lstm', seq_len=1, reuse=False,
				add_reg=False, scope=None):
		self.config = RecurrentCNNConfig()
		self.config.features_shape = features_shape
		self.config.num_classes = num_classes
		self.reuse = reuse
		self.inputs_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,)+ self.config.features_shape ))
		self.targets_placeholder = tf.placeholder(tf.int32, shape=tuple((None,None,) + self.config.targets_shape))
		self.config.seq_len = seq_len
		self.seq_len_placeholder = tf.placeholder(tf.int32, shape=tuple((None,) ))

		self.scope = scope

		if add_reg:
			self.reg_fn = tf.nn.l2_loss
		else:
			self.reg_fn = None

		if cell_type == 'rnn':
			self.cell = tf.contrib.rnn.RNNCell
		elif cell_type == 'gru':
			self.cell = tf.contrib.rnn.GRUCell
		elif cell_type == 'lstm':
			self.cell = tf.contrib.rnn.LSTMCell
		else:
			raise ValueError('Input correct cell type')


	def build_model(self):


	# def add_loss_op(self):

	# def add_optimizer_op(self):

	# def add_error_op(self):

	# def add_feed_dict(self, input_batch, target_batch, seq_len_batch):

	# def train_one_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch):

	# def test_one_batch(self, session, input_batch, target_batch, seq_len_batch):


	# def run_one_batch(self, args, session, input_batch, target_batch, seq_len_batch , init_locations_batch):

	# def get_config(self):



