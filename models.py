import tensorflow as tf 
import numpy as np
import os

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class Config():

	def __init__(self):
		self.batch_size = 64
		self.lr = 1e-3
		self.l2_lambda = 0.0000001
		self.hidden_size = 256
		self.num_epochs = 50
		self.num_layers = 3
		self.num_classes = 28 #Can change depending on the dataset
		self.num_features = 100 #TO FIX!!!!
		self.max_norm = 10
		self.keep_prob = 0.8


	class RCNN_RNN(object)

		def __init__(self, num_features, num_classes, cell_type='lstm'):
			self.config = Config()
			self.config.num_features = num_features
			self.config.num_classes = num_encodings+1

			self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.num_features))
			self.targets_placeholder = tf.placeholder(tf.int32, shape=(None,))
			self.seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))
			if cell_type == 'rnn':
				self.cell = tf.contrib.rnn.RNNCell
			elif cell_type == 'gru':
				self.cell = tf.contrib.rnn.GRUCell
			elif cell_type == 'lstm':
				self.cell = tf.contrib.rnn.LSTMCell
			else:
				raise ValueError('Input correct cell type')


		def conv2d_layer(self, prev_output, filter_dim, stride, padding, conv_name, filter_name, activ_name, 
										add_drop, add_bn phase):
			filter_wt = tf.get_variable(filter_name, filter_dim, initializer=XAVIER_INIT(dtype=tf.float32),
									 trainable=True)
			conv_layer = tf.nn.conv2d(prev_output, filter_wt, stride, padding='SAME', name=conv_name)
			if add_bn:
				conv_layer  = tf.contrib.layers.batch_norm(inputs=conv_layer ,decay=0.99,center=True, scale=True,
						is_training=phase)
	
			activation = tf.nn.relu(conv_layer, name=activ_name)
			if add_drop and phase:
				activation = tf.nn.dropout(activation, keep_prob=self.keep_prob)

			return activation


		def fc_layer(self, prev_output, hidden_units, keep_prob, weight_name, bias_name, activ_name, 
									add_drop, add_bn, phase):

			input_shape = prev_output.get_shape().as_list()
			weights = tf.get_variable(weight_name, [input_shape[1], hidden_units ], initializer=XAVIER_INIT(dtype=tf.float32),
									 trainable=True)
			bias= tf.get_variable(bias_name, [hidden_units ],
									 trainable=True)

			result = tf.matmul(prev_output, weights) + bias
			activation = tf.nn.relu(result, name=activ_name)

			if add_drop and phase:
				activation = tf.nn.dropout(activation, keep_prob)
			return activation


		def build_model(self):

		def add_loss_op(self):

		def add_optimizer_op(self):
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),self.config.max_norm)
			optimizer = tf.train.AdamOptimizer(self.config.lr)
			self.train_op = optimizer.apply_gradients(zip(grads, tvars))


		def add_error_op(self):

		def add_summary_op(self):
			self.merged_summary_op = tf.summary.merge_all()


		def add_feed_dict(self, input_batch, target_batch, seq_batch):
			feed_dict = {self.inputs_placeholder:input_batch, self.targets_placeholder:target_batch,
							self.seq_lens_placeholder:seq_batch}
			return feed_dict


		def train_one_batch(self, session, input_batch, target_batch, seq_batch):


		def test_one_batch(self, session):

		def get_config(self):
			return self.config



