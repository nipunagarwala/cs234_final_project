import tensorflow as tf 
import numpy as np
import os
from models import Config, Model

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class RecurrentCNNConfig(Config):

	def __init__(self):
		self.batch_size = 64
		self.lr = 1e-3
		self.l2_lambda = 0.0000001
		self.hidden_size = 256
		self.num_epochs = 50
		self.num_layers = 3
		self.num_classes = 28 #Can change depending on the dataset
		self.features_shape = (100,100,3) #TO FIX!!!!
		self.targets_shape = (self.num_classes,4)
		self.max_norm = 10
		self.keep_prob = 0.8


class RecurrentCNN(Model)

	def __init__(self, features_shape, num_classes, cell_type='lstm', seq_len, reuse=False, add_bn=False,
				add_reg=False, scope=None):
		self.config = Config()
		self.config.features_shape = features_shape
		self.config.num_classes = num_encodings+1
		self.reuse = reuse
		self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None,None,)+ self.config.features_shape )
		self.targets_placeholder = tf.placeholder(tf.int32, shape=(None,) + self.targets_shape)
		self.config.seq_len = seq_len

		self.scope = scope
		if add_bn:
			self.norm_fn = tf.contrib.layers.batch_norm
		else
			self.norm_fn = None

		if add_reg:
			self.reg_fn = tf.nn.l2_loss
		else
			self.reg_fn = None

		if cell_type == 'rnn':
			self.cell = tf.contrib.rnn.RNNCell
		elif cell_type == 'gru':
			self.cell = tf.contrib.rnn.GRUCell
		elif cell_type == 'lstm':
			self.cell = tf.contrib.rnn.LSTMCell
		else:
			raise ValueError('Input correct cell type')


	def build_cnn(self, cur_inputs):
		conv_out1 = tf.contrib.layers.conv2d(inputs=cur_inputs, num_outputs=32, kernel_size=[3,3],
							stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
							normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
							weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
							reuse = self.reuse, trainable=True)

		conv_out2 = tf.contrib.layers.conv2d(inputs=conv_out1, num_outputs=32, kernel_size=[3,3],
							stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
							normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
							weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
							reuse = self.reuse, trainable=True)

		max_pool1 = tf.contrib.layers.max_pool2d(inputs=conv_out2, kernel_size=[3,3],stride=[1,1],padding='SAME')

		conv_out3 = tf.contrib.layers.conv2d(inputs=max_pool1, num_outputs=32, kernel_size=[3,3],
							stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
							normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
							weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
							reuse = self.reuse, trainable=True)

		conv_out4 = tf.contrib.layers.conv2d(inputs=conv_out3, num_outputs=32, kernel_size=[3,3],
							stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
							normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
							weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
							reuse = self.reuse, trainable=True)

		max_pool2 = tf.contrib.layers.max_pool2d(inputs=conv_out4, kernel_size=[3,3],stride=[1,1],padding='SAME')

		cnn_out = max_pool2
		return cnn_out

	def build_rnn(self, rnn_inputs):
		W = tf.get_variable("Weights", shape=[self.config.hidden_size, self.config.num_classes],
							initializer=XAVIER_INIT)
		b = tf.get_variable("Bias", shape=[self.config.num_classes])

		rnnNet = tf.contrib.rnn.MultiRNNCell([self.cell(num_units = self.config.hidden_size) for _ in 
									range(self.config.num_layers)], state_is_tuple=True)
		(rnnNet_out, rnnNet_state) = tf.nn.dynamic_rnn(cell = rnnNet, inputs=rnn_inputs,
		                sequence_length=self.config.seq_len,dtype=tf.float32)

		cur_shape = tf.shape(rnnNet_out)
		rnnOut_2d = tf.reshape(rnnNet_out, [-1, cur_shape[2]])

		logits_2d = tf.matmul(rnnOut_2d, W) + b
		rnn_out = tf.reshape(logits_2d,[cur_shape[0], cur_shape[1], self.config.num_classes])

		return rnn_out

	def build_model(self):
		self.cnn_scope = self.scope + '/CNN'
		cnn_outputs = []
		with tf.variable_scope(self.cnn_scope):
			for t in xrange(self.seq_len):
				if t > 0:  tf.get_variable_scope().reuse_variables()
				cnn_outputs.append(self.build_cnn(self.inputs_placeholder[t,:,:,:,:]))

		cnn_outputs = tf.stack(cnn_outputs, axis=1)

		self.rnn_scope = self.scope + '/RNN'
		rnn_output = None
		with tf.variable_scope(self.rnn_scope):
			rnn_output = self.build_rnn(cnn_outputs)

		self.logits = rnn_output


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
		feed_dict = self.add_feed_dict(input_batch, target_batch, seq_batch)


	def test_one_batch(self, session):

	def get_config(self):
		return self.config