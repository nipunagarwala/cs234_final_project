# import tensorflow as tf 
# import numpy as np
# import os
# from models import Config, Model

# XAVIER_INIT = tf.contrib.layers.xavier_initializer


# class VisualAttentionConfig(Config):

# 	def __init__(self):
# 		self.batch_size = 64
# 		self.lr = 1e-3
# 		self.l2_lambda = 0.0000001
# 		self.hidden_size = 256
# 		self.num_epochs = 50
# 		self.num_layers = 3
# 		self.num_classes = 28 #Can change depending on the dataset
# 		self.features_shape = (100,100,3) #TO FIX!!!!
# 		self.targets_shape = (self.num_classes,4)
# 		self.max_norm = 10
# 		self.keep_prob = 0.8


# class VisualAttention(Model):

# 	def __init__(self, features_shape, num_classes, cell_type='lstm', seq_len, reuse=False, add_bn=False,
# 				add_reg=False, scope=None):
# 		self.config = Config()
# 		self.config.features_shape = features_shape
# 		self.config.num_classes = num_classes
# 		self.reuse = reuse
# 		self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None,)+ self.config.features_shape )
# 		self.targets_placeholder = tf.placeholder(tf.int32, shape=(None,) + self.targets_shape)
# 		self.config.seq_len = seq_len
# 		self.emission_num_layers =  1

# 		self.scope = scope
# 		if add_bn:
# 			self.norm_fn = tf.contrib.layers.batch_norm
# 		else
# 			self.norm_fn = None

# 		if add_reg:
# 			self.reg_fn = tf.nn.l2_loss
# 		else
# 			self.reg_fn = None

# 		if cell_type == 'rnn':
# 			self.cell = tf.contrib.rnn.RNNCell
# 		elif cell_type == 'gru':
# 			self.cell = tf.contrib.rnn.GRUCell
# 		elif cell_type == 'lstm':
# 			self.cell = tf.contrib.rnn.LSTMCell
# 		else:
# 			raise ValueError('Input correct cell type')


# 	def context_network(self, scope):
# 		with tf.variable_scope(scope):
# 			conv_out1 = tf.contrib.layers.conv2d(inputs=self.inputs_placeholder , num_outputs=32, kernel_size=[3,3],
# 								stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
# 								normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
# 								weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
# 								reuse = self.reuse, trainable=True)
# 			conv_out2 = tf.contrib.layers.conv2d(inputs=conv_out1, num_outputs=32, kernel_size=[3,3],
# 								stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
# 								normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
# 								weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
# 								reuse = self.reuse, trainable=True)
# 			conv_out3 = tf.contrib.layers.conv2d(inputs=conv_out2, num_outputs=32, kernel_size=[3,3],
# 								stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
# 								normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
# 								weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
# 								reuse = self.reuse, trainable=True)
# 			self.context_net_out = tf.contrib.layers.flatten(conv_out3)


# 	def glimpse_network(self, scope):
# 		with tf.variable_scope(scope):

# 	def emission_network(self, scope):
# 		with tf.variable_scope(scope):
# 			emission_cell = tf.contrib.rnn.MultiRNNCell([self.cell(num_units = self.config.hidden_size) for _ in 
# 										range(self.emission_num_layers)], state_is_tuple=True)
# 			(self.emission_out, self.emission_state) = tf.nn.dynamic_rnn(cell = emission_cell, 
# 							inputs=tf.zeros_like(self.context_net_out), sequence_length=self.config.seq_len+1,
# 							initial_state=self.context_net_out,dtype=tf.float32)



# 	def classification_network(self, scope):
# 		with tf.variable_scope(scope):
