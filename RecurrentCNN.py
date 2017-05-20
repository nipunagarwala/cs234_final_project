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
		self.num_classes = 4 # Mean vector of size 4
		self.features_shape = (100,100,3) #TO FIX!!!!
		self.targets_shape = (self.num_classes,)
		self.config.init_loc_size = (4,)
		self.max_norm = 10
		self.keep_prob = 0.8
		self.init_state_out_size = 32
		self.cnn_out_shape = 128
		self.variance = 1e-2


class RecurrentCNN(Model)

	def __init__(self, features_shape, num_classes, cell_type='lstm', seq_len, reuse=False, add_bn=False,
				add_reg=False, scope=None):
		self.config = Config()
		self.config.features_shape = features_shape
		self.config.num_classes = num_encodings+1
		self.reuse = reuse
		self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None,None,)+ self.config.features_shape )
		self.init_loc = tf.placeholder(tf.float32, shape=(None,)+ self.config.init_loc_size )
		self.targets_placeholder = tf.placeholder(tf.int32, shape=(None,None) + self.targets_shape)
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
		flatten_out = tf.contrib.layers.flatten(max_pool2)

		fc1 = tf.contrib.layers.fully_connected(inputs=flatten_out, num_outputs=self.config.cnn_out_shape,activation_fn=tf.nn.relu,
								normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
								weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
								reuse = self.reuse,trainable=True)
		fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=self.config.cnn_out_shape,activation_fn=tf.nn.relu,
								normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
								weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
								reuse = self.reuse,trainable=True)

		cnn_out = fc2
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

	def build_initial_state(self):
		fc1 = tf.contrib.layers.fully_connected(inputs=self.init_loc, num_outputs=self.init_state_out_size,
								activation_fn=tf.nn.relu,
								normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
								weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
								reuse = self.reuse,trainable=True)
		fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=self.init_state_out_size, 
								activation_fn=tf.nn.relu,
								normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT ,
								weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
								reuse = self.reuse,trainable=True)

		init_state_out = fc2
		return init_state_out



	def build_model(self):
		self.cnn_scope = self.scope + '/CNN'
		obs_outputs = []
		with tf.variable_scope(self.cnn_scope):
			for t in xrange(self.seq_len):
				st_state = tf.zeros(shape=(self.inputs_placeholder.get_shape()[0], self.config.init_state_out_size),
									dtype=tf.float32)
				if t == 0:
					st_state = self.build_initial_state()
				if t > 0: 
					tf.get_variable_scope().reuse_variables()
				concat_result = tf.concat([self.build_cnn(self.inputs_placeholder[:,t,:,:,:]),st_state], axis=1)
				obs_outputs.append(concat_result)

		obs_outputs = tf.stack(cnn_outputs, axis=1)

		self.rnn_scope = self.scope + '/RNN'
		rnn_output = None
		with tf.variable_scope(self.rnn_scope):
			rnn_output = self.build_rnn(obs_outputs)

		self.logits = rnn_output


	def add_loss_op(self):
		location_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.logits, scale=self.config.variance)
		location_samples = location_dist.sample(shape=self.logits.get_shape())

		rewards = - tf.reduce_mean(tf.abs(location_samples - self.targets_placeholder),axis=2) - \
					tf.reduce_max(tf.abs(location_samples - self.targets_placeholder), axis=2)

		timestep_rewards = tf.reduce_mean(rewards, axis=0)

		tvars = tf.trainable_variables()
		self.loss = 1/self.config.variance*(location_samples - self.logits)*(rewards - timestep_rewards)


	def add_optimizer_op(self):
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),self.config.max_norm)
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))


	def add_error_op(self):

	def add_summary_op(self):
		self.merged_summary_op = tf.summary.merge_all()


	def add_feed_dict(self, input_batch, target_batch, init_locations):
		feed_dict = {self.inputs_placeholder:input_batch, self.targets_placeholder:target_batch,
						self.init_loc:init_locations}
		return feed_dict


	def train_one_batch(self, session, input_batch, target_batch, init_locations):
		feed_dict = self.add_feed_dict(input_batch, target_batch, init_locations)
		_, loss = session.run([self.train_op, self.loss], feed_dict)


	def test_one_batch(self, session):

	def get_config(self):
		return self.config