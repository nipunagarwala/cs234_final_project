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
		self.attention_option = "luong"
		self.start_encode = start_encode
		self.end_encode = end_encode
		# self.bidirectional = False


class Seq2SeqCritic(Model):

	def __init__(self, features_shape, num_classes, cell_type='lstm', seq_len=1, reuse=False,
				add_reg=False, loss_type = 'negative_l1_dist', scope=None):
		self.config = RecurrentCNNConfig()
		self.config.features_shape = features_shape
		self.config.num_classes = num_classes
		self.reuse = reuse
		self.input_size = tuple((None,None,)+ self.config.features_shape )
		self.inputs_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,)+ self.config.features_shape ))
		self.targets_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,) + self.config.targets_shape))
		self.pred_loc_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,) + self.config.targets_shape))

		self.config.seq_len = seq_len
		self.seq_len_placeholder = tf.placeholder(tf.int32, shape=tuple((None,) ))
		self.num_encode = tf.placeholder(tf.int32, shape=(None,), name='Num_encode')
		self.num_decode = tf.placeholder(tf.int32, shape=(None,),  name='Num_decode')
		self.loss_type = loss_type

		self.scope = scope
		self.attn_length = 4

		if add_reg:
			self.reg_fn = tf.nn.l2_loss
		else:
			self.reg_fn = None

		if cell_type == 'rnn':
			self.encoder_cell = tf.contrib.rnn.RNNCell
			self.encoder_attention = tf.contrib.rnn.AttentionCellWrapper(self.encoder_cell(num_units = self.config.hidden_size), 
											self.attn_length, state_is_tuple=True)
			self.decoder_cell = tf.contrib.rnn.RNNCell
		elif cell_type == 'gru':
			self.encoder_cell = tf.contrib.rnn.GRUCell
			self.encoder_attention = tf.contrib.rnn.AttentionCellWrapper(self.encoder_cell(num_units = self.config.hidden_size), 
											self.attn_length, state_is_tuple=True)
			self.decoder_cell = tf.contrib.rnn.GRUCell
		elif cell_type == 'lstm':
			self.encoder_cell = tf.contrib.rnn.LSTMCell
			self.encoder_attention = tf.contrib.rnn.AttentionCellWrapper(self.encoder_cell(num_units = self.config.hidden_size), 
											self.attn_length, state_is_tuple=True)
			self.decoder_cell = tf.contrib.rnn.LSTMCell
		else:
			raise ValueError('Input correct cell type')


	def build_model(self):
		with tf.variable_scope(self.scope):
			def output_fn(outputs):
					return tf.contrib.layers.linear(outputs, 1, scope=scope)

			self.loss_weights = tf.ones([self.config.batch_size, self.input_size], dtype=tf.float32, name="loss_weights")
			encoder_multi = tf.contrib.rnn.MultiRNNCell([self.encoder_attention for _ in
										range(self.config.num_layers)], state_is_tuple=True)
			decoder_multi = tf.contrib.rnn.MultiRNNCell([self.decoder_cell(num_units = self.config.hidden_size) for _ in
										range(self.config.num_layers)], state_is_tuple=True)

			self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder_multi, inputs=self.inputs_placeholder,
								  sequence_length=self.seq_len_placeholder,time_major=True, dtype=tf.float32) #initial_state=initial_tuple)
			
			self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(cell=decoder_multi, inputs=self.targets_placeholder,
								  sequence_length=self.seq_len_placeholder,time_major=True, dtype=tf.float32,
								  initial_state=encoder_multi_state)
			
			cur_shape = tf.shape(self.decoder_outputs)
			rnnOut_2d = tf.reshape(self.decoder_outputs, [-1, cur_shape[2]])

			fc_out = tf.contrib.layers.fully_connected(inputs=rnnOut_2d, num_outputs=self.config.num_classes,
									activation_fn=tf.nn.relu,
									normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
									weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
									scope='fc1', trainable=True)
			self.logits = tf.reshape(logits_2d,[cur_shape[0], cur_shape[1], self.config.num_classes])	

	def get_iou_loss(self):
		p_left = self.pred_loc_placeholder[:, :, :, 1]
		g_left = self.targets_placeholder[:, :, 1]
		left = tf.maximum(p_left, g_left)
		p_right = self.pred_loc_placeholder[:, :, :, 1] + self.pred_loc_placeholder[:, :, :, 3]
		g_right = self.targets_placeholder[:, :, 1] + self.targets_placeholder[:, :, 3]
		right = tf.minimum(p_right, g_right)
		p_top = self.pred_loc_placeholder[:, :, :, 0]
		g_top = self.targets_placeholder[:, :, 0]
		top = tf.maximum(p_top, g_top)
		p_bottom = self.pred_loc_placeholder[:, :, :, 0] + self.pred_loc_placeholder[:, :, :, 2]
		g_bottom = self.targets_placeholder[:, :, 0] + self.targets_placeholder[:, :, 2]
		bottom = tf.minimum(p_bottom, g_bottom)
		intersection = tf.maximum((right - left), 0) * tf.maximum((bottom - top), 0)
		p_area = self.pred_loc_placeholder[:, :, :, 3] * self.pred_loc_placeholder[:, :, :, 2]
		g_area = self.targets_placeholder[:, :, 3] * self.targets_placeholder[:, :, 2]
		union = p_area + g_area - intersection

		return intersection/union

	def add_loss_op(self):
		
		if self.loss_type == 'negative_l1_dist':
			rewards = -tf.reduce_mean(tf.abs(self.inputs_placeholder - tf.cast(self.targets_placeholder,tf.float32)),axis=2,keep_dims=True) - \
					tf.reduce_max(tf.abs(self.inputs_placeholder - tf.cast(self.targets_placeholder,tf.float32)), axis=2,keep_dims=True)
		elif self.loss_type == 'iou':
			rewards = self.get_iou_loss()
			rewards = tf.expand_dims(rewards,axis=-1)

		timestep_rewards = tf.reduce_mean(rewards, axis=0, keep_dims=True)

		pred_qt = rewards + tf.reduce_sum(self.pred_loc_placeholder*self.logits, axis=2)

		tf.summary.scalar('Loss', self.loss_op)

	def add_optimizer_op(self):
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)


	def add_error_op(self):

	def add_feed_dict(self, input_batch, target_batch, pred_loc_batch, seq_len_batch, num_encode_batch, num_decode_batch ):
		feed_dict = {self.inputs_placeholder:input_batch, self.targets_placeholder:target_batch,
						self.pred_loc_placeholder:pred_loc_batch ,self.seq_len_placeholder:seq_len_batch, 
						self.num_encode:num_encode_batch , self.num_decode:num_decode_batch }
		return feed_dict

	def train_one_batch(self, session, input_batch, target_batch, pred_loc_batch, seq_len_batch, 
								num_encode_batch, num_decode_batch):


	def test_one_batch(self, session,input_batch, target_batch, pred_loc_batch, seq_len_batch, 
								num_encode_batch, num_decode_batch):


	def run_one_batch(self, args, session, input_batch, target_batch, pred_loc_batch, seq_len_batch, 
								num_encode_batch, num_decode_batch):

	def get_config(self):
		return self.config



