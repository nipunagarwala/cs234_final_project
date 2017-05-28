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
				add_reg=False, scope=None):
		self.config = RecurrentCNNConfig()
		self.config.features_shape = features_shape
		self.config.num_classes = num_classes
		self.reuse = reuse
		self.inputs_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,)+ self.config.features_shape ))
		self.targets_placeholder = tf.placeholder(tf.int32, shape=tuple((None,None,) + self.config.targets_shape))
		self.config.seq_len = seq_len
		self.seq_len_placeholder = tf.placeholder(tf.int32, shape=tuple((None,) ))
		self.num_encode = tf.placeholder(tf.int32, shape=(None,), name='Num_encode')
		self.num_decode = tf.placeholder(tf.int32, shape=(None,),  name='Num_decode')

		self.scope = scope

		if add_reg:
			self.reg_fn = tf.nn.l2_loss
		else:
			self.reg_fn = None

		if cell_type == 'rnn':
			self.encoder_cell = tf.contrib.rnn.RNNCell
			self.decoder_cell = tf.contrib.rnn.RNNCell
		elif cell_type == 'gru':
			self.encoder_cell = tf.contrib.rnn.GRUCell
			self.decoder_cell = tf.contrib.rnn.GRUCell
		elif cell_type == 'lstm':
			self.encoder_cell = tf.contrib.rnn.LSTMCell
			self.decoder_cell = tf.contrib.rnn.LSTMCell
		else:
			raise ValueError('Input correct cell type')


	def build_model(self):
		with tf.variable_scope(self.scope):
			def output_fn(outputs):
					return tf.contrib.layers.linear(outputs, self.config.vocab_size, scope=scope)

			encoder_multi = tf.contrib.rnn.MultiRNNCell([self.encoder_cell(num_units = self.config.hidden_size) for _ in
										range(self.config.num_layers)], state_is_tuple=True)
			decoder_multi = tf.contrib.rnn.MultiRNNCell([self.decoder_cell(num_units = self.config.hidden_size) for _ in
										range(self.config.num_layers)], state_is_tuple=True)

			self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder_multi, inputs=self.inputs_placeholder,
								  sequence_length=self.seq_len_placeholder,time_major=True, dtype=tf.float32) #initial_state=initial_tuple)
			attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

			attention_keys, attention_values, attention_score_fn, \
						attention_construct_fn = seq2seq.prepare_attention( attention_states=attention_states,
											attention_option=self.attention_option, num_units=self.config.hidden_size)

			decoder_fn_train = seq2seq.attention_decoder_fn_train( encoder_state=self.encoder_state,
							attention_keys=attention_keys, attention_values=attention_values,
							attention_score_fn=attention_score_fn, attention_construct_fn=attention_construct_fn,
							name='attention_decoder')

			decoder_fn_inference = seq2seq.attention_decoder_fn_inference( output_fn=output_fn, encoder_state=self.encoder_state,
							attention_keys=attention_keys, attention_values=attention_values, attention_score_fn=attention_score_fn,
							attention_construct_fn=attention_construct_fn, embeddings=self.embedding_matrix,
							start_of_sequence_id=self.config.start_encode, end_of_sequence_id=self.config.end_encode,
							maximum_length=tf.reduce_max(self.num_decode), num_decoder_symbols=1)
			
			self.decoder_outputs_train, self.decoder_state_train, \
				self.decoder_context_state_train =  seq2seq.dynamic_rnn_decoder( cell=self.decoder_cell,
							decoder_fn=decoder_fn_train, inputs=self.decoder_inputs_embedded,
							sequence_length=self.num_decode, time_major=True, scope=scope)

			scope.reuse_variables()

			self.decoder_logits_inference, self.decoder_state_inference, \
			self.decoder_context_state_inference = seq2seq.dynamic_rnn_decoder(cell=self.decoder_cell,
					decoder_fn=decoder_fn_inference, time_major=True, scope=scope)

			self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1,
											 name='decoder_prediction_inference')



	def add_loss_op(self):
		self.logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
		self.loss_op = seq2seq.sequence_loss(logits=logits, targets=targets,
										  weights=self.loss_weights)
		tf.summary.scalar('Loss', self.loss_op)

	def add_optimizer_op(self):
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op)


	# def add_error_op(self):

	# def add_feed_dict(self, input_batch, target_batch, seq_len_batch):

	# def train_one_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch):

	# def test_one_batch(self, session, input_batch, target_batch, seq_len_batch):


	# def run_one_batch(self, args, session, input_batch, target_batch, seq_len_batch , init_locations_batch):

	# def get_config(self):



