import tensorflow as tf 
import numpy as np
import os




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


	class RCNN_RNN(object)

		def __init__(self, num_features, num_classes, cell_type='lstm'):
			self.config = Config()
			self.config.num_features = num_features
			self.config.num_classes = num_encodings+1
			self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, num_features))
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


		def build_model(self):

		def add_loss_op(self):

		def add_optimizer_op(self):
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),self.config.max_norm)
			optimizer = tf.train.AdamOptimizer(self.config.lr)
			self.train_op = optimizer.apply_gradients(zip(grads, tvars))


		def add_decoder_and_wer_op(self):

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



