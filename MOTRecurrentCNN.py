import tensorflow as tf
import numpy as np
import os
from models import Config, Model
import math

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class MOTRecurrentCNNConfig(Config):

	def __init__(self):
		self.batch_size = 64
		self.lr = 1e-3
		self.l2_lambda = 0.0000001
		self.hidden_size = 512
		self.num_epochs = 50
		self.num_layers = 3
		self.num_classes = 4 # Mean vector of size 4
		self.features_shape = (100,100,3) #TO FIX!!!!
		self.targets_shape = (4,)
		self.init_loc_size = (4,)
		self.max_norm = 10
		self.keep_prob = 0.8
		self.init_state_out_size = 128
		self.cnn_out_shape = 128
		self.variance = 3e-1
		self.num_samples = 5


class MOTRecurrentCNN(Model):

	def __init__(self, features_shape, num_classes, cell_type='lstm', seq_len=8, reuse=False, add_bn=False,
				add_reg=False, deeper = False, loss_type = 'negative_l1_dist', cum_sum=False, 
				init_loc_size=(4,), num_objects=1, scope='RCNN'):

		self.config = MOTRecurrentCNNConfig()
		self.config.features_shape = features_shape
		self.config.num_classes = num_classes
		self.config.init_loc_size = init_loc_size
		self.config.seq_len = seq_len
		self.reuse = reuse
		self.deeper = deeper
		self.loss_type = loss_type
		self.cumsum = cum_sum
		self.scope = scope
		self.num_objects = num_objects

		self.inputs_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,)+ self.config.features_shape ))
		self.init_loc = tf.placeholder(tf.float32, shape=tuple((None,self.num_objects)+ self.config.init_loc_size))
		self.targets_placeholder = tf.placeholder(tf.float32, shape=tuple((None,self.num_objects, None) + self.config.targets_shape))
		self.seq_len_placeholder = tf.placeholder(tf.int32, shape=tuple((None,) ))

		if add_bn:
			self.norm_fn = tf.contrib.layers.batch_norm
		else:
			self.norm_fn = None

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

	def conv_layer(self, inputs, outputs, kernel_size, stride, reuse, scope):
		return tf.contrib.layers.conv2d(inputs=inputs, num_outputs=outputs, kernel_size=kernel_size,
								stride=stride,padding='SAME',rate=1,activation_fn=tf.nn.relu,
								normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
								weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
								reuse = reuse, scope=scope, trainable=True)


	def build_cnn(self, cur_inputs, reuse=False, scope=None):
		with tf.variable_scope(scope):
			conv_out1 = self.conv_layer(cur_inputs, 32, [3,3], [1,1], reuse, 'conv1')
			conv_out2 = self.conv_layer(conv_out1, 32, [3,3], [1,1], reuse, 'conv2')

			max_pool1 = tf.contrib.layers.max_pool2d(inputs=conv_out2, kernel_size=[3,3],stride=[2,2],
								scope='maxpool1', padding='SAME')
			conv_out3 = self.conv_layer(max_pool1, 32, [3,3], [1,1], reuse, 'conv3')
			conv_out4 = self.conv_layer(conv_out3, 32, [3,3], [1,1], reuse, 'conv4')

			max_pool2 = tf.contrib.layers.max_pool2d(inputs=conv_out4, kernel_size=[3,3],stride=[2,2],
										scope='maxpool1',padding='SAME')
			flatten_out = tf.contrib.layers.flatten(max_pool2,scope='flatten')

			fc1 = tf.contrib.layers.fully_connected(inputs=flatten_out, num_outputs=self.config.cnn_out_shape,activation_fn=tf.nn.relu,
									normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
									weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
									reuse = reuse,scope='fc1',trainable=True)
			fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=self.config.cnn_out_shape,activation_fn=tf.nn.relu,
									normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
									weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
									reuse = reuse,scope='fc2',trainable=True)

		cnn_out = fc2
		return cnn_out

	def build_deeper_cnn(self, cur_inputs, reuse=False, scope=None):
		with tf.variable_scope(scope):
			conv_out1 = self.conv_layer(cur_inputs, 32, [3,3], [1,1], reuse, 'conv1')
			conv_out2 = self.conv_layer(conv_out1, 32, [3,3], [1,1], reuse, 'conv2')

			max_pool1 = tf.contrib.layers.max_pool2d(inputs=conv_out2, kernel_size=[3,3],stride=[1,1],
								scope='maxpool1', padding='SAME')
			conv_out3 = self.conv_layer(max_pool1, 32, [3,3], [1,1], reuse, 'conv3')
			conv_out4 = self.conv_layer(conv_out3, 32, [3,3], [1,1], reuse, 'conv4')

			max_pool2 = tf.contrib.layers.max_pool2d(inputs=conv_out4, kernel_size=[3,3],stride=[2,2],
										scope='maxpool2',padding='SAME')

			conv_out5 = self.conv_layer(max_pool2, 32, [3,3], [1,1], reuse, 'conv5')
			conv_out6 = self.conv_layer(conv_out5, 32, [3,3], [1,1], reuse, 'conv6')

			max_pool3 = tf.contrib.layers.max_pool2d(inputs=conv_out6, kernel_size=[3,3],stride=[1,1],
										scope='maxpool3',padding='SAME')

			conv_out7 = self.conv_layer(max_pool3, 32, [3,3], [1,1], reuse, 'conv7')
			conv_out8 = self.conv_layer(conv_out7, 32, [3,3], [1,1], reuse, 'conv8')

			max_pool4 = tf.contrib.layers.max_pool2d(inputs=conv_out8, kernel_size=[3,3],stride=[2,2],
										scope='maxpool4',padding='SAME')

			flatten_out = tf.contrib.layers.flatten(max_pool4,scope='flatten')

			fc1 = tf.contrib.layers.fully_connected(inputs=flatten_out, num_outputs=self.config.cnn_out_shape,activation_fn=tf.nn.relu,
									normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
									weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
									reuse = reuse,scope='fc1',trainable=True)
			fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=self.config.cnn_out_shape,activation_fn=tf.nn.relu,
									normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
									weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
									reuse = reuse,scope='fc2',trainable=True)

		cnn_out = fc2
		return cnn_out

	def build_rnn(self, rnn_inputs):
		# W = tf.get_variable("Weights", shape=[self.config.hidden_size, self.config.num_classes],
		# 					initializer=XAVIER_INIT(uniform=True))
		# b = tf.get_variable("Bias", shape=[self.config.num_classes])

		rnnNet = tf.contrib.rnn.MultiRNNCell([self.cell(num_units = self.config.hidden_size) for _ in
									range(self.config.num_layers)], state_is_tuple=True)
		(rnnNet_out, rnnNet_state) = tf.nn.dynamic_rnn(cell = rnnNet, inputs=rnn_inputs,
		                sequence_length=self.seq_len_placeholder,dtype=tf.float32)

		cur_shape = tf.shape(rnnNet_out)
		rnnOut_2d = tf.reshape(rnnNet_out, [-1, cur_shape[2]])

		# logits_2d = tf.matmul(rnnOut_2d, W) + b
		logits_objects = []

		for i in xrange(self.num_objects):
			W = tf.get_variable("Weights_"+ str(i), shape=[self.config.hidden_size, self.config.num_classes],
								initializer=XAVIER_INIT(uniform=True))
			b = tf.get_variable("Bias_" + str(i), shape=[self.config.num_classes])
			logits_2d = tf.matmul(rnnOut_2d, W) + b
			# logits_2d = tf.contrib.layers.fully_connected(inputs=rnnOut_2d, num_outputs=self.config.num_classes,
			# 							activation_fn=tf.nn.relu,
			# 							normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
			# 							weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
			# 							scope='logits_out_'+str(i),trainable=True)
			logits_objects.append(logits_2d)

		logits_objects = tf.stack(logits_objects, axis=1)

		rnn_out = tf.reshape(logits_objects,[cur_shape[0], self.num_objects, cur_shape[1], self.config.num_classes])

		return rnn_out

	def build_initial_state(self, loc_inputs, reuse=False, scope=None):
		with tf.variable_scope(scope):

			fc_combined_list = []

			for i in xrange(self.num_objects):
				should_reuse = reuse or (i > 0)
				fc1 = tf.contrib.layers.fully_connected(inputs=loc_inputs[:,i,:], num_outputs=self.config.init_state_out_size,
										activation_fn=tf.nn.relu,
										normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
										weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
										reuse = should_reuse, scope='fc1', trainable=True)

				fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=self.config.init_state_out_size,
										activation_fn=tf.nn.relu,
										normalizer_fn=self.norm_fn,	weights_initializer=XAVIER_INIT(uniform=True) ,
										weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
										reuse = should_reuse,scope='fc2',trainable=True)

				fc_combined_list.append(fc2)

		init_state_out = tf.concat(fc_combined_list, axis=1)
		print init_state_out.get_shape().as_list()
		return init_state_out



	def build_model(self):
		self.cnn_scope = 'CNN'
		self.fc_scope = 'FC'
		self.rnn_scope = 'RNN'

		obs_outputs = []
		reuse = False
		zero_vec = tf.zeros_like(self.init_loc)
		with tf.variable_scope(self.scope):
			for t in xrange(self.config.seq_len):
				print("Current iteration: {0}".format(t))
				x = tf.placeholder(tf.float32, shape=[None, self.config.init_state_out_size])
				st_state = tf.zeros_like(x)
				if t == 0:
					reuse = False
					st_state = self.build_initial_state(self.init_loc, reuse, self.fc_scope)
				if t > 0:
					# tf.get_variable_scope().reuse_variables()
					reuse = True
					st_state = self.build_initial_state(zero_vec , reuse, self.fc_scope)

				if not self.deeper:
					concat_result = tf.concat([self.build_cnn(self.inputs_placeholder[:,t,:,:,:], reuse, self.cnn_scope),st_state],
											 axis=1)
					obs_outputs.append(concat_result)
				else:
					concat_result = tf.concat([self.build_deeper_cnn(self.inputs_placeholder[:,t,:,:,:], reuse, self.cnn_scope),st_state],
											 axis=1)
					obs_outputs.append(concat_result)

			obs_outputs = tf.stack(obs_outputs, axis=1)

			rnn_output = None
			with tf.variable_scope(self.rnn_scope):
				rnn_output = self.build_rnn(obs_outputs)

			self.logits = tf.nn.sigmoid(rnn_output)
		# self.logits = rnn_output


	def get_iou_loss(self):
		p_left = self.location_samples[:, :, :, :, 1]
		g_left = self.targets_placeholder[:, :, :, 1]
		left = tf.maximum(p_left, g_left)
		p_right = self.location_samples[:, :, :, :, 1] + self.location_samples[:, :, :, :, 3]
		g_right = self.targets_placeholder[:, :, :, 1] + self.targets_placeholder[:, :, :, 3]
		right = tf.minimum(p_right, g_right)
		p_top = self.location_samples[:, :, :, :, 0]
		g_top = self.targets_placeholder[:, :, :, 0]
		top = tf.maximum(p_top, g_top)
		p_bottom = self.location_samples[:, :, :, :, 0] + self.location_samples[:, :, :, :, 2]
		g_bottom = self.targets_placeholder[:, :, :, 0] + self.targets_placeholder[:, :, :, 2]
		bottom = tf.minimum(p_bottom, g_bottom)
		intersection = tf.maximum((right - left), 0) * tf.maximum((bottom - top), 0)
		p_area = self.location_samples[:, :, :, :, 3] * self.location_samples[:, :, :, :, 2]
		g_area = self.targets_placeholder[:, :, :,3] * self.targets_placeholder[:, :, :, 2]
		union = p_area + g_area - intersection

		return intersection/union


	def add_loss_op(self, loss_type='negative_l1_dist'):
		self.loss_type = loss_type
		logits_shape = tf.shape(self.logits)
		logits_flat = tf.reshape(self.logits, [-1])
		location_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=logits_flat,
									diag_stdev=self.config.variance*tf.ones_like(logits_flat))
		location_samples = location_dist.sample([self.config.num_samples])

		new_logits_shape = tf.concat([[self.config.num_samples,] , logits_shape], axis=0)
		location_samples = tf.reshape(location_samples, new_logits_shape)
		self.location_samples = location_samples

		if self.loss_type == 'negative_l1_dist':
			rewards = -tf.reduce_mean(tf.abs(location_samples - tf.cast(self.targets_placeholder,tf.float32)),axis=4,keep_dims=True) - \
					tf.reduce_max(tf.abs(location_samples - tf.cast(self.targets_placeholder,tf.float32)), axis=4,keep_dims=True)
		elif self.loss_type == 'iou':
			rewards = self.get_iou_loss()
			rewards = tf.expand_dims(rewards,axis=-1)

		print location_samples.get_shape().as_list()
		print rewards.get_shape().as_list()
		timestep_rewards = tf.reduce_mean(rewards, axis=0, keep_dims=True)

		print timestep_rewards.get_shape().as_list()
		self.timestep_rewards = timestep_rewards

		if self.cumsum:
			tot_cum_rewards = tf.cumsum(rewards, axis=3, reverse=True)
		else:
			tot_cum_rewards = tf.tile(tf.reduce_sum(rewards, axis=3, keep_dims = True),multiples=[1,1,1, self.config.seq_len, 1])
		
		self.tot_cum_rewards = tot_cum_rewards

		timestep_rewards_grad_op = tf.stop_gradient(timestep_rewards)
		rewards_grad_op = tf.stop_gradient(rewards)
		location_samples_op = tf.stop_gradient(location_samples)
		tot_cum_rewards_op = tf.stop_gradient(tot_cum_rewards)


		const1 = 1.0 / (np.sqrt(2.0 * math.pi) * self.config.variance)
		const2 = 2.0 * self.config.variance**2
		squared_diff = tf.square(self.targets_placeholder - self.logits)

		density_func = tf.log(const1 * tf.exp(-squared_diff / const2))
		self.density_func = density_func

		self.loss = tf.reduce_mean(tf.reduce_sum(density_func*(tot_cum_rewards_op - timestep_rewards_grad_op), axis=3),
											axis=[1, 0])
		self.total_rewards = tf.reduce_mean(tf.reduce_sum(timestep_rewards, axis=3), axis=[2,1])
		tf.summary.scalar('Total Rewards', self.total_rewards[0][0])


	def add_optimizer_op(self):
		tvars = tf.trainable_variables()
		grads = tf.gradients(self.loss, tvars)
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars))



	def add_error_op(self):
		# VOT metrics (MOT only makes sense for multiple object)
 		# Accuracy:
		# intersection / union
 		# Robustness
 		# average count of number of resets (0 overlap in predicted and actual)

		# y, x, height, width
		# left = x
		# right = x + width
		# top = y
		# bottom = y + height

		p_left = self.logits[:, :, :, 1]
		g_left = self.targets_placeholder[:, :, :, 1]
		left = tf.maximum(p_left, g_left)
		self.left = left

		p_right = self.logits[:, :, :, 1] + self.logits[:, :, :, 3]
		self.p_right = p_right
		g_right = self.targets_placeholder[:, :, :, 1] + self.targets_placeholder[:, :, :, 3]
		self.g_right = g_right
		right = tf.minimum(p_right, g_right)
		self.right = right

		p_top = self.logits[:, :, :, 0]
		g_top = self.targets_placeholder[:, :, :, 0]
		top = tf.maximum(p_top, g_top)
		self.top = top

		p_bottom = self.logits[:, :, :, 0] + self.logits[:, :, :, 2]
		g_bottom = self.targets_placeholder[:, :, :, 0] + self.targets_placeholder[:, :, :, 2]
		bottom = tf.minimum(p_bottom, g_bottom)
		self.bottom = bottom

		intersection = tf.maximum((right - left), 0) * tf.maximum((bottom - top), 0)
		self.intersection = intersection
		p_area = self.logits[:, :, :, 3] * self.logits[:, :, :, 2]
		g_area = self.targets_placeholder[:, :, :, 3] * self.targets_placeholder[:, :, :, 2]
		union = p_area + g_area - intersection
		self.union = union

		self.area_accuracy = tf.reduce_mean(intersection / union)
		tf.summary.scalar('IOU Area Accuracy', self.area_accuracy)

		# Bounding box summaries
		p_seq_image_bboxes = []
		g_seq_image_bboxes = []
		for i in xrange(self.config.seq_len):
			p_left_i = self.logits[:, :, i, 1]
			g_left_i = self.targets_placeholder[:, :, i, 1]
			p_right_i = self.logits[:, :, i, 1] + self.logits[:, :, i, 3]
			g_right_i = self.targets_placeholder[:, :, i, 1] + self.targets_placeholder[:, :, i, 3]
			p_top_i = self.logits[:, :, i, 0]
			g_top_i = self.targets_placeholder[:, :, i, 0]
			p_bottom_i = self.logits[:, :, i, 0] + self.logits[:, :, i, 2]
			g_bottom_i = self.targets_placeholder[:, :, i, 0] + self.targets_placeholder[:, :, i, 2]

			p_top_i = tf.expand_dims(p_top_i, axis=-1)
			p_left_i = tf.expand_dims(p_left_i, axis=-1)
			p_bottom_i = tf.expand_dims(p_bottom_i, axis=-1)
			p_right_i = tf.expand_dims(p_right_i, axis=-1)

			g_top_i = tf.expand_dims(g_top_i, axis=-1)
			g_left_i = tf.expand_dims(g_left_i, axis=-1)
			g_bottom_i = tf.expand_dims(g_bottom_i, axis=-1)
			g_right_i = tf.expand_dims(g_right_i, axis=-1)

			p_bboxes_i = tf.concat([p_top_i, p_left_i, p_bottom_i, p_right_i], axis=-1)
			g_bboxes_i = tf.concat([g_top_i, g_left_i, g_bottom_i, g_right_i], axis=-1)
			print p_bboxes_i.get_shape().as_list()

			p_image_bboxes = tf.image.draw_bounding_boxes(self.inputs_placeholder[:, i, :, :, :], p_bboxes_i)
			g_image_bboxes = tf.image.draw_bounding_boxes(self.inputs_placeholder[:, i, :, :, :], g_bboxes_i)
			p_seq_image_bboxes.append(p_image_bboxes)
			g_seq_image_bboxes.append(g_image_bboxes)

		p_image_bboxes = tf.concat(p_seq_image_bboxes, axis=2)
		g_image_bboxes = tf.concat(g_seq_image_bboxes, axis=2)
		bbox_summary = tf.concat([p_image_bboxes, g_image_bboxes], axis=1)
		tf.summary.image('bounding boxes', bbox_summary, max_outputs=10)


	def add_summary_op(self):
		self.summary_op = tf.summary.merge_all()


	def add_feed_dict(self, input_batch, target_batch, seq_len_batch , init_locations_batch):
		feed_dict = {self.inputs_placeholder:input_batch, self.targets_placeholder:target_batch,
						self.init_loc:init_locations_batch, self.seq_len_placeholder:seq_len_batch}
		return feed_dict


	def train_one_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
		feed_dict = self.add_feed_dict(input_batch, target_batch, seq_len_batch , init_locations_batch)

		_, loss, summary, density_func, total_rewards, area_accuracy = session.run([
				self.train_op,
				self.loss,
				self.summary_op,
				self.density_func,
				self.total_rewards,
				self.area_accuracy],
				feed_dict)


		return summary, loss, total_rewards[0][0], area_accuracy


	def test_one_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
		feed_dict = self.add_feed_dict(input_batch, target_batch,seq_len_batch , init_locations_batch)
		# Accuracy
		loss, summary, rewards, area_accuracy = session.run([self.loss, self.summary_op,self.total_rewards, self.area_accuracy], feed_dict)

		return summary, loss, rewards, area_accuracy


	def run_one_batch(self, args, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
		if args.train == 'train':
			summary, loss, rewards, area_accuracy = self.train_one_batch(session, input_batch, target_batch, seq_len_batch , init_locations_batch)
		else:
			summary, loss, rewards, area_accuracy = self.test_one_batch(session, input_batch, target_batch, seq_len_batch , init_locations_batch)
		return summary, loss, rewards, area_accuracy


	def get_config(self):
		return self.config

	def add_update_weights_op(self, input_model, gamma):
		q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=input_model.scope)
		target_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

		update_ops = []
		for targ, orig in zip(target_q_vars, q_vars):
			new_targ = tf.assign(targ, gamma*orig + (1-gamma)*targ)
			update_ops.append(new_targ)

		self.update_target_op = tf.group(*update_ops)

	def update_weights(self, session):
		session.run(self.update_target_op)

