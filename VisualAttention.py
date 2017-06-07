import tensorflow as tf 
import numpy as np
import os
import math
from models import Config, Model

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class VisualAttentionConfig(Config):

    def __init__(self):
        self.batch_size = 64
        self.lr = 1e-3
        self.l2_lambda = 0.0000001
        self.hidden_size = 256
        self.num_epochs = 50
        self.rnn_num_layers = 1
        self.num_classes = 4 # Mean vector of size 4
        self.features_shape = (100,100,3) #TO FIX!!!!
        self.targets_shape = (4,)
        self.init_loc_size = (4,)
        
        self.max_norm = 10
        self.keep_prob = 0.8
        self.patch_size = 30
        self.glimpse_out_shape = 128
        
        self.variance = 1e-1
        self.num_samples = 5


class VisualAttention(Model):

    def __init__(self, features_shape, num_classes, seq_len, cell_type='lstm', reuse=False, add_bn=False,
                 add_reg=False, scope="VA"):
        self.config = VisualAttentionConfig()
        self.config.features_shape = features_shape
        self.config.num_classes = num_classes
        self.reuse = reuse
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,)+ self.config.features_shape ))
        self.init_loc = tf.placeholder(tf.float32, shape=tuple((None,)+ self.config.init_loc_size))
        self.targets_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,) + self.config.targets_shape))
        self.config.seq_len = seq_len
        self.seq_len_placeholder = tf.placeholder(tf.int32, shape=tuple((None,) ))
        self.emission_num_layers =  1

        self.loss_type = 'negative_l1_dist'
        self.cumsum = False
        
        self.scope = scope
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

    '''
    def classification_network(self, scope, class_rnn_inputs):
        W = tf.get_variable("Weights", shape=[self.config.hidden_size, self.config.num_classes],
							initializer=XAVIER_INIT(uniform=True))
        b = tf.get_variable("Bias", shape=[self.config.num_classes])

        rnnNet = tf.contrib.rnn.MultiRNNCell([self.cell(num_units = self.config.hidden_size) for _ in
									range(self.config.rnn_num_layers)], state_is_tuple=True)
        (rnnNet_out, rnnNet_state) = tf.nn.dynamic_rnn(cell = rnnNet, inputs=class_rnn_inputs,
		                sequence_length=self.seq_len_placeholder,dtype=tf.float32)

        cur_shape = tf.shape(rnnNet_out)
        rnnOut_2d = tf.reshape(rnnNet_out, [-1, cur_shape[2]])

        logits_2d = tf.matmul(rnnOut_2d, W) + b
        rnn_out = tf.reshape(logits_2d,[cur_shape[0], cur_shape[1], self.config.num_classes])

        return rnn_out
    
    def emission_network(self, scope, emit_rnn_inputs):
        W = tf.get_variable("Weights", shape=[self.config.hidden_size, self.config.num_classes],
							initializer=XAVIER_INIT(uniform=True))
        b = tf.get_variable("Bias", shape=[self.config.num_classes])

        rnnNet = tf.contrib.rnn.MultiRNNCell([self.cell(num_units = self.config.hidden_size) for _ in
									range(self.config.num_layers)], state_is_tuple=True)
        (rnnNet_out, rnnNet_state) = tf.nn.dynamic_rnn(cell = rnnNet, inputs=emit_rnn_inputs,
		                sequence_length=self.seq_len_placeholder,dtype=tf.float32)

        cur_shape = tf.shape(rnnNet_out)
        rnnOut_2d = tf.reshape(rnnNet_out, [-1, cur_shape[2]])

        logits_2d = tf.matmul(rnnOut_2d, W) + b
        rnn_out = tf.reshape(logits_2d,[cur_shape[0], cur_shape[1], self.config.num_classes])

        return rnn_out
    '''

    def context_network(self, scope):
        with tf.variable_scope(scope):
            conv_out1 = tf.contrib.layers.conv2d(inputs=self.inputs_placeholder[:,0,:,:,:] , num_outputs=32, kernel_size=[3,3],
                                 stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
                                 normalizer_fn=self.norm_fn,    weights_initializer=XAVIER_INIT(uniform=True),
                                 weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
                                 reuse = self.reuse, trainable=True)
            conv_out2 = tf.contrib.layers.conv2d(inputs=conv_out1, num_outputs=32, kernel_size=[3,3],
                                 stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
                                 normalizer_fn=self.norm_fn,    weights_initializer=XAVIER_INIT(uniform=True),
                                 weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
                                 reuse = self.reuse, trainable=True)
            conv_out3 = tf.contrib.layers.conv2d(inputs=conv_out2, num_outputs=32, kernel_size=[3,3],
                                 stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
                                 normalizer_fn=self.norm_fn,    weights_initializer=XAVIER_INIT(uniform=True),
                                 weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
                                 reuse = self.reuse, trainable=True)
            flatten_out = tf.contrib.layers.flatten(conv_out3)
            fc1 = tf.contrib.layers.fully_connected(inputs=flatten_out, num_outputs=self.config.hidden_size,activation_fn=tf.nn.relu,
                        normalizer_fn=self.norm_fn,    weights_initializer=XAVIER_INIT(uniform=True) ,
                        weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
                        reuse=self.reuse,scope='context_fc',trainable=True)
            return fc1

    def glimpse_image(self, scope, inputs):
        with tf.variable_scope(scope):
            conv_out1 = tf.contrib.layers.conv2d(inputs=inputs , num_outputs=32, kernel_size=[3,3],
                                                 stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
                                                 normalizer_fn=self.norm_fn,    weights_initializer=XAVIER_INIT(uniform=True) ,
                                                 weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
                                                 reuse = self.reuse, trainable=True)
            conv_out2 = tf.contrib.layers.conv2d(inputs=conv_out1, num_outputs=32, kernel_size=[3,3],
                                                 stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
                                                 normalizer_fn=self.norm_fn,    weights_initializer=XAVIER_INIT(uniform=True),
                                                 weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
                                                 reuse = self.reuse, trainable=True)
            conv_out3 = tf.contrib.layers.conv2d(inputs=conv_out2, num_outputs=32, kernel_size=[3,3],
                                                 stride=[1,1],padding='SAME',rate=1,activation_fn=tf.nn.relu,
                                                 normalizer_fn=self.norm_fn,    weights_initializer=XAVIER_INIT(uniform=True),
                                                 weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
                                                 reuse = self.reuse, trainable=True)
                                                 
            flatten_out = tf.contrib.layers.flatten(conv_out3)
            fc1 = tf.contrib.layers.fully_connected(inputs=flatten_out, num_outputs=self.config.glimpse_out_shape,activation_fn=tf.nn.relu,
                                    normalizer_fn=self.norm_fn,    weights_initializer=XAVIER_INIT(uniform=True) ,
                                    weights_regularizer=self.reg_fn , biases_regularizer=self.reg_fn ,
                                    reuse=self.reuse,scope='fc1',trainable=True)
            return fc1
         
    def glimpse_loc(self, scope, inputs):
        return tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=self.config.glimpse_out_shape, 
                                activation_fn=tf.nn.relu, normalizer_fn=self.norm_fn, 
                                weights_initializer=XAVIER_INIT(uniform=True),
                                weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                                reuse=self.reuse,scope=scope,trainable=True)
        
    def glimpse_network(self, scope, inputs, loc):
        x = tf.image.extract_glimpse(inputs, (self.config.patch_size, self.config.patch_size), loc[:, 0:2], normalized=True)
        gimage = self.glimpse_image(scope, x)
        gloc = self.glimpse_loc(scope, loc)
        return tf.multiply(gimage, gloc)

    def classification_network(self, scope, class_rnn_inputs):
        return tf.contrib.layers.fully_connected(inputs=class_rnn_inputs, num_outputs=10, 
                            activation_fn=None, normalizer_fn=self.norm_fn, 
                            weights_initializer=XAVIER_INIT(uniform=True),
                            weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                            reuse=True,scope="emission_network",trainable=True)
        
    def emission_network(self, scope, emit_rnn_inputs):
        return tf.contrib.layers.fully_connected(inputs=emit_rnn_inputs, num_outputs=self.config.num_classes, 
                            activation_fn=None, normalizer_fn=self.norm_fn, 
                            weights_initializer=XAVIER_INIT(uniform=True),
                            weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                            reuse=True,scope="emission_network",trainable=True)
        
    def build_model(self):
        loc = []
        init_glimpse = self.context_network(self.scope)
        '''
        emit_cell = self.cell(num_units = self.config.hidden_size)
        class_cell = self.cell(num_units = self.config.hidden_size)
        
        (classNet_out, classNet_state) = tf.nn.dynamic_rnn(cell = class_cell, inputs=rnn_inputs,
		                sequence_length=self.seq_len_placeholder,dtype=tf.float32)
        (emitNet_out, emitNet_state) = tf.nn.dynamic_rnn(cell = emit_cell, inputs=rnn_inputs,
		                sequence_length=self.seq_len_placeholder,dtype=tf.float32)
                        
        '''
        print tf.shape(init_glimpse)
        emit_cell = self.cell(num_units = self.config.hidden_size)
        class_cell = self.cell(num_units = self.config.hidden_size)
        
        state_emit = emit_cell.zero_state(tf.shape(self.inputs_placeholder)[0], tf.float32)
        with tf.variable_scope("emit_rnn", reuse=False):
            h_emit, state_emit = emit_cell(init_glimpse, state_emit)
            loc.append(self.emission_network(self.scope, h_emit))
            
        state_class = class_cell.zero_state(tf.shape(self.inputs_placeholder)[0], tf.float32)
        
        for t in xrange(self.config.seq_len):
            print("Current iteration: {0}".format(t))
            
            # Get a glimpse based on location
            glimpse = self.glimpse_network(self.scope, self.inputs_placeholder[:,t,:,:,:], loc[t])
            
            # Classification RNN takes in glimpse and previous state
            with tf.variable_scope("class_rnn", reuse=(t != 0)):
                h_class, state_class = class_cell(glimpse, state_class)
            # Emission RNN takes in hidden layer of classification and previous state
            with tf.variable_scope("emit_rnn", reuse=True):
                h_emit, state_emit = emit_cell(h_class, state_emit)
                loc.append(self.emission_network(self.scope, h_emit))
        
        locations = tf.transpose(tf.stack(loc[1:]), [1, 0, 2])       
        self.logits = tf.nn.sigmoid(locations)
        

    def get_iou_loss(self):
        p_left = self.location_samples[:, :, :, 1]
        g_left = self.targets_placeholder[:, :, 1]
        left = tf.maximum(p_left, g_left)
        p_right = self.location_samples[:, :, :, 1] + self.location_samples[:, :, :, 3]
        g_right = self.targets_placeholder[:, :, 1] + self.targets_placeholder[:, :, 3]
        right = tf.minimum(p_right, g_right)
        p_top = location_samples[:, :, :, 0]
        g_top = self.targets_placeholder[:, :, 0]
        top = tf.maximum(p_top, g_top)
        p_bottom = self.location_samples[:, :, :, 0] + self.location_samples[:, :, :, 2]
        g_bottom = self.targets_placeholder[:, :, 0] + self.targets_placeholder[:, :, 2]
        bottom = tf.minimum(p_bottom, g_bottom)
        intersection = tf.maximum((right - left), 0) * tf.maximum((bottom - top), 0)
        p_area = self.location_samples[:, :, :, 3] * self.location_samples[:, :, :, 2]
        g_area = self.targets_placeholder[:, :, 3] * self.targets_placeholder[:, :, 2]
        union = p_area + g_area - intersection

        return intersection/union       
 
    def add_loss_op(self):
        logits_shape = tf.shape(self.logits)
        logits_flat = tf.reshape(self.logits, [-1])
        location_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=logits_flat,
                                    diag_stdev=self.config.variance*tf.ones_like(logits_flat))
        location_samples = location_dist.sample([self.config.num_samples])
        self.location_samples = location_samples

        new_logits_shape = tf.concat([[self.config.num_samples,] , logits_shape], axis=0)
        location_samples = tf.reshape(location_samples, new_logits_shape)

        if self.loss_type == 'negative_l1_dist':
            rewards = -tf.reduce_mean(tf.abs(location_samples - tf.cast(self.targets_placeholder,tf.float32)),axis=3,keep_dims=True) - \
                       tf.reduce_max(tf.abs(location_samples - tf.cast(self.targets_placeholder,tf.float32)), axis=3,keep_dims=True)
        elif self.loss_type == 'iou':
            rewards = self.get_iou_loss()
            rewards = tf.expand_dims(rewards,axis=-1)

        timestep_rewards = tf.reduce_mean(rewards, axis=0, keep_dims=True)
        self.timestep_rewards = timestep_rewards

        if self.cumsum:
            tot_cum_rewards = tf.cumsum(rewards, axis=2, reverse=True)
        else:
            tot_cum_rewards = tf.tile(tf.reduce_sum(rewards, axis=2, keep_dims = True),multiples=[1,1,self.config.seq_len, 1])
		
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

        self.loss = tf.reduce_mean(tf.reduce_sum(density_func*(tot_cum_rewards_op - timestep_rewards_grad_op), axis=2),
											axis=[1, 0])
        self.total_rewards = tf.reduce_mean(tf.reduce_sum(timestep_rewards, axis=2), axis=1)

    def add_optimizer_op(self):
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def add_error_op(self):
		p_left = self.logits[:, :, 1]
		g_left = self.targets_placeholder[:, :, 1]
		left = tf.maximum(p_left, g_left)
		self.left = left

		p_right = self.logits[:, :, 1] + self.logits[:, :, 3]
		self.p_right = p_right
		g_right = self.targets_placeholder[:, :, 1] + self.targets_placeholder[:, :, 3]
		self.g_right = g_right
		right = tf.minimum(p_right, g_right)
		self.right = right

		p_top = self.logits[:, :, 0]
		g_top = self.targets_placeholder[:, :, 0]
		top = tf.maximum(p_top, g_top)
		self.top = top

		p_bottom = self.logits[:, :, 0] + self.logits[:, :, 2]
		g_bottom = self.targets_placeholder[:, :, 0] + self.targets_placeholder[:, :, 2]
		bottom = tf.minimum(p_bottom, g_bottom)
		self.bottom = bottom

		intersection = tf.maximum((right - left), 0) * tf.maximum((bottom - top), 0)
		self.intersection = intersection
		p_area = self.logits[:, :, 3] * self.logits[:, :, 2]
		g_area = self.targets_placeholder[:, :, 3] * self.targets_placeholder[:, :, 2]
		union = p_area + g_area - intersection
		self.union = union

		self.area_accuracy = tf.reduce_mean(intersection / union)

    def add_summary_op(self):
        self.summary_op = tf.summary.merge_all()

    def add_feed_dict(self, input_batch, target_batch, seq_len_batch , init_locations_batch):
        feed_dict = {self.inputs_placeholder:input_batch, self.targets_placeholder:target_batch,
                     self.init_loc:init_locations_batch, self.seq_len_placeholder:seq_len_batch}
        return feed_dict

    def train_one_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
        feed_dict = self.add_feed_dict(input_batch, target_batch, seq_len_batch , init_locations_batch)
        # Accuracy
        _, loss, rewards, area_accuracy = session.run([self.train_op, self.loss, self.total_rewards, self.area_accuracy], feed_dict)
        return None, loss, rewards, area_accuracy


    def test_one_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
        feed_dict = self.add_feed_dict(input_batch, target_batch, init_locations)
        # Accuracy
        loss, area_accuracy = session.run([self.loss, self.total_rewards, self.area_accuracy], feed_dict)
        return None, loss, rewards, area_accuracy


    def run_one_batch(self, args, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
        if args.train == 'train':
            summary, loss, rewards, area_accuracy = self.train_one_batch(session, input_batch, target_batch, seq_len_batch , init_locations_batch)
        else:
            summary, loss, rewards, area_accuracy = self.test_one_batch(session, input_batch, target_batch, seq_len_batch , init_locations_batch)
        return summary, loss, rewards, area_accuracy


    def get_config(self):
        return self.config
