import tensorflow as tf 
import numpy as np
import math
from models import Config, Model

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class RecurrentVisualAttentionConfig(Config):

    def __init__(self):
        self.batch_size = 64
        self.lr = 1e-3
        self.l2_lambda = 0.000001
        self.hidden_size = 256
        self.num_epochs = 50
        self.rnn_num_layers = 1
        self.num_classes = 4 # Mean vector of size 4
        self.features_shape = (100,100,3) #TO FIX!!!!
        self.targets_shape = (4,)
        self.init_loc_size = (4,)
        self.max_norm = 10
        self.keep_prob = 0.8
        self.variance = 1e-2
        self.num_samples = 5
        
        self.patch_size = 5
        self.glimpse_out_shape = 128
        self.num_patches = 3


class RecurrentVisualAttention(Model):

    def __init__(self, features_shape, num_classes, seq_len, cell_type='lstm', reuse=False, add_bn=False,
                 add_reg=False, scope="RVA"):
        self.config = RecurrentVisualAttentionConfig()
        self.config.features_shape = features_shape
        self.config.num_classes = num_classes
        self.reuse = reuse
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,)+ self.config.features_shape ))
        self.init_loc = tf.placeholder(tf.float32, shape=tuple((None,)+ self.config.init_loc_size))
        self.targets_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,) + self.config.targets_shape))
        self.config.seq_len = seq_len
        self.seq_len_placeholder = tf.placeholder(tf.int32, shape=tuple((None,) ))

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


    def glimpse_sensor(self, inputs, prev_loc):
        '''
        glimpse sensor extract patches of varying resolutions
        '''
        patch_size = self.config.patch_size
        max_width = self.config.patch_size * (2**(self.config.num_patches - 1))
        glimpse_imgs = []
        
        for k in range(self.config.num_patches):
            glimpse_center = prev_loc[:, 0:2]
            glimpse_img = tf.image.extract_glimpse(inputs, [self.config.patch_size, self.config.patch_size], glimpse_center)
            glimpse_img = tf.image.resize_bilinear(glimpse_img, [max_width, max_width])
            patch_size *= 2
            glimpse_imgs.append(glimpse_img)
        
        glimpse_imgs = tf.stack(glimpse_imgs, axis=1)
        return tf.contrib.layers.flatten(glimpse_imgs)
        
    def glimpse_network(self, inputs, loc):
        '''
        Glimpse network takes in location, gets glimpses and produces glimpse
        representation from location and glimpse patches.
        '''
        glimpse_input = self.glimpse_sensor(inputs, loc)
        hg = tf.contrib.layers.fully_connected(inputs=glimpse_input, num_outputs=128, 
                        activation_fn=tf.nn.relu, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=self.reuse,scope="glimpse_hidden",trainable=True)
        hl = tf.contrib.layers.fully_connected(inputs=loc, num_outputs=128, 
                        activation_fn=tf.nn.relu, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=self.reuse,scope="loc_hidden",trainable=True)
        hg1 = tf.contrib.layers.fully_connected(inputs=hg, num_outputs=256, 
                        activation_fn=None, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=self.reuse,scope="glimpse_hidden_linear",trainable=True)
        hl1 = tf.contrib.layers.fully_connected(inputs=hl, num_outputs=256, 
                        activation_fn=None, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=self.reuse,scope="loc_hidden_linear",trainable=True)
        return tf.nn.relu(hg1 + hl1)
        
    def location_network(self, h):
        '''
        Gives mean location vector extracted from hidden state of RNN
        '''
        #return tf.contrib.layers.fully_connected(inputs=h, num_outputs=4, 
        #                activation_fn=None, normalizer_fn=self.norm_fn, 
        #                weights_initializer=XAVIER_INIT(uniform=True),
        #                weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
        #                reuse=self.reuse,scope="loc_network",trainable=True)
        mean_loc = tf.contrib.layers.fully_connected(inputs=h, num_outputs=4, 
                        activation_fn=None, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=True,scope="loc_network",trainable=True)
        mean_loc = tf.clip_by_value(mean_loc, 0., 1.)
        #mean_loc = tf.stop_gradient(mean_loc)
        return mean_loc
        
    
    def get_next_loc(self, loc):
        '''
        Sample from gaussian to get next glimpse location
        '''
        #location_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=loc,
        #                            diag_stdev=self.config.variance*tf.ones_like(loc))
        #location_samples = location_dist.sample()
        #return location_samples
        
        location_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=loc,
                                    diag_stdev=self.config.variance*tf.ones_like(loc))
        location_samples = location_dist.sample()
        location_samples = tf.clip_by_value(location_samples, 0., 1.)
        #location_samples = tf.stop_gradient(location_samples)
        return location_samples
    
    def get_next_input(self, output, t):
        loc = self.location_network(output)
        next_loc = self.get_next_loc(loc)
        glimpse = self.glimpse_network(self.inputs_placeholder[:, t, :], next_loc)
        return glimpse
        
    def build_model(self):
        
        cell = self.cell(num_units = self.config.hidden_size)
        init_state = cell.zero_state(tf.shape(self.inputs_placeholder)[0], tf.float32)
        
        init_glimpse = self.glimpse_network(self.inputs_placeholder[:, 0, :], self.init_loc)
        
        inputs = [init_glimpse] + [0] * (self.config.seq_len - 1)
        outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, init_state, cell, loop_function=self.get_next_input)
        self.logits = tf.transpose(self.location_network(outputs), [1, 0, 2])

    def add_loss_op(self):
        logits_shape = tf.shape(self.logits)
        logits_flat = tf.reshape(self.logits, [-1])
        location_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=logits_flat,
                                    diag_stdev=self.config.variance*tf.ones_like(logits_flat))
        location_samples = location_dist.sample([self.config.num_samples])

        new_logits_shape = tf.concat([[self.config.num_samples,] , logits_shape], axis=0)
        location_samples = tf.reshape(location_samples, new_logits_shape)

        rewards_orig = -tf.reduce_mean(tf.abs(location_samples - tf.cast(self.targets_placeholder,tf.float32)),axis=3,keep_dims=True) - \
                    tf.reduce_max(tf.abs(location_samples - tf.cast(self.targets_placeholder,tf.float32)), axis=3,keep_dims=True)
     
        p_left = location_samples[:, :, :, 1]
        g_left = self.targets_placeholder[:, :, 1]
        left = tf.maximum(p_left, g_left)
        p_right = location_samples[:, :, :, 1] + location_samples[:, :, :, 3]
        g_right = self.targets_placeholder[:, :, 1] + self.targets_placeholder[:, :, 3]
        right = tf.minimum(p_right, g_right)
        p_top = location_samples[:, :, :, 0]
        g_top = self.targets_placeholder[:, :, 0]
        top = tf.maximum(p_top, g_top)
        p_bottom = location_samples[:, :, :, 0] + location_samples[:, :, :, 2]
        g_bottom = self.targets_placeholder[:, :, 0] + self.targets_placeholder[:, :, 2]
        bottom = tf.minimum(p_bottom, g_bottom)
        intersection = tf.maximum((right - left), 0) * tf.maximum((bottom - top), 0)
        p_area = location_samples[:, :, :, 3] * location_samples[:, :, :, 2]
        g_area = self.targets_placeholder[:, :, 3] * self.targets_placeholder[:, :, 2]
        union = p_area + g_area - intersection

        rewards_miou = intersection / union
        rewards_miou = tf.expand_dims(rewards_miou, axis=-1)

        # Edit this!
        rewards = rewards_orig

        timestep_rewards = tf.reduce_mean(rewards, axis=0, keep_dims=True)
        self.timestep_rewards = timestep_rewards

        tot_cum_rewards = tf.tile(tf.reduce_sum(rewards, axis=2, keep_dims = True),multiples=[1,1,self.config.seq_len, 1])
        self.tot_cum_rewards = tot_cum_rewards

        timestep_rewards_grad_op = tf.stop_gradient(timestep_rewards)
        rewards_grad_op = tf.stop_gradient(rewards)
        location_samples_op = tf.stop_gradient(location_samples)
        tot_cum_rewards_op = tf.stop_gradient(tot_cum_rewards)


        #const1 = 1.0 / (np.sqrt(2.0 * math.pi) * self.config.variance)
        #const2 = 2.0 * self.config.variance**2
        #squared_diff = tf.square(location_samples_op - self.logits)

        #density_func = tf.log(const1 * tf.exp(-squared_diff / const2))
        # self.loss = 1/self.config.variance*tf.reduce_mean(tf.reduce_sum((location_samples - self.logits)*(rewards_grad_op - timestep_rewards_grad_op),
        #                                     axis=1),axis=0)
        
        loglikelihood = tf.contrib.distributions.Normal(self.logits, self.config.variance).log_pdf(location_samples_op)
        
        self.loss = tf.reduce_mean(tf.reduce_sum(loglikelihood*(tot_cum_rewards_op - timestep_rewards_grad_op), axis=2),
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
        feed_dict = self.add_feed_dict(input_batch, target_batch, init_locations_batch)
        # Accuracy
        _, loss, rewards, area_accuracy = session.run([self.loss, self.total_rewards, self.area_accuracy], feed_dict)
        return None, loss, rewards, area_accuracy


    def run_one_batch(self, args, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
        if args.train == 'train':
            summary, loss, rewards, area_accuracy = self.train_one_batch(session, input_batch, target_batch, seq_len_batch , init_locations_batch)
        else:
            summary, loss, rewards, area_accuracy = self.test_one_batch(session, input_batch, target_batch, seq_len_batch , init_locations_batch)
        return summary, loss, rewards, area_accuracy


    def get_config(self):
        return self.config