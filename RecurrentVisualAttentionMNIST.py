import tensorflow as tf 
import numpy as np
import math
from models import Config, Model

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class RecurrentVisualAttentionConfig(Config):

    def __init__(self):
        self.batch_size = 64
        self.lr = 1e-3
        self.lr_min = 1e-4
        self.l2_lambda = 0.000001
        self.num_epochs = 50
        self.max_norm = 5.
        
        self.num_classes = 10
        self.features_shape = (28, 28, 1)
        self.location_size = 2
        
        self.targets_shape = (1,)
        self.init_loc_size = (2,)
        self.variance = 0.2
        
        self.batch_size = 64
        self.iter_per_epoch = 10000
        
        self.patch_size = 5
        self.num_patches = 2
        self.seq_len = 6
        
        self.glimpse_h_size = 128
        self.glimpse_hl_size = 256
        
        self.hidden_size = 256
        self.rnn_num_layers = 1


class RecurrentVisualAttention(Model):

    def __init__(self, features_shape, num_classes, seq_len, cell_type='lstm', reuse=False, add_bn=False,
                 add_reg=False, scope="RVA"):
        self.config = RecurrentVisualAttentionConfig()
        self.config.features_shape = features_shape
        self.config.num_classes = num_classes
        self.reuse = reuse
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,)+ self.config.features_shape ))
        self.init_loc = tf.placeholder(tf.float32, shape=tuple((None,)+ self.config.init_loc_size))
        self.targets_placeholder = tf.placeholder(tf.float32, shape=tuple((None,) + self.config.targets_shape))
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
            
        self.mean_loc = []
        self.sampled_loc = []
        self.baselines = []


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
        hg = tf.contrib.layers.fully_connected(inputs=glimpse_input, 
                        num_outputs=self.config.glimpse_h_size, 
                        activation_fn=tf.nn.relu, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=self.reuse,scope="glimpse_hidden",trainable=True)
        hl = tf.contrib.layers.fully_connected(inputs=loc,
                        num_outputs=self.config.glimpse_h_size, 
                        activation_fn=tf.nn.relu, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=self.reuse,scope="loc_hidden",trainable=True)
        hg1 = tf.contrib.layers.fully_connected(inputs=hg, 
                        num_outputs=self.config.glimpse_hl_size, 
                        activation_fn=None, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=self.reuse,scope="glimpse_hidden_linear",trainable=True)
        hl1 = tf.contrib.layers.fully_connected(inputs=hl, 
                        num_outputs=self.config.glimpse_hl_size, 
                        activation_fn=None, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=self.reuse,scope="loc_hidden_linear",trainable=True)
        return tf.nn.relu(hg1 + hl1)
        
    def location_network(self, h):
        '''
        Gives mean location vector extracted from hidden state of RNN
        '''
        mean_loc = tf.contrib.layers.fully_connected(inputs=h, num_outputs=self.config.location_size, 
                        activation_fn=None, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                        reuse=True,scope="loc_network",trainable=True)
        mean_loc = tf.clip_by_value(mean_loc, -1., 1.)
        mean_loc = tf.stop_gradient(mean_loc)
        return mean_loc
                        
    def action_network(self, h):
        '''
        Gives mean location vector extracted from hidden state of RNN
        '''
        return tf.contrib.layers.fully_connected(inputs=h, num_outputs=self.config.num_classes, 
                        activation_fn=None, normalizer_fn=self.norm_fn, 
                        weights_initializer=XAVIER_INIT(uniform=True),
                        weights_regularizer=self.reg_fn, biases_initializer=None,
                        reuse=True,scope="act_network",trainable=True)
    
    def baseline_network(self, h):
        return tf.contrib.layers.fully_connected(inputs=h, num_outputs=1, 
                            activation_fn=None, normalizer_fn=self.norm_fn, 
                            weights_initializer=XAVIER_INIT(uniform=True),
                            weights_regularizer=self.reg_fn, biases_regularizer=self.reg_fn,
                            reuse=True,scope="baseline_network",trainable=True)
    
    def get_next_loc(self, loc):
        '''
        Sample from gaussian to get next glimpse location
        '''
        location_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=loc,
                                    diag_stdev=self.config.variance*tf.ones_like(loc))
        location_samples = location_dist.sample()
        location_samples = tf.clip_by_value(location_samples, -1., 1.)
        location_samples = tf.stop_gradient(location_samples)
        return location_samples
    
    def get_next_input(self, output, t):
        loc = self.location_network(output)
        next_loc = self.get_next_loc(loc)
        self.mean_loc.append(loc)
        self.sampled_loc.append(next_loc)
        glimpse = self.glimpse_network(self.inputs_placeholder[:, t, :], next_loc)
        return glimpse
        
    def build_model(self):
        cell = self.cell(num_units = self.config.hidden_size, state_is_tuple=True)
        init_state = cell.zero_state(tf.shape(self.inputs_placeholder)[0], tf.float32)
        
        self.mean_loc.append(self.init_loc)
        init_loc = self.get_next_loc(self.init_loc)
        self.sampled_loc.append(init_loc)
        
        init_glimpse = self.glimpse_network(self.inputs_placeholder[:, 0, :], init_loc)
        
        inputs = [init_glimpse] + [0] * (self.config.seq_len - 1)
        outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, init_state, cell, loop_function=self.get_next_input)
        self.logits = self.action_network(outputs[-1]) # tf.transpose(self.action_network(outputs), [1, 0, 2]) # bs, sl, 1 from paper, only look at last time step
        self.predicted_class = tf.nn.softmax(self.logits)
        self.baselines = tf.transpose(self.baseline_network(outputs), [1, 0, 2]) # bs, sl, 1
        
    def add_loss_op(self):
        # max_class = tf.transpose(tf.expand_dims(tf.arg_max(self.predicted_class, 2), axis=2), [1, 0, 2])
        max_class = tf.expand_dims(tf.arg_max(self.logits, 1), axis=1)
        true_labels = tf.cast(self.targets_placeholder, tf.int64)
        rewards = tf.cast(tf.equal(max_class, true_labels), tf.float32)
        tot_cum_rewards = rewards
        
        baseline_op = tf.stop_gradient(self.baselines)
        stable_rewards = tf.tile(tot_cum_rewards, (1, self.config.seq_len)) - tf.squeeze(baseline_op, axis=2)
        baseline_mse = tf.reduce_mean(tf.square((stable_rewards)))
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.squeeze(true_labels,axis=1)))
        
        ll = tf.contrib.distributions.Normal(tf.stack(self.mean_loc), self.config.variance).log_pdf(tf.stack(self.sampled_loc))
        ll = tf.transpose(tf.reduce_sum(ll, axis=2))
        reward_loss = tf.reduce_mean(ll*stable_rewards, axis=[0, 1])
        
        self.loss = -reward_loss + baseline_mse + self.cross_entropy
        self.total_rewards = tf.reduce_mean(tot_cum_rewards)
        
    def add_optimizer_op(self):
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        grads,_ = tf.clip_by_global_norm(grads, self.config.max_norm)                       
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def add_error_op(self):
        self.accuracy = tf.constant(0)

    def add_summary_op(self):
        self.summary_op = tf.summary.merge_all()

    def add_feed_dict(self, input_batch, target_batch, seq_len_batch , init_locations_batch):
        feed_dict = {self.inputs_placeholder:input_batch, self.targets_placeholder:target_batch,
                     self.init_loc:init_locations_batch, self.seq_len_placeholder:seq_len_batch}
        return feed_dict

    def train_one_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
        feed_dict = self.add_feed_dict(input_batch, target_batch, seq_len_batch , init_locations_batch)
        # Accuracy
        _, loss, rewards, cross_entropy, accuracy = session.run([self.train_op, self.loss, self.total_rewards, self.cross_entropy, self.accuracy], feed_dict)
        return None, loss, rewards, cross_entropy, accuracy

    def run_one_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
        summary, loss, rewards, cross_entropy, accuracy = self.train_one_batch(session, input_batch, target_batch, seq_len_batch , init_locations_batch)
        return summary, loss, rewards, cross_entropy, accuracy

import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5

def main():
    num_iters = 100000
    num_classes = 10
    seq_len = 6
    features_shape = (28, 28, 1)
    batch_size = 32
    img_size = 28
    channels = 1
    
    model = RecurrentVisualAttention(features_shape,
                                     num_classes,
                                     cell_type='lstm',
                                     seq_len=seq_len,
                                     reuse=True,
                                     add_bn=False,
                                     add_reg=False,
                                     scope="visual_attention")
    model.build_model()
    model.add_loss_op()
    model.add_error_op()
    model.add_optimizer_op()
    model.add_summary_op()
        
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    
    with tf.Session(config=GPU_CONFIG) as session:
        print "Inititialized TF Session!"

        # Make computational graph
        init_op = tf.global_variables_initializer()
        init_op.run()
        
        for i in xrange(num_iters):
            #print "Running epoch ({0})...".format(i)
			# Shuffle dataset on new epoch
            # random.shuffle(dataset)
            images, labels = mnist.train.next_batch(batch_size)
            data_batch = np.tile(images, seq_len)
            data_batch = np.reshape(data_batch, (batch_size, seq_len, img_size, img_size, channels))
            label_batch = np.expand_dims(labels, axis=1) # np.expand_dims(np.reshape(np.tile(labels, [seq_len]), (batch_size, seq_len)),axis=2)
            seq_lens_batch = seq_len * np.ones(len(images))
            init_loc = np.random.uniform(size=(len(images), 2), low=-1, high=1)

            summary, loss, rewards, cross_entropy, accuracy = model.run_one_batch(session, data_batch, label_batch, seq_lens_batch, init_loc)
            if i%1000 == 0:
                print("Loss of the current batch is {0}".format(loss))
                print("Finished iteration {0}/{1}".format(i,num_iters))
                print("Cumulative average rewards for batch: {0}".format(rewards))
                print("Cross entropy: {0}".format(cross_entropy))
                print("Accuracy per sequence per batch: {0}".format(accuracy))

if __name__ == "__main__":
    main()
