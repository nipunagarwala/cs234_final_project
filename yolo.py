import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


class YOLONet(object):

    def __init__(self, inputs_placeholder, is_training=False): # Dont want to train, pretrained model

        # NOTE: These values are irrevelant since we stop at last conv layers
        #       (default valeus for yolo network)
        self.image_size = 448
        self.cell_size = 7
        self.boxes_per_cell = 2
        self.num_class = 21
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)

        self.alpha = 0.01
        self.inputs_placeholder = inputs_placeholder
        self.net, self.end_points = self.build_network(self.inputs_placeholder, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)


    def build_network(self,
                      inputs_placeholder,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=False,
                      scope='yolo'):
        with tf.variable_scope(scope) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                #activation_fn=leaky_relu(alpha),
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                outputs_collections=end_points_collection):
                net = tf.pad(inputs_placeholder, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob,
                                   is_training=is_training, scope='dropout_35')
                net = slim.fully_connected(net, num_outputs,
                                           activation_fn=None, scope='fc_36')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points

def leaky_relu(x, alpha=0.05):
    return tf.maximum(alpha*x, x, name='leaky_relu')
