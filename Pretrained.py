from models import Config, Model
from RecurrentCNN import RecurrentCNN
from MOTRecurrentCNN import MOTRecurrentCNN
import tensorflow as tf
import numpy as np
import os
from yolo import YOLONet
import math


slim = tf.contrib.slim
XAVIER_INIT = tf.contrib.layers.xavier_initializer


class Pretrained(RecurrentCNN):
	def build_cnn(self, cur_inputs, reuse=False, scope=None):
		with tf.variable_scope(scope) as sc:
			yolo = YOLONet(cur_inputs)
			encoded_layer_name = '/'.join([self.scope, scope, 'yolo', 'conv_30'])
			encoded_layer = yolo.end_points[encoded_layer_name]

			# Per YOLO network architecture - these layers do not have variables
			encoded_layer_T = tf.transpose(encoded_layer, [0, 3, 1, 2], name='trans_31')
			encoded_layer_flat = slim.flatten(encoded_layer_T, scope='flat_32')

		self.variables_to_restore = {}
		for variable in slim.get_variables(sc):
			layer_suffix = variable.name.split("_")[-1]
			layer_num = int(layer_suffix.split("/")[0])
			if layer_num > 32:
				continue
			original_string = '/'.join(variable.name.split("/")[-3:])
			ckpt_string = original_string[:-2]
			print variable.name
			print ckpt_string
			self.variables_to_restore[ckpt_string] = variable

		return encoded_layer_flat


	def add_optimizer_op(self):
		fc_full_scope = '/'.join([self.scope, self.fc_scope])
		rnn_full_scope = '/'.join([self.scope, self.rnn_scope])
		fc_vars = tf.contrib.framework.get_variables(fc_full_scope)
		rnn_vars = tf.contrib.framework.get_variables(rnn_full_scope)
		var_list = fc_vars + rnn_vars
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_op = optimizer.minimize(self.loss, var_list=var_list)


class MOTPretrained(MOTRecurrentCNN):
	def build_cnn(self, cur_inputs, reuse=False, scope=None):
		with tf.variable_scope(scope) as sc:
			yolo = YOLONet(cur_inputs)
			encoded_layer_name = '/'.join([self.scope, scope, 'yolo', 'conv_30'])
			encoded_layer = yolo.end_points[encoded_layer_name]

			# Per YOLO network architecture - these layers do not have variables
			encoded_layer_T = tf.transpose(encoded_layer, [0, 3, 1, 2], name='trans_31')
			encoded_layer_flat = slim.flatten(encoded_layer_T, scope='flat_32')

		self.variables_to_restore = {}
		for variable in slim.get_variables(sc):
			layer_suffix = variable.name.split("_")[-1]
			layer_num = int(layer_suffix.split("/")[0])
			if layer_num > 32:
				continue
			original_string = '/'.join(variable.name.split("/")[-3:])
			ckpt_string = original_string[:-2]
			print variable.name
			print ckpt_string
			self.variables_to_restore[ckpt_string] = variable

		return encoded_layer_flat


	def add_optimizer_op(self):
		fc_full_scope = '/'.join([self.scope, self.fc_scope])
		rnn_full_scope = '/'.join([self.scope, self.rnn_scope])
		fc_vars = tf.contrib.framework.get_variables(fc_full_scope)
		rnn_vars = tf.contrib.framework.get_variables(rnn_full_scope)
		var_list = fc_vars + rnn_vars
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_op = optimizer.minimize(self.loss, var_list=var_list)
