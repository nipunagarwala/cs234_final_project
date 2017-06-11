import tensorflow as tf
import numpy as np
import os
from models import Config, Model
from RecurrentCNN import *
from Seq2SeqCritic import *
from Pretrained import *
import math

XAVIER_INIT = tf.contrib.layers.xavier_initializer


class RecurrentCNNActor(Pretrained):

	def __init__(self, *args, **kwargs):

		super(Pretrained, self).__init__(*args, **kwargs)
		self.qvalues_placeholder = tf.placeholder(tf.float32, shape=tuple((None,None,) + self.config.targets_shape))

	def get_outputs(self):
		return self.logits

	def get_prob(self):
		return self.density_func

	def add_loss_op(self, loss_type='negative_l1_dist', pretrain=False):
		self.loss_type = loss_type
		logits_shape = tf.shape(self.logits)
		logits_flat = tf.reshape(self.logits, [-1])
		location_dist = tf.contrib.distributions.MultivariateNormalDiag(mu=logits_flat,
									diag_stdev=self.config.variance*tf.ones_like(logits_flat))
		location_samples = location_dist.sample([self.config.num_samples])

		new_logits_shape = tf.concat([[self.config.num_samples,] , logits_shape], axis=0)
		location_samples = tf.reshape(location_samples, new_logits_shape)
		self.location_samples = location_samples

		# print self.location_samples.get_shape().as_list()

		if pretrain:
			if self.loss_type == 'negative_l1_dist':
				rewards = -tf.reduce_mean(tf.abs(self.location_samples - tf.cast(self.targets_placeholder,tf.float32)),axis=3,keep_dims=True) - \
						tf.reduce_max(tf.abs(self.location_samples - tf.cast(self.targets_placeholder,tf.float32)), axis=3,keep_dims=True)
			elif self.loss_type == 'iou':
				rewards = self.get_iou_loss()
				rewards = tf.expand_dims(rewards,axis=-1)
		else:
			rewards = self.qvalues_placeholder

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

	def add_actor_feed_dict(self, input_batch, target_batch, seq_len_batch , init_locations_batch, qvalues_batch):
		feed_dict = {self.inputs_placeholder:input_batch, self.targets_placeholder:target_batch,
						self.init_loc:init_locations_batch, self.seq_len_placeholder:seq_len_batch,
						self.qvalues_placeholder: qvalues_batch}
		return feed_dict


	def train_one_actor_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch, qvalues_batch):
		feed_dict = self.add_actor_feed_dict(input_batch, target_batch, seq_len_batch , init_locations_batch, qvalues_batch)

		_, loss, density_func, total_rewards, area_accuracy = session.run([
				self.train_op,
				self.loss,
				self.density_func,
				self.total_rewards,
				self.area_accuracy],
				feed_dict)
		print("Density Func: {0}".format(density_func))


		return None, loss, total_rewards, area_accuracy


	def test_one_actor_batch(self, session, input_batch, target_batch, seq_len_batch , init_locations_batch, qvalues_batch):
		feed_dict = self.add_actor_feed_dict(input_batch, target_batch, init_locations, qvalues_batchs)
		# Accuracy
		loss, rewards, area_accuracy = session.run([self.loss, self.total_rewards, self.area_accuracy], feed_dict)

		return None, loss, rewards, area_accuracy


	def run_one_actor_batch(self, args, session, input_batch, target_batch, seq_len_batch , init_locations_batch):
		if args.train == 'train':
			summary, loss, rewards, area_accuracy = self.train_one_batch(session, input_batch, target_batch, seq_len_batch , 
											init_locations_batch, qvalues_batch)
		else:
			summary, loss, rewards, area_accuracy = self.test_one_batch(session, input_batch, target_batch, seq_len_batch, 
											init_locations_batch, qvalues_batch)
		return summary, loss, rewards, area_accuracy



class ActorCritic(object):

	def __init__(self, actor, critic, actor_target,critic_target):
		self.actor = actor
		self.critic = critic
		self.actor_target = actor_target
		self.critic_target = critic_target


	def build_pretrain_actor(self,loss_type='negative_l1_dist', pretrain=False):
		self.actor.add_loss_op(loss_type, pretrain)
		self.actor.add_summary_op()

	def load_yolo(self, session):
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
                            model_path='/data/yolo/YOLO_small.ckpt',
                            var_list=self.actor.variables_to_restore)
		init_fn(session)
		init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
		                    model_path='/data/yolo/YOLO_small.ckpt',
		                    var_list=self.actor_target.variables_to_restore)
		init_fn(session)

	# def build_pretrain_critic(self):


	def run_pretrain_actor_batch(self, args, session, input_batch, target_batch, seq_len_batch, 
									init_locations_batch):
		return self.actor.run_one_batch(args, session, input_batch, target_batch, seq_len_batch,
									init_locations_batch)

	def run_pretrain_critic_batch(self, args, session, input_batch, target_batch, seq_len_batch, init_locations_batch,
								 num_encode_batch, num_decode_batch):
		actor_feed_dict = self.actor.add_feed_dict(input_batch, target_batch, seq_len_batch, 
										init_locations_batch)
		actor_out = session.run([self.actor.get_outputs()], feed_dict=actor_feed_dict)
		return self.critic.run_one_batch(args, session, actor_out, target_batch, seq_len_batch, 
								num_encode_batch, num_decode_batch)


	def run_one_batch(self,args, session, input_batch, target_batch, seq_len_batch, 
									init_locations_batch):
		target_actor_feed_dict = self.actor_target.add_feed_dict(input_batch, target_batch, seq_len_batch, 
														init_locations_batch)
		target_actor_out = session.run([self.actor_target.get_outputs()], feed_dict=target_actor_feed_dict)

		target_critic_feed_dict = self.actor_target.add_feed_dict(input_batch, target_batch, seq_len_batch, 
														init_locations_batch)
		critic_outputs = session.run([self.critic_target.get_outputs()], feed_dict=target_critic_feed_dict)
		






