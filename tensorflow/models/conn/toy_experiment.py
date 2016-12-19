"""
Created 17 December 2016
@author: Frederick Heidrich


Toy experiment with LSTM and shared variables

Idea is to learn the connection through a board with a
LSTM RNN and shared variables

0.1. 19 Dec.
Splitting prediction and error into more verbose inference and loss + evaluation
	for statistical purposes.
Adding statistics to the runner.
Reuse weights and bias between reflect layers

todo:
add graph
save model for later evaluation
load model from previous training
LSTM grid instead of looped convolution layers

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# - - - - - - - - - - -
# Imports


from datetime import datetime
from time import time
from functools import wraps  # for the fancy decorators

# import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim  # convenience convolution

from grid_dataset import load_grid_dataset


# - - - - - - - - - - -
# Constants


GRID_SIZE = 9
BATCH_SIZE = 128
STATE_SIZE = 50


# - - - - - - - - - - -
# Decorators


def doublewrap(function):
	@wraps(function)
	def decorator(*args, **kwargs):
		if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
			return function(args[0])
		else:
			return lambda wrapee: function(wrapee, *args, **kwargs)
	return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
	attribute = '_cache_' + function.__name__
	name = scope or function.__name__

	@property
	@wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			with tf.variable_scope(name, *args, **kwargs):  # pylint: disable=undefined-variable
				setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator


# - - - - - - - - - - -
# Model


class Model:
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.loss
		self.inference
		self.optimize

	@define_scope
	def inference(self):
		X = self.X
		# y = self.y

		with slim.arg_scope([slim.conv2d],
			activation_fn=tf.nn.relu):

			num_reflect_layers = 9
			state_size = 10
			
			net = slim.conv2d(X, 10, [3, 3], scope='conv')
			net = slim.conv2d(net, 10, [1, 1], scope='reflect')
			for i in range(num_reflect_layers):
				net = slim.conv2d(net, 10, [1, 1], reuse=True, scope='reflect')
			# net = slim.repeat(net, reflect_layers, slim.conv2d, state_size, [1, 1], reuse=None, scope='reflect')  # Can't reuse since it generates scope name
			net = slim.conv2d(net, 1, [1, 1], scope='out_red')
			reshape = tf.reshape(net, [-1, GRID_SIZE * GRID_SIZE])
			net = slim.dropout(reshape, keep_prob=0.5, scope='dropout')

			weights_sm = tf.Variable(tf.zeros([GRID_SIZE * GRID_SIZE, 2]), name='output_weights')
			biases_sm = tf.Variable(tf.zeros([2]), name='output_biases')
			logits = tf.matmul(net, weights_sm) + biases_sm  # BUG

			# Verbose
			for v in tf.trainable_variables():
				print ('name = {}'.format(v.value()))

			return logits

		# pass  # work on this later
		# num_hidden = 30
		# cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
		# val, state = tf.nn.dynamic_rnn(cell, X)

		# # take the last of the output of the last layer
		# # transpose it and shift
		# val = tf.transpose(val, [1, 0, 2, 3])  # permutes batch_size with row
		# last = tf.gather(val, int(val.get_shape()[0]) - 1)

		# # apply a final transformation and map to output classes
		# weight = tf.Variable(tf.truncated_normal([num_hidden, int(y.get_shape()[1])]))
		# bias = tf.Variable(tf.constant(0.1, shape=[y.get_shape()[1]]))

		# logits = tf.nn.softmax(tf.matmul(last, weight) + bias)
		# return logits

	@define_scope
	def optimize(self):
		optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
		minimize = optimizer.minimize(self.loss)
		return minimize

	@define_scope
	def loss(self):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
			self.inference, self.y, name='cross_entropy')
		loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
		return loss

	@define_scope
	def evaluate(self):
		correct = tf.nn.in_top_k(self.inference, self.y, 1)
		return tf.reduce_mean(tf.cast(correct, tf.float32))


# - - - - - - - - - - - -
# Runner


def main(args=None):
	datasets = load_grid_dataset('/Users/fred/Developer/data/conn-batches-bin/grids_9x9_1000.hdf5')
	
	# Create placeholders
	# input dimensions are [batch size, rows, columns, depth]
	X_pl = tf.placeholder(tf.float32, shape=[None, GRID_SIZE, GRID_SIZE, 1], name='X')
	y_pl = tf.placeholder(tf.int32, shape=[None], name='y')  # 2 classes, either connect, or not, 0 or 1

	# with tf.Graph().as_default():
	model = Model(X_pl, y_pl)

	init_op = tf.global_variables_initializer()
	# summary_op = tf.summary.merge_all()

	batch_size = 128
	num_train_batches = int(datasets.train.num_examples / batch_size)
	epochs = 500  # for optimization

	with tf.Session() as sess:
		
		# train_writer = tf.train.summaryWriter('/Users/fred/Developer/data/conn-logs/train', graph=sess.graph)
		# test_writer = tf.train.summaryWriter('/Users/fred/Developer/data/conn-logs/test')

		sess.run(init_op)  # initialize

		start_time = time()
		for epoch in range(epochs):
			for batch in range(num_train_batches):
				X, y = datasets.train.next_batch(batch_size)
				_, loss, accuracy = sess.run([model.optimize, model.loss, model.evaluate], feed_dict={X_pl: X, y_pl: y})

			if epoch % 50 == 0:
				duration = time() - start_time
				epochs_per_sec = epoch / duration
				print('Epoch = {}, loss = {:.4f}, accuracy(training) = {:3.1f}%, duration = {:.2f}s, epochs/sec = {:.2f}'.format(
					str(epoch),
					loss,
					accuracy * 100,
					duration,
					epochs_per_sec))
				# summary = sess.run(summary_op, feed_dict={X_pl: X, y_pl: y})
				# train_writer.add_summary(summary, epoch)

			# evaluate a full epoch now and then
			if epoch % 500 == 0:
				accuracy_total = 0.0
				num_test_batches = int(datasets.test.num_examples / batch_size)
				for epoch_test in range(num_test_batches):
					X_test, y_test = datasets.test.next_batch(batch_size) 
					accuracy_total += sess.run(model.evaluate, feed_dict={X_pl: X_test, y_pl: y_test})
				if num_test_batches:
					accuracy_test = accuracy_total / num_test_batches
					print('Epoch {:2d} accuracy(test) = {:3.1f}%'.format(epoch, accuracy_test * 100))

		# run an evaluation of the whole test set (if possible)
		accuracy = sess.run(model.evaluate, feed_dict={X_pl: datasets.test.X, y_pl: datasets.test.y})
		print('Epoch {:2d} accuracy(test) {:3.1f}%'.format(epoch + 1, accuracy * 100))


if __name__ == "__main__":
	tf.app.run()
