"""grid_utils wrapper to hold split and shuffle databases"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from grid_utils import load_dataset  # lazy version
import numpy as np

# from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.framework import dtypes
from collections import namedtuple


# collection of all datasets in one neat tuple
Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])
data = namedtuple('data', ['X', 'y'])  # convenience


class Dataset(object):

	def __init__(self,
				 X,
				 y,
				 dtype=dtypes.float32):
		"""Construct the dataset"""
		self._num_examples = X.shape[0]
		self._X = X
		self._y = y
		self._epochs_completed = 0
		self._index_in_epoch = 0

		# for convenience
		self.data = (X, y)

	@property
	def X(self):
		return self._X

	@property
	def y(self):
		return self._y

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		self._epochs_completed

	def next_batch(self, batch_size):
		"""Return the next batch_size examples from this dataset

		Args:
			batch_size

		Returns:
			grids, connections: Input data and connections
		"""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1

			# Shuffle data (with numpy, tensorflow can also do this?).
			perm = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._X = self._X[perm]
			self._y = self._y[perm]

			# Start next epoch.
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples

		end = self._index_in_epoch
		return self._X[start:end], self._y[start:end]


def read_data_set(train_dir,
				  reshape=True,
				  test_size=0.8,
				  validation_size=0.1):
	"""Process the saved dataset and create a dataset

	Returns:
		Datasets: A collection of train, validation and test sets
	"""
	# Load from h5py for now
	X, y = load_dataset(train_dir)

	# Convert to proper tensor format: [num, x, y, depth].
	# New X. Each row is a grid.
	X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
	X = X.astype(np.float32)  # for tensorflow

	# New y. label[i] is 1.0 iff grid[i] is connected
	y = (y < 9 * 9) * 1.0
	y = y.astype(np.int32)

	# Shuffle. Ugly version.
	perm = np.arange(len(X))
	np.random.shuffle(perm)
	X = X[perm]
	y = y[perm]

	# Split the data
	# Create test set from train set
	# default 0.8
	test_split = int(len(X) * test_size)
	train_X = X[0:test_split]
	train_y = y[0:test_split]
	test_X = X[test_split:]
	test_y = y[test_split:]

	# Create validation set from the train set.
	validation_split = int(len(train_X) * validation_size)
	validation_X = train_X[:validation_split]
	validation_y = train_y[:validation_split]
	train_X = train_X[validation_split:]
	train_y = train_y[validation_split:]

	# Create individual datasets.
	test = Dataset(test_X, test_y)
	validation = Dataset(validation_X, validation_y)
	train = Dataset(train_X, train_y)

	# Return collection of all datasets for easy access.
	return Datasets(test=test, validation=validation, train=train)


def load_grid_dataset(train_dir=''):
	return read_data_set(train_dir)
