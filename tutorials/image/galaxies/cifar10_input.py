# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '/Users/kaylan1/tensorflow/galaxies/data_dir',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('train_dir', '/Users/kaylan1/tensorflow/galaxies/train_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_epochs', 5, """""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# Global constants describing the CIFAR-10 data set.
tf.app.flags.DEFINE_integer('IMAGE_SIZE', 200,"""""")
tf.app.flags.DEFINE_integer('NUM_CLASSES', 5,"""""")
tf.app.flags.DEFINE_integer('NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN', 10,"""""")
tf.app.flags.DEFINE_integer('NUM_EXAMPLES_PER_EPOCH_FOR_EVAL', 100,"""""")
# Constants describing the training process.
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.9999,"""""") # The decay to use for the moving average.
tf.app.flags.DEFINE_float('NUM_EPOCHS_PER_DECAY', 350.0,"""""") # Epochs after which learning rate decays. 
tf.app.flags.DEFINE_float('LEARNING_RATE_DECAY_FACTOR', 0.1,"""""") # Learning rate decay factor.
tf.app.flags.DEFINE_float('INITIAL_LEARNING_RATE', 0.1,"""""") # Initial learning rate
# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
tf.app.flags.DEFINE_string('TOWER_NAME', 'tower',"""""")

FLAGS = tf.app.flags.FLAGS



# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 200
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2620
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200

def read_and_decode(filename_queue):
  '''Reading a TFRecord created by converting from another data format
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
  '''
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.float32)
  raise ValueError
  #image.set_shape([IMAGE_PIXELS])

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label

def inputs(train=None, batch_size=None, num_epochs=None,\
		   use_train=True,use_ivar=False):
	"""Reads input data num_epochs times.

	Args:
	train: Selects between the training (True) and validation (False) data.
	batch_size: Number of examples per returned batch.
	num_epochs: Number of times to read the input data, or 0/None to
	   train forever.

	Returns:
	A tuple (images, labels), where:
	* images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
	* labels is an int32 tensor with shape [batch_size] with the true label,
	  a number in the range [0, mnist.NUM_CLASSES).
	Note that an tf.train.QueueRunner is added to the graph, which
	must be run using e.g. tf.train.start_queue_runners().
	"""
	if not num_epochs: num_epochs = None
	if use_train:
	  if use_ivar:
		  filenames = [os.path.join(FLAGS.data_dir, 'train_3k_ivar.tfrecords')]
	  else:
		  filenames = [os.path.join(FLAGS.data_dir, 'train_3k_noivar.tfrecords')]
	else:
		if use_ivar:
			filenames = [os.path.join(FLAGS.data_dir, 'val_3k_ivar.tfrecords')]
		else:
			filenames = [os.path.join(FLAGS.data_dir, 'val_3k_noivar.tfrecords')]
	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(filenames,\
														num_epochs=num_epochs)
		# Even when reading in multiple threads, share the filename
		# queue.
		image, label = read_and_decode(filename_queue)
		raise ValueError
		# Shuffle the examples and collect them into batch_size batches.
		# (Internally uses a RandomShuffleQueue.)
		# We run this in two threads to avoid being a bottleneck.
		images, sparse_labels = tf.train.shuffle_batch(
			[image, label], batch_size=batch_size, num_threads=2,
			capacity=500 + 3 * batch_size,
			# Ensures a minimum amount of shuffling of examples.
			min_after_dequeue=500)

	return images, sparse_labels



