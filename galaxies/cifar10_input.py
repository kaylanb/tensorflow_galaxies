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
tf.app.flags.DEFINE_integer('IMAGE_DEPTH', 3,"""""")
tf.app.flags.DEFINE_integer('NUM_CLASSES', 5,"""""")
tf.app.flags.DEFINE_integer('NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN', 2602,"""""")
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

#########
data_initializer = tf.placeholder(dtype=training_data.dtype,
                                    shape=training_data.shape)
  label_initializer = tf.placeholder(dtype=training_labels.dtype,
                                     shape=training_labels.shape)
  input_data = tf.Variable(data_initializer, trainable=False, collections=[])
  input_labels = tf.Variable(label_initializer, trainable=False, collections=[])
  ...
  sess.run(input_data.initializer,
           feed_dict={data_initializer: training_data})
  sess.run(input_labels.initializer,
           feed_dict={label_initializer: training_labels})



def read_and_decode(filename_queue):
  '''Reading a TFRecord created by converting from another data format
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
  '''
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
	  features={'label': tf.FixedLenFeature([], tf.int64),\
				'image': tf.FixedLenFeature([], tf.string)}
	)
	#'height': tf.FixedLenFeature([], tf.int64),\
	#'width': tf.FixedLenFeature([], tf.int64),\
	#'depth': tf.FixedLenFeature([], tf.int64),\

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  #image = tf.decode_raw(features['image'], tf.float32)
  #image.set_shape([FLAGS.IMAGE_SIZE,FLAGS.IMAGE_SIZE,3])
  #image_bytes= 4 * FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * FLAGS.IMAGE_DEPTH 
  image_bytes= FLAGS.IMAGE_SIZE * FLAGS.IMAGE_SIZE * FLAGS.IMAGE_DEPTH 
  record = tf.reshape(tf.decode_raw(features['image'], tf.float32), [image_bytes])
  shap= [FLAGS.IMAGE_SIZE,FLAGS.IMAGE_SIZE,FLAGS.IMAGE_DEPTH]
  image = tf.cast( tf.reshape(record,shap), tf.float32)

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int8)

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
		# Creates a QueueRunner
		filename_queue = tf.train.string_input_producer(filenames) #num_epochs=num_epochs)
		# Even when reading in multiple threads, share the filename
		# queue.
		image, label = read_and_decode(filename_queue)

		# Ensure that the random shuffling has good mixing properties.
		min_fraction_of_examples_in_queue = 0.4
		min_queue_examples = int(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
							     min_fraction_of_examples_in_queue)
		print ('Filling queue with %d CIFAR images before starting to train. '
			 'This will take a few minutes.' % min_queue_examples)

		# Shuffle the examples and collect them into batch_size batches.
		# (Internally uses a RandomShuffleQueue.)
		# We run this in two threads to avoid being a bottleneck.
		images, sparse_labels = tf.train.shuffle_batch(
			[image, label], batch_size=batch_size, num_threads=2,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)

	return images, sparse_labels



