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

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

class TrainingProps(object):
	def __init__(self):
		self.num_bands= 3 #grz
		self.image_size= 64 

class HyperParams(TrainingProps):
	def __init__(self):
		super(HyperParams,).__init__()
		self.conv1= self.conv1_dict()
		self.conv2= self.conv2_dict()
		self.lrn= self.local_response_norm()
		self.pool_strides= self.max_pool_strides()
		self.fully_conn= self.fully_connected_layers()

	def conv1_dict(self):
		return dict(filter_size= 5,
								in_channels= self.num_bands,
								out_channels= 64)

	def conv2_dict(self):
		return dict(filter_size= 5,
								in_channels= self.conv1['out_channels'],
								out_channels=64)

	def local_response_norm(self):
		return dict(depth_radius= 4, # default 5
								bias= 1., #default 1,
								alpha= 0.001 / 9.0, #default 1
					      beta= 0.75 #default 0.5
							 )

	def max_pool_strides(self):
		return [1,2,2,1]

	def fully_connected_layers(self):
    # Cifar10 ex set layer size to pool2 feat map size * num_feat maps
		# We did two max_pools...
    first= int( self.image_size / self.pool_strides[1]**2 * self.conv2['out_channels'] )
		# Cifar10 did half this for second layer
		second= first/2
		assert(second % 2 == 0)
		return dict(first=first,\
								second=second)

def inference(images, hyper=None):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().
    hyper: HyperParams() object

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[hyper.conv1['filter_size'],hyper.conv1['filter_size'], \
																								hyper.conv1['in_channels'],hyper.conv1['out_channels'],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [hyper.conv1['out_channels']], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides= hyper.pool_strides,
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.local_response_normalization(pool1, hyper.lrn['depth_radius'],\
												bias=hyper.lrn['bias'], alpha=hyper.lrn['alpha'], beta=hyper.lrn['beta'],\
												name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[hyper.conv2['filter_size'],hyper.conv2['filter_size'], \
																								hyper.conv2['in_channels'],hyper.conv2['out_channels'],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [hyper.conv2['out_channels']], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.local_response_normalization(conv2, hyper.lrn['depth_radius'], \
												bias=hyper.lrn['bias'], alpha=hyper.lrn['alpha'], beta=hyper.lrn['beta'],\
												name='norm2')
  # pool2
  # ksize -- window size for each dim of Tensor norm2 (batch,width,height,channels)
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=hyper.pool_strides, padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, hyper.fully_conn['first'] ],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [ hyper.fully_conn['first'] ], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[size_layer, hyper.fully_conn['second'] ],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [ hyper.fully_conn['second'] ], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [, FLAGS.NUM_CLASSES],
                                          stddev=1/float(hyper.fully_conn['second']), wd=0.0)
    biases = _variable_on_cpu('biases', [FLAGS.NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear




def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')



def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))



def get_samples(x_,#instances x 6 x 64 x 64
                y_,#instances 
                b_,#type x b_instances x 6 x 64 x 64
                q_,
                batch_size=128):

    for i in range(x_.shape[0]/batch_size):
        ind = np.random.randint(0,x_.shape[0], batch_size)
        bind = np.random.randint(0,b_.shape[0], batch_size)
		# variance of stamp is sqrt(stamp^2) if want to use that
        xt = x_[ind]
        yb_ = yt_[ind]        
        ind_types = q_[ind]
        bb_ = xt_[ind_types,bind,...]
        if np.random.rand() > .5: 
            xb_ = xb[:,:,::-1,:] #indecies are wrong!
        yield xb_, yb_, bb_

def main():
  #read your data
  # invvar --> var - mean() and setting bad pixels some high << 2^32 value
  # but could just use bad pixel map for now
  x_= galaxy images # instances x n_rotations x 6 x 64 x 64
  y_= galaxy/start/etc # instances
  q_ = type of night #instances integer, make only 1 type night for now
  b_= backgrounds # type x b_instances x 6 x 64 x 64

  with tf.Graph().as_default():
      # Generate placeholders for the images and labels.
    x = placeholder_inputs(shape=[10, None, 6, 64, 64], dtype=tf.float32)
    y = placeholder_inputs(shape=[10, None], dtype=tf.int32)
    b = placeholder_inputs(shape=[10, None], dtype=tf.int32)
    x = x + b
	  # config
    hp= HyperParams()
    
    global_step = tf.contrib.framework.get_or_create_global_step()
    
	#xall = variable(x_)
    #index = placeholder_inputs(shape=[None], dtype=int32)
    # Build a Graph that computes predictions from the inference model.
    logits = inference(x, hyper=hp)
                             #100,
                             #100, 10)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, y)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)
    
train_op = cifar10.train(loss, global_step)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder) #using sparse_softmax_cross_entropy_with_logits

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()


    # Create a session for running Ops on the Graph.
    sess = tf.Session()


    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(1000000):
        start_time = time.time()

      #for ind in range(total_samples/minibatch_size):
      #   feed_dict = {index:ind}
      for xb_,yb_, bb_ in get_samples(x_, y_):
          feed_dict = {x:xb_, y:yb_}
          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.
          _, loss_value, xnp_ = sess.run([train_op, loss, x_],
                                   feed_dict=feed_dict)

      duration = time.time() - start_time

if __name__ == '__main__':
	main()    

