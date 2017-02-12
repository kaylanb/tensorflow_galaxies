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

class TrainingExamples(object):
	def __init__(self):
		self.data_dir= '/Users/kaylan1/tensorflow/galaxies/data_dir'
		self.num_bands= 3 #grz
		self.image_size= 64 

class HyperParams(TrainingExamples):
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


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



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
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
  # MNIST:
  #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  #    labels=labels, logits=logits, name='xentropy')
  #return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(FLAGS.INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  FLAGS.LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      FLAGS.MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
  # MNIST simpler verions
  ## Add a scalar summary for the snapshot loss.
  #tf.summary.scalar('loss', loss)
  ## Create the gradient descent optimizer with the given learning rate.
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  ## Create a variable to track the global step.
  #global_step = tf.Variable(0, name='global_step', trainable=False)
  ## Use the optimizer to apply the gradients that minimize the loss
  ## (and also increment the global step counter) as a single training step.
  #train_op = optimizer.minimize(loss, global_step=global_step)
  #return train_op


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
  # Simplified version of MNIST
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  #correct = tf.nn.in_top_k(logits, labels, 1)
  correct = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  # Return the number of true entries.
  # I think this is an average over the batches?
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
  train_obj= TrainingExamples()
  fobj= h5py.File(os.path.join(train_obj.data_dir,'training_gathered_all.hdf5'),'r')
  # FIX ME!! need to ensure background ids match star,qso ids...
  # FIX ME: Add in variances!
  x_ np.concatenate( (fobj['elg'] + fobj['back'],\ # instances (2500) x n_rotations(none) x 64 x 64 x 3
					  fobj['star'] + fobj['back'],\
					  fobj['qso'] + fobj['back'],\
					  fobj['back']),axis=0)
  d= dict(star=0,qso=1,elg=2,back=3)
  y_= np.concatenate( (np.zeros(fobj['elg'].shape[0]).astype(np.int32)+d['elg'],\ # instances
					   np.zeros(fobj['star'].shape[0]).astype(np.int32)+d['star'],\
					   np.zeros(fobj['qso'].shape[0]).astype(np.int32)+d['qso'],\
					   np.zeros(fobj['back'].shape[0]).astype(np.int32)+d['back']),axis=0)
  # Shuffle
 
  #
  raise ValueError 
  x_=  
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
    loss = loss(logits, y)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, global_step)
    
    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder) #using sparse_softmax_cross_entropy_with_logits

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
          _, loss_value, eval_value, xnp_ = sess.run([train_op, loss, eval_correct,x_],
																										 feed_dict=feed_dict)

      duration = time.time() - start_time

if __name__ == '__main__':
	main()    

