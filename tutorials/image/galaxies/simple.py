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
import h5py
import os
import numpy as np
from argparse import ArgumentParser
from glob import glob
import re
import time
import tensorflow as tf

TOWER_NAME = 'tower'

class DataProperties(object):
	def __init__(self,debug=False):
		if debug: 
			self.data_fn= 'training_gathered_small.hdf5'
			self.n_val= 10 #validation size
			self.n_test= 10 #validation size
			self.batch= 20 # batch size
		else:
			self.data_fn= 'training_gathered_1000.hdf5'
			self.n_val= 500 
			self.n_test= 500 
			self.batch= 128 
		self.data_dir= '/Users/kaylan1/tensorflow/galaxies/data_dir'
		self.train_dir= '/Users/kaylan1/tensorflow/galaxies/train_dir'
		self.eval_dir= '/Users/kaylan1/tensorflow/galaxies/eval_dir'
		self.num_bands= 3 #grz
		self.image_size= 64 
		self.num_classes= 4 
		self.epochs= 10 
		for dr in [self.data_dir,self.train_dir,self.eval_dir]:
			if not os.path.exists(dr):
				os.makedirs(dr)


class HyperParams(object):
	def __init__(self,num_bands=None,image_size=None):
		assert(not num_bands is None)
		assert(not image_size is None)
		self.learnrate=0.1 # cifar10
		self.conv1= self.conv1_dict(num_bands=num_bands)
		self.conv2= self.conv2_dict()
		self.lrn= self.local_response_norm()
		self.pool_strides= self.max_pool_strides()
		self.fully_conn= self.fully_connected_layers(image_size=image_size)

	def conv1_dict(self,num_bands=None):
		return dict(filter_size= 5,
								in_channels= num_bands,
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

	def fully_connected_layers(self,image_size=None):
	# Cifar10 ex set layer size to pool2 feat map size * num_feat maps
		# We did two max_pools...
		first= int( image_size / self.pool_strides[1]**2 * self.conv2['out_channels'] )
		# Cifar10 did half this for second layer
		second= first/2
		assert(second % 2 == 0)
		return dict(first=first,\
					second=second)



class DataSet(object):
	def __init__(self,debug=False):
		self.info= DataProperties(debug=debug)
		self.hyper= HyperParams(num_bands= self.info.num_bands,\
								image_size= self.info.image_size)
		# Store data in memory
		print('Reading all data')
		self.read_all_data()

	def read_all_data(self):
		# FIX ME! val and test are batch_size sized samples
		fobj= h5py.File(os.path.join(self.info.data_dir,self.info.data_fn),'r')
		# FIX ME!! need to ensure background ids match star,qso ids...
		# FIX ME: Add in variances!
		# FIX ME! invvar --> var - mean() and setting bad pixels some high << 2^32 value
		# FIX ME! could just use bad pixel map for now
		# instances (2500) x n_rotations(none) x 64 x 64 x 3 
		self.x_= np.concatenate( (fobj['elg'][:] + fobj['back'][:],\
								  fobj['star'][:] + fobj['back'][:],\
								  fobj['qso'][:] + fobj['back'][:],\
								  fobj['back'][:]),axis=0)
		d= dict(star=0,qso=1,elg=2,back=3)
		self.y_= np.concatenate( (np.zeros(fobj['elg'].shape[0]).astype(np.int32)+d['elg'],\
								  np.zeros(fobj['star'].shape[0]).astype(np.int32)+d['star'],\
								  np.zeros(fobj['qso'].shape[0]).astype(np.int32)+d['qso'],\
								  np.zeros(fobj['back'].shape[0]).astype(np.int32)+d['back']),axis=0)
		#q_ = type of night #instances integer, make only 1 type night for now
		#b_= backgrounds # type x b_instances x 6 x 64 x 64
		# Shuffle, split train vs. val
		ind= np.arange(self.x_.shape[0]).astype(int)
		np.random.shuffle(ind)
		# Val
		sz= self.info.batch
		self.val_x_ = self.x_[ ind[:sz],...]
		self.val_y_ = self.y_[ ind[:sz],...]
		# Test
		self.test_x_ = self.x_[ ind[sz : 2*sz],...]
		self.test_y_ = self.y_[ ind[sz : 2*sz],...]
		# Train
		self.x_= self.x_[ ind[2*sz : ],...]
		self.y_= self.y_[ ind[2*sz : ],...]
		assert(self.x_.shape[0] > self.val_x_.shape[0])
	


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

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  #tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
  #tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity',
  #                                     tf.nn.zero_fraction(x))
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                    tf.nn.zero_fraction(x))


class _LoggerHook(tf.train.SessionRunHook):
	"""
	Use with cifar10 tf.train.MonitoredTrainingSession()
	Logs loss and runtime.
	"""

	def begin(self):
		self._step = -1

	def before_run(self, run_context):
		self._step += 1
		self._start_time = time.time()
		return tf.train.SessionRunArgs(loss)  # Asks for loss value.

	def after_run(self, run_context, run_values):
		duration = time.time() - self._start_time
		loss_value = run_values.results
		if self._step % 10 == 0:
			num_examples_per_step = FLAGS.batch_size
			examples_per_sec = num_examples_per_step / duration
			sec_per_batch = float(duration)

			format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
						'sec/batch)')
			print (format_str % (datetime.now(), self._step, loss_value,
							   examples_per_sec, sec_per_batch))



def inference(images, hyper=None,info=None):
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
                                                hyper.conv1['in_channels'],hyper.conv1['out_channels']],\
                                         stddev=5e-2,\
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
												hyper.conv2['in_channels'],hyper.conv2['out_channels']],\
                                         stddev=5e-2,\
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
    reshape = tf.reshape(pool2, [info.batch, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, hyper.fully_conn['first'] ],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [ hyper.fully_conn['first'] ], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[hyper.fully_conn['first'], \
                                                            hyper.fully_conn['second'] ],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [ hyper.fully_conn['second'] ], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [hyper.fully_conn['second'], info.num_classes],
                                          stddev=1/float(hyper.fully_conn['second']), wd=0.0)
    biases = _variable_on_cpu('biases', [info.num_classes],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear




def cross_entropy_loss(logits, labels):
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
  return tf.add_n(tf.get_collection('losses'), name='loss')
  # MNIST:
  #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  #    labels=labels, logits=logits, name='xentropy')
  #return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, global_step, learnrate=None):
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
  # Simple verion
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learnrate)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  
  return train_op

  # Variables that affect learning rate.
  #num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  #decay_steps = int(num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY)

  ## Decay the learning rate exponentially based on the number of steps.
  #lr = tf.train.exponential_decay(FLAGS.INITIAL_LEARNING_RATE,
  #                                global_step,
  #                                decay_steps,
  #                                FLAGS.LEARNING_RATE_DECAY_FACTOR,
  #                                staircase=True)
  #tf.summary.scalar('learning_rate', lr)

  ## Generate moving averages of all losses and associated summaries.
  #loss_averages_op = _add_loss_summaries(total_loss)

  ## Compute gradients.
  #with tf.control_dependencies([loss_averages_op]):
  #  opt = tf.train.GradientDescentOptimizer(lr)
  #  grads = opt.compute_gradients(total_loss)

  ## Apply gradients.
  #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  ## Add histograms for trainable variables.
  #for var in tf.trainable_variables():
  #  tf.summary.histogram(var.op.name, var)

  ## Add histograms for gradients.
  #for grad, var in grads:
  #  if grad is not None:
  #    tf.summary.histogram(var.op.name + '/gradients', grad)

  ## Track the moving averages of all trainable variables.
  #variable_averages = tf.train.ExponentialMovingAverage(
  #    FLAGS.MOVING_AVERAGE_DECAY, global_step)
  #variables_averages_op = variable_averages.apply(tf.trainable_variables())

  #with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
  #  train_op = tf.no_op(name='train')

  #return train_op

def accuracy_on_valdata(logits, labels):
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
  correct_bool = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  # Return the number of true entries.
  # I think this is an average over the batches?
  return tf.reduce_sum(tf.cast(correct_bool, tf.int32))
  #return tf.div(num_corr, tf.cast(correct_bool.get_shape()[0], tf.float32) )


def get_samples(x_,y_, batch_size=None): 
                #b_,#type x b_instances x 6 x 64 x 64
                #q_,
                #batch_size=128):
    '''
    x_ : instances x 6 x 64 x 64
    y_ : instances
    '''
    n_batches= int(x_.shape[0]/batch_size)
    for i in range(n_batches):
        ind = np.random.randint(0,x_.shape[0], batch_size)
        #bind = np.random.randint(0,b_.shape[0], batch_size)
		# variance of stamp is sqrt(stamp^2) if want to use that
        xb_ = x_[ind]
        yb_ = y_[ind]        
        #ind_types = q_[ind]
        #bb_ = b_[ind_types,bind,...]
        #if np.random.rand() > .5: 
        #    xb_ = xb[:,:,::-1,:] #indecies are wrong!
        yield i+1,xb_, yb_ #, bb_

class TextWriter(object):
	def __init__(self,train_dir=None):
		self.train_dir=train_dir
		fns= glob(os.path.join(self.train_dir,'epoch*loss.txt'))
		if len(fns) > 0:
			for fn in fns:
				os.remove(fn)

	def write(self,loss,epoch=None,step=None):
		fn= os.path.join(self.train_dir,'epoch_%d_loss.txt' % epoch)
		if not os.path.exists(fn):
			with open(fn,'a+') as foo:
				foo.write('#step loss\n')	
		with open(fn,'a+') as foo:
			foo.write('%d %.2f\n' % (step,loss))

def main(args=None):
	# Data, hyper params, etc
	data= DataSet(debug=args.debug)
	# Log myown stuff
	kjb_writer= TextWriter(train_dir= data.info.train_dir)
	with tf.Graph().as_default():
		print('Building Graph')
		# Generate placeholders for the images and labels.
		x = tf.placeholder(tf.float32, shape=[data.info.batch, 64, 64, 3])
		y = tf.placeholder(tf.int32, shape=[data.info.batch])
		#b = placeholder(tf.float32, shape=[10, None])
		#x = x + b

		global_step = tf.contrib.framework.get_or_create_global_step()

		# CNN Infrastructure
		logits = inference(x, hyper=data.hyper,info=data.info)
								 #100,
								 #100, 10)

		# Add to the Graph the Ops for loss calculation.
		loss= cross_entropy_loss(logits, y)
		tf.summary.scalar('my_loss', loss)

		# Add to the Graph the Ops that calculate and apply gradients.
		train_op = training(loss, global_step, learnrate=data.hyper.learnrate)
		tf.summary.scalar('my_learnrate', data.hyper.learnrate)

		# Add the Op to compare the logits to the labels during evaluation.
		#using sparse_softmax_cross_entropy_with_logits
		accuracy = accuracy_on_valdata(logits, y) 
		tf.summary.scalar('my_accuracty', accuracy)
		
		# Add histograms for trainable variables.
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)

		# Summar tensor
		summary = tf.summary.merge_all()
		# Saver tensor for training stuff
		saver = tf.train.Saver()
	
		# Add the variable initializer Op.
		init = tf.global_variables_initializer()

		## Launch Session with Verbose Outputs
		#step=0
		#with tf.train.MonitoredTrainingSession(\
		#		checkpoint_dir=FLAGS.train_dir,\
		#		hooks=[tf.train.StopAtStepHook(last_step=hp.epochs),\
		#			   tf.train.NanTensorHook(loss),\
		#			   _LoggerHook()],\
		#		config=tf.ConfigProto(\
		#			   log_device_placement=FLAGS.log_device_placement),\
		#		) as mon_sess:
		#	while not mon_sess.should_stop():
		#		print('Epoch: %d/%d' % (step+1,hp.epochs))
		#		mon_sess.run(init)
		#		mon_sess.run(train_op)
		# Begin
		print('Beginning Session')
		sess = tf.Session()
		
		# Write checkpoint stuff
		summary_writer = tf.summary.FileWriter(data.info.train_dir,sess.graph)
		print('Writing events to %s' % data.info.train_dir)
		
		sess.run(init)
		# Epochs
		for epoch in np.arange(data.info.epochs):
			print('Epoch: %d/%d' % (epoch+1,data.info.epochs))
			start_time = time.time()
			# Batches
			for step,xb_,yb_, in get_samples(data.x_, data.y_, batch_size=data.info.batch):
				print('Batch: %d/%d' % (step,int(data.x_.shape[0]/data.info.batch)))
				batch_time = time.time()
				# Train step.  The return values are the activations
				# vars after "train_op" are tensors to return for inspection
				feed_dict={x:xb_, y:yb_}
				_,kjb_loss= sess.run([train_op,loss], feed_dict=feed_dict)
				kjb_writer.write(kjb_loss,epoch=epoch,step=step)
				#raise ValueError
				#_, myloss, myx = sess.run([train_op, loss, x_],\
				#                           feed_dict={x:xb_, y:yb_})
				if step % 10 == 0:
					# Update the events file.
					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, step)
					summary_writer.flush()
					# Stats
					corr= sess.run(accuracy,feed_dict=feed_dict)
					duration= time.time() - batch_time
					print("Training: step %d (%.3f sec), Correct %d/%d" % \
							(step, duration,corr,data.info.batch))
			duration = time.time() - start_time
			# Save a checkpoint and evaluate the model every epoch
			#if step % 1000 == 0:
			checkpoint_file = os.path.join(data.info.eval_dir, 'model.ckpt')
			saver.save(sess, checkpoint_file, global_step=step)
			# Validation, Test
			corr= sess.run(accuracy,feed_dict={x:data.val_x_, y:data.val_y_})
			duration= time.time() - batch_time
			print("Validation: epoch %d (%.3f sec), Correct %d/%d" % \
					(epoch+1, duration,corr,data.info.batch))
			corr= sess.run(accuracy,feed_dict={x:data.test_x_, y:data.test_y_})
			duration= time.time() - batch_time
			print("Test: epoch %d (%.3f sec), Correct %d/%d" % \
					(epoch+1, duration,corr,data.info.batch))


if __name__ == '__main__':
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug', action="store_true", 
                        help='Load fraction of training set')
    args = parser.parse_args() 

    main(args=args)    

