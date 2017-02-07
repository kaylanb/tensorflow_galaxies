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

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import h5py
import numpy as np

import tensorflow as tf

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("data_dir", "/Users/kaylan1/tensorflow/galaxies/data_dir", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("val_size", 200, "Index of task within the job")
tf.app.flags.DEFINE_integer("test_size", 200, "Index of task within the job")
tf.app.flags.DEFINE_integer("train_size", 2620, "Index of task within the job")
tf.app.flags.DEFINE_boolean("use_ivar", False, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, savename='test'):
	"""Converts a dataset to tfrecords."""
	filename = os.path.join(FLAGS.data_dir, savename + '.tfrecords')
	#if not os.path.exists(filename):
	if os.path.exists(filename):
		# Nsample x 200 x 200 x 3 numpy array of type float32
		images = data_set.images
		# Nsample numpy array of type uint8
		labels = data_set.labels
		num_examples = data_set.num_examples

		if images.shape[0] != num_examples:
			raise ValueError('Images size %d does not match label size %d.' %
											 (images.shape[0], num_examples))
		rows = images.shape[1]
		cols = images.shape[2]
		depth = images.shape[3]
		raise ValueError
		writer = tf.python_io.TFRecordWriter(filename)
		for index in range(num_examples):
			image_raw = images[index,...].tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
					'height': _int64_feature(rows),
					'width': _int64_feature(cols),
					'depth': _int64_feature(depth),
					'label': _int64_feature(int(labels[index])),
					'image_raw': _bytes_feature(image_raw)}))
			writer.write(example.SerializeToString())
		writer.close()
		print('Wrote', filename)
	else:
		print('Already exists',filename)


def add_invvar(stamp_ivar,img_ivar):
	'''stamp_ivar, img_ivar -- galsim Image objects'''
	# Use img_ivar when stamp_ivar == 0, both otherwise
	use_img_ivar= np.ones(img_ivar.shape).astype(bool)
	use_img_ivar[ stamp_ivar > 0 ] = False
	# Compute using both
	ivar= np.power(stamp_ivar.copy(), -1) + np.power(img_ivar.copy(), -1)
	ivar= np.power(ivar,-1)
	keep= np.ones(ivar.shape).astype(bool)
	keep[ (stamp_ivar > 0)*\
				(img_ivar > 0) ] = False
	ivar[keep] = 0.
	# Now use img_ivar only where need to
	ivar[ use_img_ivar ] = img_ivar.copy()[ use_img_ivar ]
	return ivar

class BuildTraining(object):
	"""
	Takes hdf5 file created by GatherTraining and saves as single N-Dim array
	that will be converted to tensorflow format 
	""" 
	def __init__(self,data_dir=None):
		assert(not data_dir is None)
		self.data_dir= data_dir
		self.f= h5py.File(os.path.join(self.data_dir,'training_3k.hdf5'),'r')
		# map classification to integer
		labs= ['elg','lrg','qso','star','back']
		self.cats={}
		for lab,val in zip(labs,range(len(labs))):
			self.cats[lab]= val

	def create_srcpimage(self,use_ivar=False):
		if use_ivar:
			savefn= os.path.join(self.data_dir,'training_3k_ivar.hdf5')
		else:	
			savefn= os.path.join(self.data_dir,'training_3k_noivar.hdf5')
		fobj= h5py.File(savefn,'a')
		node= '/srcpimg'
		n_samples= len(self.f.keys()) * 5
		if not node in fobj:
			if use_ivar:	
				data= np.zeros((n_samples,200,200,6)).astype(np.float32)+np.nan
			else:
				data= np.zeros((n_samples,200,200,3)).astype(np.float32)+np.nan
			#label= np.zeros(n_samples).astype(np.string_) # Fixed ASCII for hdf5 
			label= np.zeros(n_samples).astype(np.int8)-1 
			cnt=-1
			for id in list(self.f.keys()):
				for obj in ['elg','lrg','qso','star','back']:
					cnt += 1
					if cnt % 100 == 0:
						print('Storing image %d/%d' % (cnt,n_samples))
					label[cnt]= self.classToInt(obj=obj)
					if obj == 'back':
						data[cnt,...]= np.array(self.f['/%s/%s' % (id,obj)])
					else:
						data[cnt,...]= np.array(self.f['/%s/%s' % (id,obj)]) + np.array(self.f['/%s/%s' % (id,'back')])
					if use_ivar:
						ivar= add_invvar(src_ivar,back_ivar)
			assert( np.where( np.isfinite(data).flatten() == False )[0].size == 0)
			assert( np.where( label < 0 )[0].size == 0)
			dset = fobj.create_dataset('%s/images' % node, data=data,chunks=True)
			dset = fobj.create_dataset('%s/labels' % node, data=label,chunks=True)
			#dset = fobj.create_dataset('%s/labels' % node, data=label,dtype="S4",chunks=True)
		print('srcpimage_noivar has been created')
		return fobj

	def get_srcpimage(self,use_ivar=False):
		fobj= self.create_srcpimage(use_ivar=use_ivar)
		return fobj['/srcpimg/images'],fobj['/srcpimg/labels']

	def classToInt(self,obj=None):
		assert(obj in self.cats.keys())
		return self.cats[obj]


def main(unused_argv):
	builder= BuildTraining(data_dir=FLAGS.data_dir)
	images,labels= builder.get_srcpimage(use_ivar=FLAGS.use_ivar)

	# Shuffle samples
	inds= np.arange(images.shape[0])
	np.random.shuffle(inds)
	keep={}
	keep['test']= inds[ :FLAGS.test_size ]
	keep['val']= inds[ FLAGS.test_size: FLAGS.test_size+FLAGS.val_size ]
	keep['train']= inds[ FLAGS.test_size+FLAGS.val_size: ]
	for key,sz in zip(['test','val','train'],[FLAGS.test_size,FLAGS.val_size,FLAGS.train_size]):
		assert(keep[key].size == sz)
	# Convert to TFRecords
	class DataSet(object):
		def __init__(self,images,labels,keep):
			self.images= np.array(images)[keep]
			self.labels= np.array(labels)[keep]
			self.num_examples = keep.size
	data={}
	for key in keep.keys():
		data[key]= DataSet(images,labels,keep[key])
		# Write as TFRecords.
		savename='%s_3k_noivar' % key
		convert_to(data[key], savename='%s_3k_noivar' % key)
	print('Done')


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])


