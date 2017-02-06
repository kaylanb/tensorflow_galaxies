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

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

# Flags for defining the tf.train.Server
#tf.app.flags.DEFINE_string("directory", "hello", "One of 'ps', 'worker'")
#tf.app.flags.DEFINE_integer("validation_size", 100, "Index of task within the job")
#FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#def _float_feature(value):
#  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
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

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


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
        self.f= h5py.File(os.path.join(self.data_dir,'training_10k.hdf5'),'r')

    def create_srcpimage(self,use_ivar=False):
		if use_ivar:
			savefn= os.path.join(self.data_dir,'training_10k_ivar.hdf5')
		else:	
			savefn= os.path.join(self.data_dir,'training_10k_noivar.hdf5')
		fobj= h5py.File(savefn,'a')
		node= '/srcpimg'
		if not node in fobj:
			if use_ivar:	
				data= np.zeros(10000,200,200,3).astype(np.float32)+np.nan
			else:
				data= np.zeros(10000,200,200,6).astype(np.float32)+np.nan
			label= np.zeros(10000).astype(str)
			for cnt,id in enumerate(self.f.keys()):
				for obj in ['elg','lrg']:
					print(self.f['/%s/%s' % (id,obj)].shape)
					img= src+back
					if use_invvar:
						ivar= add_invvar(src_ivar,back_ivar)
				if cnt > 5: break
			assert( np.where( np.isfinite(data).flatten() == False )[0].size == 0)
			dset = fobj.create_dataset('%s/images' % node, data=data,chunks=True)
			dset = fobj.create_dataset('%s/labels' % node, data=label,chunks=True)
		print('srcpimage_noivar has been created')
		return fobj

	def get_srcpimage(self,use_ivar=False):
		fobj= self.create_srcpimage(use_ivar=use_ivar)
		return fobj['/srcpimg/images'],fobj['/srcpimg/labels']

def main(unused_argv):
  # Get the data.
  #data_sets = mnist.read_data_sets(FLAGS.directory,
  #                                 dtype=tf.uint8,
  #                                 reshape=False,
  #                                 validation_size=FLAGS.validation_size)
  builder= BuildTraining(data_dir=FLAGS.directory)
  images,labels= builder.get_srcpimage(use_ivar=FLAGS.use_ivar)
  raise ValueError
  # Python3 h5py containers arg!
  #keys= list(fobj.keys())
  #keys2= list(fobj[keys[0]].keys())
  #fobj['/95941990/elg'].shape
  #fobj['/*/elg'].shape

  # Shuffle samples
  inds= np.shuffle(range(dset.shape[0]))
  keep={}
  keep['test']= inds[ :FLAGS.test_size ]
  keep['val']= inds[ FLAGS.test_size:FLAGS.val_size ]
  keep['train']= inds[ FLAGS.test_size+FLAGS.val_size: ]
  for key,sz in zip(['test','val','train'],[1000,1000,8000]):
	assert(keep[key].size == sz)
  # data set object
  class DataSet(object):
    def __init__(self,images,labels,keep):
		self.images= images[keep]
		self.labels= labels[keep]
		self.data_set.num_examples = keep.size
  data={}
  for key in keep.keys():
    data[key]= DataSet(images,labels,keep[key])
  # Convert to Examples and write the result to TFRecords.
  convert_to(data['train'], 'train')
  convert_to(data['val'], 'validation')
  convert_to(data['test'], 'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/Users/kaylan1/tensorflow/galaxies/data_dir',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--val_size',
      type=int,
      default=1000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  parser.add_argument(
      '--test_size',
      type=int,
      default=1000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  parser.add_argument(
      '--use_ivar',
      action='store_true',
      help="""\
      put invvar in the samples\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


