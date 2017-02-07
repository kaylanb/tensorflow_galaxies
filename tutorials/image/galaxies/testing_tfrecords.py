import tensorflow as tf
import numpy as np
import os

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#def _int_feature(value):
#    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def dump_tfrecord(data, out_file):
	if os.path.exists(out_file):
		os.remove(out_file)
	writer = tf.python_io.TFRecordWriter(out_file)
	for x in data:
		example = tf.train.Example(
			features=tf.train.Features(feature={
				'label': _int64_feature(10),\
				'x': _bytes_feature(x.tostring())})
		)
		writer.write(example.SerializeToString())
	writer.close()


def load_tfrecord(file_name):
	reader = tf.TFRecordReader()
	fn_queue = tf.train.string_input_producer([file_name])
	_, serialized_example = reader.read(fn_queue)
	feats = tf.parse_single_example(
		serialized_example,
		features = {'label': tf.FixedLenFeature([], tf.int64),\
					'x': tf.FixedLenFeature([], tf.string)}
	)
	data = tf.reshape(tf.decode_raw(feats['x'], tf.float32), [4 * X.size])
	return data
	#for s_example in tf.python_io.tf_record_iterator(file_name):
	#    example = tf.parse_single_example(s_example, features=features)
	#    data.append(tf.expand_dims(example['x'], 0))
	#return tf.concat(0, data)


if __name__ == "__main__":
	sess = tf.InteractiveSession()
	X = np.zeros((32,32,3)).astype(np.float32)
	X[:,16,0]=10
	X[16,:,1]=5
	X[:,4,2]=2
	Y= tf.convert_to_tensor(X)
	print(Y.eval()[:,16,0])
	print(Y.eval()[16,:,1])
	print(Y.eval()[:,4,2])
	dump_tfrecord([X,X], 'test_tfrecord')
	print('dump ok')
	vec = load_tfrecord('test_tfrecord')
	print('loaded')
	arr= tf.reshape(tf.slice(vec, [0],[X.size]),list(X.shape))
	print('reshaped')
	Z= tf.convert_to_tensor(arr)
	print('converted')
	print(Z.eval())
	raise ValueError
	print(vec.eval())
	#print(arr.eval()[:,16,0])
	#print(arr.eval()[16,:,1])
	#print(arr.eval()[:,4,2])
	#with tf.Session() as sess:
	#	sess.run(tf.global_variables_initializer())
	#	Y = sess.run([arr])
	#	print(Y[:,16,0]) #=10
	#	print(Y[16,:,1]) #=5
	#	print(Y[:,4,2]) #=2
	sess.close()
	raise ValueError
