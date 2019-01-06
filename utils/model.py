import tensorflow as tf
import numpy as np
from deform_conv_wrapper import deform_conv2d

def get_weight(weight_shape, name, trainable=True):
	initial = tf.truncated_normal(weight_shape, stddev=0.1)
	return tf.Variable(initial, name=name+'_weight', trainable=trainable)

def get_bias(bias_shape, name, trainable=True):
	initial = tf.constant(0., shape=bias_shape)
	return tf.Variable(initial, name=name+'_bias', trainable=trainable)

def get_offset(x, kernel_size, stride, name, deform_group=1):
	offset = tf.layers.conv2d(inputs=x,
							  filters=2*deform_group*kernel_size**2,
							  kernel_size=kernel_size,
							  strides=(stride, stride),
							  padding='SAME',
							  data_format='channels_first',
							  activation=None,
							  kernel_initializer=tf.zeros_initializer(),
							  name=name+'_offset')
	return offset

def deform_conv2d_block(x,
						output_channel,
						kernel_size,
						stride,
						name,
						is_training,
						deform_group=1,
						bn=False,
						trainable=True):
	input_channel = x.get_shape().as_list()[1]
	weight_shape = [kernel_size, kernel_size, input_channel, output_channel]
	bias_shape = [output_channel]
	W = get_weight(weight_shape, name=name, trainable=trainable)
	W = tf.transpose(W, perm=[3, 2, 0, 1]) # the order filter in deform conv is arranged as [output_channel, input_channel, kernel_size, kernel_size]
	b = get_bias(bias_shape, name=name, trainable=trainable)
	offset = get_offset(x, kernel_size, stride, name, deform_group)

	x = deform_conv2d(x=x,
					  filter=W,
					  offset=offset,
					  strides=[1, 1, stride, stride],
					  rates=[1, 1, 1, 1],
					  padding='SAME',
					  deform_group=deform_group,
					  data_format='NCHW')
	x = tf.nn.bias_add(x, b, data_format='NCHW', name=name+'_bias')
	if bn:
		x = tf.layers.batch_normalization(x, axis=1, training=is_training, name=name+'_bn', trainable=trainable)
	x = tf.nn.relu(x)	

	return x, offset # offset is also returned for visualization
	

def regular_conv2d_block(x, 
						  output_channel,
						  kernel_size,
						  stride,
						  name,
						  is_training,
						  bn=False,
						  trainable=True):
	input_channel = x.get_shape().as_list()[1]
	weight_shape = [kernel_size, kernel_size, input_channel, output_channel]
	bias_shape = [output_channel]
	
	W = get_weight(weight_shape, name=name, trainable=trainable)
	b = get_bias(bias_shape, name=name, trainable=trainable)
	x = tf.nn.conv2d(input=x, 
					 filter=W, 
					 strides=[1, 1, stride, stride], 
					 padding='SAME', 
					 data_format='NCHW',
					 name=name+'_conv2d')
	x = tf.nn.bias_add(x, b, data_format='NCHW', name=name+'_bias')
	if bn:
		x = tf.layers.batch_normalization(x, axis=1, training=is_training, name=name+'_bn', trainable=trainable)
	x = tf.nn.relu(x)	

	return x

def regular_model(input_pl, is_training_pl, bn=False):
	x = regular_conv2d_block(x=input_pl,
							  output_channel=32,
							  kernel_size=3,
							  stride=1,
							  name='conv11',
							  is_training=is_training_pl,
							  bn=bn)
	x = regular_conv2d_block(x=x,
							  output_channel=64,
							  kernel_size=3,
							  stride=2,
							  name='conv12',
							  is_training=is_training_pl,
							  bn=bn)
	x = regular_conv2d_block(x=x,
							  output_channel=128,
							  kernel_size=3,
							  stride=1,
							  name='conv21',
							  is_training=is_training_pl,
							  bn=bn)
	x = regular_conv2d_block(x=x,
							  output_channel=128,
							  kernel_size=3,
							  stride=2,
							  name='conv22',
							  is_training=is_training_pl,
							  bn=bn)
	# x = tf.transpose(x, perm=[0, 2, 3, 1])
	x = tf.nn.avg_pool(value=x,
					   ksize=[1, 1, 7, 7],
					   strides=[1, 1, 1, 1],
					   padding='VALID',
					   data_format='NCHW',
					   name='avg_pool')
	x = tf.squeeze(x, [2, 3])
	x = tf.layers.dense(inputs=x,
						units=10,
						name='fc1')
	return x

def deform_model(input_pl, is_training_pl, bn=False, trainable=True):
	x, _ = deform_conv2d_block(x=input_pl,
							  output_channel=32,
							  kernel_size=3,
							  stride=1,
							  name='conv11',
							  is_training=is_training_pl,
							  bn=bn,
							  trainable=trainable)
	x, _ = deform_conv2d_block(x=x,
							  output_channel=64,
							  kernel_size=3,
							  stride=2,
							  name='conv12',
							  is_training=is_training_pl,
							  bn=bn,
							  trainable=trainable)
	x, _ = deform_conv2d_block(x=x,
							  output_channel=128,
							  kernel_size=3,
							  stride=1,
							  name='conv21',
							  is_training=is_training_pl,
							  bn=bn,
							  trainable=trainable)
	x, _ = deform_conv2d_block(x=x,
							  output_channel=128,
							  kernel_size=3,
							  stride=2,
							  name='conv22',
							  is_training=is_training_pl,
							  bn=bn,
							  trainable=trainable)
	# x = tf.transpose(x, perm=[0, 2, 3, 1])
	x = tf.nn.avg_pool(value=x,
					   ksize=[1, 1, 7, 7],
					   strides=[1, 1, 1, 1],
					   padding='VALID',
					   data_format='NCHW',
					   name='avg_pool')
	x = tf.squeeze(x, [2, 3])
	x = tf.layers.dense(inputs=x,
						units=10,
						name='fc1',
						trainable=trainable)
	return x



def deform_viz_model(input_pl, is_training_pl, bn=False, trainable=False):
	x, offset11 = deform_conv2d_block(x=input_pl,
							  output_channel=32,
							  kernel_size=3,
							  stride=1,
							  name='conv11',
							  is_training=is_training_pl,
							  bn=bn,
							  trainable=trainable)
	x, offset12 = deform_conv2d_block(x=x,
							  output_channel=64,
							  kernel_size=3,
							  stride=2,
							  name='conv12',
							  is_training=is_training_pl,
							  bn=bn,
							  trainable=trainable)
	x, offset21 = deform_conv2d_block(x=x,
							  output_channel=128,
							  kernel_size=3,
							  stride=1,
							  name='conv21',
							  is_training=is_training_pl,
							  bn=bn,
							  trainable=trainable)
	x, offset22 = deform_conv2d_block(x=x,
							  output_channel=128,
							  kernel_size=3,
							  stride=2,
							  name='conv22',
							  is_training=is_training_pl,
							  bn=bn,
							  trainable=trainable)
	# Only the offsets are necessary for visualization, the rest remaining part can be omitted.
	return offset11, offset12, offset21, offset22
