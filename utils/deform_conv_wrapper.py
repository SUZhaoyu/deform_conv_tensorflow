import tensorflow as tf
from tensorflow.python.framework import ops
import os
from os.path import join

current_dir = os.path.dirname(__file__)
build_dir = join(current_dir, '../build')
deform_conv2d_op_exe = tf.load_op_library(join(build_dir, 'deform_conv_op.so'))
deform_conv2d_grad_op_exe = tf.load_op_library(join(build_dir, 'deform_conv_grad_op.so'))
deform_conv2d = deform_conv2d_op_exe.deform_conv2d_op

@ops.RegisterGradient("DeformConv2dOp")
def _deform_conv2d_grad_op(op, grad):
	input_tensor = op.inputs[0]
	filter_tensor = op.inputs[1]
	offset_tensor = op.inputs[2]
	strides = op.get_attr('strides')
	rates = op.get_attr('rates')
	padding = op.get_attr('padding')
	deform_group = op.get_attr('deform_group') 
	data_format = op.get_attr('data_format')

	data_grad = deform_conv2d_grad_op_exe.deform_conv2d_grad_op(x=input_tensor,
																filter=filter_tensor,
																offset=offset_tensor,
																out_grad=grad,
																strides=strides,
																rates=rates,
																padding=padding,
																deform_group=deform_group,
																data_format=data_format)
	return data_grad