import os
from os.path import join

import tensorflow as tf
from tensorflow.python.framework import ops

current_dir = os.path.dirname(__file__)
build_dir = join(current_dir, '../build')
deform_conv2d_op_exe = tf.load_op_library(join(build_dir, 'deform_conv_op.so'))
deform_conv2d_grad_op_exe = tf.load_op_library(join(build_dir, 'deform_conv_grad_op.so'))


def deform_conv2d(x,
                  filter,
                  offset,
                  strides,
                  rates,
                  padding,
                  deform_group,
                  data_format):
    '''
    :param x: Input tensor, a 4-d tf_tensor in "NCHW" order;
    :param filter: Convolution kernel, a 4-d tf_tensor, the order shoule be
            '[output_channel, input_channel, kernel_height, kernel_width]';
    :param offset: A 4-d tf_tensor, offsets of the convolution sampling location, the order should be
            '[batch_size, offset_dim(deform_group * kernel_height * kernel_width * 2), output_height, output_width]';
    :param strides: stride, a list of length 4 in "NCHW" order;
    :param rates: rate (dilation), a list of length 4 in "NCHW" order;
    :param padding: padding method, string. Can either be "SAME" or "VALID";
    :param deform_group: a non-zero int, indicating how many different groups of offsets are applied, Default=1;
    :param data_format: string, only "NCHW" is supported for now.

    :return: Output tensor, a 4-d tf_tensor in "NCHW" order.
    '''
    return deform_conv2d_op_exe.deform_conv2d_op(x=x,
                                                 filter=filter,
                                                 offset=offset,
                                                 strides=strides,
                                                 rates=rates,
                                                 padding=padding,
                                                 deform_group=deform_group,
                                                 data_format=data_format)


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
