import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from deform_conv_wrapper import deform_conv2d
import numpy as np

# np.random.seed(0)
batch_size = 2
input_size = 9
input_channel = 2
output_channel = 8
filter_size = 3
stride = 1
rate = 1
output_size = 4
deform_group = 2
pad = 'SAME'
dtype = tf.float32

input = np.random.random((batch_size, input_channel, input_size, input_size))
filter = np.random.random((output_channel, input_channel, filter_size, filter_size))
offset = np.random.random((batch_size, deform_group * 2 * filter_size ** 2, output_size, output_size))
output_grad = np.random.random((batch_size, output_channel, output_size, output_size))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
with tf.Session(config=config) as sess:
    with tf.device('/gpu:0'):
        input_tensor = tf.constant(input, dtype=dtype)
        filter_tensor = tf.constant(filter, dtype=dtype)
        offset_tensor = tf.constant(offset, dtype=dtype)
        output_grad_tensor = tf.constant(output_grad, dtype=dtype)

        res = deform_conv2d(x=input_tensor,
                            filter=filter_tensor,
                            offset=offset_tensor,
                            strides=[1, 1, stride, stride],
                            rates=[1, 1, rate, rate],
                            padding=pad,
                            deform_group=deform_group,
                            data_format="NCHW")
        grad = tf.gradients(res, [input_tensor, filter_tensor, offset_tensor])
        output = [sess.run(g) for g in grad]
print(output[0])
print(output[1])
print(output[2])
