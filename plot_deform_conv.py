from __future__ import division
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import sys
import cv2
from utils.model import deform_viz_model
from utils.mnist_gen import get_gen
from utils.viz_tools import plot_offset, get_scaled_imgs, plot_gif, plot_concat

batch_size = 64
learning_rate = 1e-4 # Decrease the learning rate by 10 for fine-tune.
steps_per_epoch = int(np.ceil(60000 / batch_size))
validation_steps = 10000

train_data_generator_scaled = get_gen('train', batch_size=batch_size, shuffle=True, scaled=True)
test_data_generator_scaled = get_gen('test', batch_size=1, shuffle=True, scaled=True)
input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1, 28, 28], name='input')
is_training_placeholder = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

offset11, offset12, offset21, offset22 = deform_viz_model(input_placeholder, is_training_placeholder, bn=True)

restorer = tf.train.Saver(tf.global_variables())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False

if __name__ == '__main__':
	img = cv2.imread('img/test_img.png')[:, :, 0].astype(np.uint8)
	
	with tf.Session(config=config) as sess:
		restorer.restore(sess, "checkpoint/deform_model")
		print("Weights restored from 'checkpoint/deform_model'.")
		total_frames = 60
		input = get_scaled_imgs(img=img, total_frames=total_frames)
		input = np.expand_dims(input, axis=1) # to fit "NCHW" data format

		offset11_output, offset12_output, offset21_output, offset22_output = sess.run([offset11, offset12, offset21, offset22], 
														    feed_dict={input_placeholder: input / 255.,
														 			  is_training_placeholder: False}) 
														 			  # The moving average of batch normalization has to be frozen
		frames = []
		for i in range(total_frames):
			offset_11_plot = plot_offset(img=input[i, 0, :, :], offset=offset11_output[i,...], kernel_size=3, stride=1, padding="SAME", thres=0.1, text="offset_11, thres=0.1")
			offset_12_plot = plot_offset(img=input[i, 0, :, :], offset=offset12_output[i,...], kernel_size=3, stride=2, padding="SAME", thres=0.5, text="offset_12, thres=0.5")
			frames.append(plot_concat([offset_11_plot, offset_12_plot], space=0.1))
		plot_gif(frames)
			