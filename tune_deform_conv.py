from __future__ import division
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import sys
sys.path.append('utils')
slim = tf.contrib.slim

from utils.model import deform_model, regular_model
from utils.mnist_gen import get_gen

batch_size = 64
learning_rate = 1e-4 # Decrease the learning rate by 10 for fine-tune.
steps_per_epoch = int(np.ceil(60000 / batch_size))
validation_steps = 10000

train_data_generator_scaled = get_gen('train', batch_size=batch_size, shuffle=True, scaled=True)
test_data_generator_scaled = get_gen('test', batch_size=1, shuffle=True, scaled=True)
input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 1, 28, 28], name='input')
label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='label')
is_training_placeholder = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')


output_tensor = deform_model(input_placeholder, is_training_placeholder, bn=True, trainable=False)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_placeholder, logits=output_tensor))

correct_prediction = tf.equal(tf.argmax(output_tensor, 1), tf.argmax(label_placeholder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False



# Restore all the variables from the regular model except for those related to offsets.
variables_to_restore = []
for var in tf.global_variables():
	if 'offset' not in var.name:
		variables_to_restore.append(var)
restorer = tf.train.Saver(variables_to_restore)

# Define saver
saver = tf.train.Saver(tf.global_variables())

# Check if the trainable variables are correctly set.
print("Only the following variables are trainable:")
for var in tf.trainable_variables():
	print(var.name)

if __name__ == '__main__':
	with tf.Session(config=config) as sess:
		sess.run(init)
		restorer.restore(sess, "checkpoint/regular_model")
		print("Weights restored from 'checkpoint/regular_model'.")
		for i in range(10):
			acc_sum, loss_sum = 0., 0.
			batch_count = 0
			while batch_count < steps_per_epoch:
				input, label = next(train_data_generator_scaled)
				input = np.transpose(input, axes=[0, 3, 1, 2]) # to fit "NCHW" data format
				loss, acc, _ = sess.run([cross_entropy, accuracy, train_op], 
									    feed_dict={input_placeholder: input,
									 			  label_placeholder: label,
									 			  is_training_placeholder: False}) 
									 			  # The moving average of batch normalization has to be frozen
				loss_sum += loss
				acc_sum += acc
				batch_count += 1
			print('Epoch {} training: loss={}, acc={}'.format(i + 1, loss_sum / batch_count, acc_sum / batch_count))

		# Save session into checkpoint file
		try:
			os.mkdir('checkpoint')
		except:
			pass
		save_path = saver.save(sess, "checkpoint/deform_model")
		print("Model saved in path: %s" % save_path)

		# Testing on scaled MNIST
		acc_sum, loss_sum = 0., 0.
		batch_count = 0
		while batch_count < validation_steps:
			input, label = next(test_data_generator_scaled)
			input = np.transpose(input, axes=[0, 3, 1, 2]) # to fit "NCHW" data format
			loss, acc = sess.run([cross_entropy, accuracy], 
							    feed_dict={input_placeholder: input,
							 			  label_placeholder: label,
							 			  is_training_placeholder: False})
			loss_sum += loss
			acc_sum += acc
			batch_count += 1
		print('Testing for deform conv model on scaled MNIST: loss={}, acc={}'.format(loss_sum / batch_count, acc_sum / batch_count))
