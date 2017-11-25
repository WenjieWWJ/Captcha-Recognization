#coding=utf-8
import os
import numpy as np
import tensorflow as tf
from data_util import data_batch,data_test
# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets("../MNIST", one_hot=True)

lr = 0.001
iter_num = 500

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

images = tf.placeholder(tf.float32,[None,784])
labels = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)



# First Convolutional Layer
x_image = tf.reshape(images, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

results=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(labels*tf.log(results))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(results,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	iterator = data_batch('./MNIST',50,iter_num)
	for i in range(iter_num):

		(a,b)= iterator.next()
		if(i%10==0):
			# print "iter {} :".format(i),sess.run(cross_entropy,feed_dict={images:a,labels:b,keep_prob: 1.0})
			print "iter {} :".format(i),sess.run(accuracy,feed_dict={images:a,labels:b,keep_prob: 1.0})
		sess.run(train_step,feed_dict={images:a,labels:b, keep_prob: 0.5})

	images_test,labels_test  = data_test("./MNIST",50)

	print sess.run(accuracy, feed_dict={images: images_test, labels:labels_test, keep_prob: 1.0 })
    
	path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'models_res', 'cnn_mnist.ckpt'))
	print("Model saved... :", path)

	# for i in range(iter_num):

	# 	batch_xs, batch_ys = data.train.next_batch(100)
	# 	if(i%100==0):
	# 		print "iter {} :".format(i),sess.run(accuracy,feed_dict={images:batch_xs,labels:batch_ys, keep_prob: 1.0})
	# 	sess.run(train_step, feed_dict={images: batch_xs, labels: batch_ys, keep_prob: 0.5})

	# print(sess.run(accuracy, feed_dict={images: data.test.images, labels: data.test.labels, keep_prob: 1.0}))

