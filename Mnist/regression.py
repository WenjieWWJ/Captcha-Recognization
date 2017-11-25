#coding=utf-8

'''
this is a simple introductory tutorial of tensorflow using mnist.

	softmax,regression,...

'''

import numpy as np
import tensorflow as tf
from data_util import data_batch,data_test
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
# data = input_data.read_data_sets("../MNIST", one_hot=True)

lr = 0.01
iter_num = 1000

images = tf.placeholder(tf.float32,[None,784])
labels = tf.placeholder(tf.float32,[None,10])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

results = tf.nn.softmax(tf.matmul(images,w)+b)

cross_entropy = -tf.reduce_sum(labels*tf.log(results))

# train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)


correct_prediction = tf.equal(tf.argmax(results,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# iterator = data_batch('../MNIST',1,iter_num)

# batch_xs, batch_ys = data.train.next_batch(1)
# (a,b)= iterator.next()
# file = open("test.txt","w")
# file.write(batch_xs)
# file.write(a)

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)

	iterator = data_batch('../MNIST',100,iter_num)
	for i in range(iter_num):

		(a,b)= iterator.next()
		# im = a[1].reshape(28,28)
		# fig = plt.figure()
		# plotwindow = fig.add_subplot(111)
		# plt.imshow(im , cmap='gray')
		# print np.argmax(b[1])
		if(i%100==0):
			print "iter {} :".format(i),sess.run(cross_entropy,feed_dict={images:a,labels:b})
			print "iter {} :".format(i),sess.run(accuracy,feed_dict={images:a,labels:b})
		sess.run(train_step,feed_dict={images:a,labels:b})

	images_test,labels_test  = data_test("../MNIST",50,1)

	print sess.run(accuracy, feed_dict={images: images_test, labels:labels_test })




	# for i in range(iter_num):

	# 	batch_xs, batch_ys = data.train.next_batch(100)
	# 	if(i%100==0):
	# 		print "iter {} :".format(i),sess.run(accuracy,feed_dict={images:batch_xs,labels:batch_ys})
	# 	sess.run(train_step, feed_dict={images: batch_xs, labels: batch_ys})

	# print(sess.run(accuracy, feed_dict={images: data.test.images, labels: data.test.labels}))







