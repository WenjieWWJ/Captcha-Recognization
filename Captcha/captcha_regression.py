#coding=utf-8

'''
this is a simple captcha recognization program implemented by softmax in tensorflow

'''


import os
import numpy as np
import tensorflow as tf
from data_util import data_batch,data_test
import matplotlib.pyplot as plt

lr = 0.001
iter_num = 50000
batch_num = 200

dataset_dir = './data'
ckpt_name = 'softmax_data.ckpt'

image_width = 110
image_height = 60
digits_num = 4
digits_size = 10


images = tf.placeholder(tf.float32,[None,image_height*image_width])
labels = tf.placeholder(tf.float32,[None,digits_num*digits_size])


w = tf.Variable(tf.zeros([image_height*image_width,digits_num*digits_size]))
b = tf.Variable(tf.zeros([digits_num*digits_size]))

y = tf.matmul(images, w) + b
results = tf.nn.softmax(y)
# cross_entropy = -tf.reduce_sum(labels*tf.log(results),1)

'''
problem :
	
		-tf.reduce_sum(labels*tf.log(tf.nn.softmax(y)),1)  != tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y)

'''


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y)
cross_entropy = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)



labels_ = tf.reshape(labels,[-1,digits_size])
results = tf.reshape(results,[-1,digits_size])

correct_prediction = tf.equal(tf.argmax(results,1), tf.argmax(labels_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


saver = tf.train.Saver()

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	max_acc = 0
	iterator = data_batch(dataset_dir+'/train',digits_num,digits_size,batch_num,iter_num)
	for i in range(iter_num):

		(a,b)= iterator.next()
		# print a[0],b[0]
		# im = a[1].reshape(60,110)
		# fig = plt.figure()
		# plotwindow = fig.add_subplot(111)
		# plt.imshow(im , cmap='gray')
		# print np.argmax(b[1])
		# plt.show()
		if(i%100==0):
			tmp_loss = sess.run(cross_entropy,feed_dict={images:a,labels:b})
			tmp_accuracy = sess.run(accuracy,feed_dict={images:a,labels:b}),
			print "iter {} , loss: {}  accurary : {} ".format(i,tmp_loss,tmp_accuracy)
			images_test,labels_test  = data_test(dataset_dir+'/test',digits_num,digits_size,batch_num)
			valid_acc = sess.run(accuracy, feed_dict={images: images_test, labels:labels_test })
			print " valid : ", valid_acc
			if valid_acc>max_acc:
				max_acc = valid_acc
				path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'models_res', ckpt_name))
				print("Model saved... :", path)

		sess.run(train_step,feed_dict={images:a,labels:b})

	images_test,labels_test  = data_test(dataset_dir+'/test',digits_num,digits_size,batch_num)
	valid_acc = sess.run(accuracy, feed_dict={images: images_test, labels:labels_test })
	print " valid : ", valid_acc
	if valid_acc>max_acc:
		max_acc = valid_acc
		path = saver.save(sess, os.path.join(os.path.dirname(__file__), 'models_res', ckpt_name))
		print("Model saved... :", path)








