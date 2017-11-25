#coding=utf-8

'''
load Mnist data and process them . 
provide the methods to return a batch

'''


import numpy as np
import struct
import random

### mnist data_load
def data_input(data_dir_images,data_dir_labels):

	filename = data_dir_images
	binfile = open(filename , 'rb')
	buf = binfile.read()
	 
	index = 0
	magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
	index += struct.calcsize('>IIII')


	data_mnist_X = np.zeros([numImages,numColumns*numRows],dtype = np.int32)

	for i in range(numImages):
		
		im = struct.unpack_from('>784B' ,buf, index)
		index += struct.calcsize('>784B')
		im = np.array(im).astype(np.float32)

		data_mnist_X[i,:]=im
	data_mnist_X = np.multiply(data_mnist_X, 1.0 / 255.0)#normalize


	filename = data_dir_labels
	binfile = open(filename , 'rb')
	buf = binfile.read()
	 
	index = 0
	magic_label, numLabels  = struct.unpack_from('>II' , buf , index)
	index += struct.calcsize('>II')

	numLabels = numImages
	data_mnist_Y = np.zeros([numLabels,10],dtype = np.float32)

	for i in range(numLabels):
		
		im = struct.unpack_from('>B' ,buf, index)
		index += struct.calcsize('>B')
		data_mnist_Y[i,im]=1.0

	return data_mnist_X,data_mnist_Y,numImages

def data_batch(data_dir,batch_num,iter_number):

	data_dir_images = data_dir+'/train-images.idx3-ubyte'
	data_dir_labels = data_dir+'/train-labels.idx1-ubyte'
	images,labels,numImages = data_input(data_dir_images,data_dir_labels)
	
	iter_num = numImages//batch_num
	if(iter_num<1):
		print("error: iter_num < 1")

	i = 0
	for j in range(iter_number):

		image,label = (images[i*batch_num:(i+1)*batch_num,:],labels[i*batch_num:(i+1)*batch_num])
		i = i + 1
		if(i==iter_num):
			i=0
		yield (image,label)

def data_test(data_dir,batch_num):
	
	data_dir_images = data_dir+'/t10k-images.idx3-ubyte'
	data_dir_labels = data_dir+'/t10k-labels.idx1-ubyte'
	images,labels,numImages = data_input(data_dir_images,data_dir_labels)

	iter_num = numImages//batch_num
	if(iter_num<1):
		error("iter_num < 1")

	i = random.randint(0,iter_num-1)
	image,label = (images[i*batch_num:(i+1)*batch_num,:],labels[i*batch_num:(i+1)*batch_num])

	return image,label

