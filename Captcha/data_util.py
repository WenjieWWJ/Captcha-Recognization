#coding=utf-8
import os
import numpy as np
import struct
import random
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  

###  data_load

def char2number(char):

	if char >= '0' and char <= '9':
		return int(char)
	elif char >= 'A' and char <= 'Z':
		res = ord(char) - ord('A') + 10
		return res
	else:
		print char+' error'
		return 0
		

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def data_input(data_dir_images,digits_num,digits_size):

	data_images = []
	data_labels = []

	for _,_,files in os.walk(data_dir_images):
		for file in files:
			test = mpimg.imread(data_dir_images+"/"+file)
			label_start = file.find("_")
			label = file[label_start+1:label_start+digits_num+1]
			label_one = [0]*digits_num*digits_size
			for i in range(digits_num):
				label_one[i*digits_size+char2number(label[i])]=1.0
			gray = rgb2gray(test)
			tmp = gray.reshape(-1)
			data_images.append(tmp)
			data_labels.append(label_one)
	data_images = np.array(data_images,dtype = np.float32)
	data_labels = np.array(data_labels,dtype = np.float32)
	data_images = np.ones(data_images.shape)-data_images
	# data_labels = np.ones(data_labels.shape)-data_labels
	(numImages,dimensions) = data_images.shape
	return data_images,data_labels,numImages,dimensions


def data_batch(data_dir_images,digits_num,digits_size,batch_num,iter_number):


	data_images = []
	data_labels = []

	files = os.listdir(data_dir_images)
	i = 0
	iter_num = len(files)//batch_num

	for j in range(iter_number):
		if i == iter_num:
			i==0
		i += 1
		data_images = []
		data_labels = []

		for file in files[i*batch_num:i*batch_num+batch_num]:

			test = mpimg.imread(data_dir_images+"/"+file)
			label_start = file.find("_")
			label = file[label_start+1:label_start+digits_num+1]
			label_one = [0]*digits_num*digits_size
			for i in range(digits_num):
				label_one[i*digits_size+char2number(label[i])]=1.0
			gray = rgb2gray(test)
			tmp = gray.reshape(-1)
			data_images.append(tmp)
			data_labels.append(label_one)

		data_images = np.array(data_images,dtype = np.float32)
		data_labels = np.array(data_labels,dtype = np.float32)
		data_images = np.ones(data_images.shape)-data_images
		yield (data_images,data_labels)




	if(iter_num<1):
		print("iter_num < 1")

	i = 0
	for j in range(iter_number):

		image,label = (data_images[i*batch_num:(i+1)*batch_num,:],data_labels[i*batch_num:(i+1)*batch_num])
		i = i + 1
		if(i==iter_num):
			i=0
		yield (image,label)

def data_test(data_dir_images,digits_num,digits_size,batch_num):


	data_images,data_labels,numImages,dimensions = data_input(data_dir_images,digits_num,digits_size)
	
	iter_num = numImages//batch_num
	if(iter_num<1):
		error("iter_num < 1")

	i = random.randint(0,iter_num-1)
	image,label = (data_images[i*batch_num:(i+1)*batch_num,:],data_labels[i*batch_num:(i+1)*batch_num])

	return image,label

def data_sample(path):
	image = []
	test = mpimg.imread(path)
	gray = rgb2gray(test)
	gray = gray.reshape(-1)
	image.append(gray)
	image = np.array(image,dtype = np.float32)
	image = np.ones(image.shape)-image
	return image

## for test the code

# iter_num = 5
# iterator = data_batch('./data_2/train',4,36,2,5)
# for i in range(iter_num):

# 	(a,b)= iterator.next()
# 	# print a,b
# 	print b[0]
# 	t = a[0].reshape(60,110)
# 	plt.imshow(t, cmap='Greys_r')
# 	plt.show()
