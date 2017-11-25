#coding=utf-8
import os
import numpy as np
import struct
import random
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  

from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha
import random 


vocab = [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
image = ImageCaptcha(width = 160,height=60 )



###  data_load

def char2number(char):

	if char >= '0' and char <= '9':
		return int(char)
	elif char >= 'a' and char <= 'z':
		res = ord(char) - ord('a') + 10
		return res	
	elif char >= 'A' and char <= 'Z':
		res = ord(char) - ord('A') + 36
		return res
	else:
		print char+' error:char2number()'
		return 0
		

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def data_batch(data_dir_images,digits_num,digits_size,batch_num,iter_number):


	data_images = []
	data_labels = []
	for i in range(batch_num):
		x = []
		name = ''
		for j in range(digits_num):
			x.append(random.randint(0,digits_size-1))
			name = name+bytes(vocab[x[j]])

		file = './data_3/train/train-{}_{}.jpg'.format(i,name)
		image.write(name, file)
		test = mpimg.imread(file)
		label_one = [0]*digits_num*digits_size

		for j in range(digits_num):
			label_one[j*digits_size+ x[j]]=1.0
		gray = rgb2gray(test)
		tmp = gray.flatten()/255
		data_images.append(tmp)
		data_labels.append(label_one)
		os.remove(file)

	data_images = np.array(data_images,dtype = np.float32)
	data_labels = np.array(data_labels,dtype = np.float32)

	return (data_images,data_labels)



def data_sample(path):
	image = []
	test = mpimg.imread(path)
	gray = rgb2gray(test)
	gray = gray.reshape(-1)
	image.append(gray)
	image = np.array(image,dtype = np.float32)
	# image = np.ones(image.shape)-image
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
