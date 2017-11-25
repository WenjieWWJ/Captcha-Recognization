#coding=utf-8

'''
captcha_generator used captcha(https://github.com/lepture/captcha) to generate a dataset

requires:
	* pip install captcha

'''

from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha
import random 


# audio = AudioCaptcha(voicedir='/path/to/voices')
# image = ImageCaptcha()
# image = ImageCaptcha(width = 110,height=60,fonts=['/path/A.ttf', '/path/B.ttf'])
image = ImageCaptcha(width = 160,height=60 )

# data = audio.generate('1234')
# audio.write('1234', 'out.wav')


# print data

vocab = [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# for i in range(400000):

# 	# x1 = random.randint(0,9)
# 	# x2 = random.randint(0,9)
# 	# x3 = random.randint(0,9)
# 	# x4 = random.randint(0,9)

# 	x1 = random.randint(0,35)
# 	x2 = random.randint(0,35)
# 	x3 = random.randint(0,35)
# 	x4 = random.randint(0,35)
# 	# data = image.generate('1A34')
# 	# name = bytes(x1)+bytes(x2)+bytes(x3)+bytes(x4)
# 	name = bytes(vocab[x1])+bytes(vocab[x2])+bytes(vocab[x3])+bytes(vocab[x4])
# 	# name = bytes(x1) 
# 	image.write(name, './data_3/train/train-{}_{}.png'.format(i,name))

for i in range(2000):

	x1 = random.randint(0,61)
	x2 = random.randint(0,61)
	x3 = random.randint(0,61)
	x4 = random.randint(0,61)
	name = bytes(vocab[x1])+bytes(vocab[x2])+bytes(vocab[x3])+bytes(vocab[x4])
	# x1 = random.randint(0,9)
	# x2 = random.randint(0,9)
	# x3 = random.randint(0,9)
	# x4 = random.randint(0,9)
	# # data = image.generate('1A34')
	# name = bytes(x1) 
	# name = bytes(x1)+bytes(x2)+bytes(x3)+bytes(x4)
	image.write(name, './data_3/test/test-{}_{}.png'.format(i,name))


