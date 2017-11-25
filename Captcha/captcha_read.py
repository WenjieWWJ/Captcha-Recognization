#coding=utf-8

'''
for testing captcha

'''




import numpy as np




import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import numpy as np


from captcha.audio import AudioCaptcha
from captcha.image import ImageCaptcha
import random 


# audio = AudioCaptcha(voicedir='/path/to/voices')
# image = ImageCaptcha()
# image = ImageCaptcha(width = 110,height=60,fonts=['/path/A.ttf', '/path/B.ttf'])
image = ImageCaptcha(width = 160,height=60 )
image.write('1234', './test.jpg')
test = mpimg.imread('./test.jpg')  
print test.shape


def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
print test
gray = rgb2gray(test)  
# plt.imshow(gray, cmap='Greys_r')

print gray.shape

t = gray.reshape(-1)
print t.shape
print t
# plt.axis('off')
# plt.show()
