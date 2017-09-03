# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 08:55:46 2017

@author: yudah
"""
import cv2
import numpy as np
from PIL import Image

inputPath = "E:\\test_rice.jpg"


img = Image.open(inputPath)
(width, height) = img.size

print(width, height)

if(height > width):
    img = img.transpose(Image.ROTATE_90)
    #box = (208,0,width,height)
    #img = img.crop(box)
    #crop_img = n_img[208:, :]
    #rz_img = cv2.resize(crop_img,(int(height/10), int(width/10)), interpolation = cv2.INTER_CUBIC)
    #cv2.namedWindow('a_window', cv2.WINDOW_AUTOSIZE) #Facultative

    #cv2.imshow("a_window", n_img)
    #cv2.waitKey(0)

box = (208,68,height,width)
img = img.crop(box)
(n_width, n_height) = img.size
img = img.resize((int(n_width/5), int(n_height/5)))
img.save('E:\\resize_rice.jpg',"JPEG")

print(img.size)

