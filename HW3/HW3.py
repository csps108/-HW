# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:21:48 2019

@author: user
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def RandomNoise(img):
    noise_img=img
    for noiseX in range(3,img.shape[0]):
        for noiseY in range(3,img.shape[1]):
            if random.random()<=0.25:
                noise_img[noiseX,noiseY]=0
            elif 0.25<random.random()<=0.5:
                noise_img[noiseX,noiseY]=255
            else:
                noise_img[noiseX,noiseY]=noise_img[noiseX,noiseY]
    return noise_img 

def AdaptProcess(img, i, j, minSize, maxSize):
    kernelSize = minSize // 2
    Pix = img[i-kernelSize:i+kernelSize+1, j-kernelSize:j+kernelSize+1] #根據filter_size取得所有Pixel值
    Zmin = np.min(Pix)
    Zmax = np.max(Pix)
    Zmed = np.median(Pix)
    Zxy = img[i,j]
    if (Zmed>Zmin) and (Zmed<Zmax):  #跳到B
        if (Zxy>Zmin) and (Zxy<Zmax):
            return Zxy
        else:
            return Zmed
    else:                                
        minSize = minSize + 2
        if minSize <= maxSize:
            return AdaptProcess(img, i, j, minSize, maxSize)
        else:
            return Zmed

def Adaptive_Median_Filter(img, minsize, maxsize):
    borderSize = maxsize // 2
    img= cv2.copyMakeBorder(img, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_REFLECT)
    height = img.shape[0]
    weight = img.shape[1]
    for x in range(borderSize,height):
        for y in range(borderSize,weight):
            img[x,y] = AdaptProcess(img, x, y, minsize, maxsize)
    dst = img[0:img.shape[0], 0:img.shape[1]]
    return dst 

img=cv2.imread('./2.png',0)
height = img.shape[0]
weight = img.shape[1]
noise_img = np.zeros((height,weight))
for i in range(height):
    for j in range(weight):
        noise_img[i,j]=img[i,j]
noise_img=RandomNoise(noise_img)
filter_img=Adaptive_Median_Filter(noise_img,3,7)
plt.subplot(2,2,1)
plt.imshow(img,'gray'),plt.title('img')
plt.subplot(2,2,2)
plt.imshow(filter_img,'gray'),plt.title('Adaptive Median Filter')
plt.subplot(2,2,3)
plt.imshow(noise_img,'gray'),plt.title('Noise')
