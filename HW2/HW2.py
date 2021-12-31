import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('1.png',0)
height = img.shape[0]
weight = img.shape[1]
dSobel = np.zeros((height,weight))
reg_res = np.zeros((height,weight))
conv_res = np.zeros((height,weight))
Gx = np.zeros(img.shape)
Gy = np.zeros(img.shape)
#====================Sobel==================================#
for i in range(height-1):
    for j in range(weight-1):
        Gx[i,j]=abs((img[i+1,j-1]+2*img[i+1,j]+img[i+1,j+1])-(img[i-1,j-1]+2*img[i-1,j]+img[i-1,j+1]))
        Gy[i,j]=abs((img[i-1,j-1]+2*img[i,j-1]+img[i+1,j-1])-(img[i-1,j+1]+2*img[i,j+1]+img[i+1,j+1]))
for i in range(height-1):
    for j in range(weight-1):
        if Gx[i,j]>255:
            Gx[i,j]=255
        elif Gx[i,j]<0:
            Gx[i,j]=0
        else:
            Gx[i,j]=Gx[i,j]
for i in range(height-1):
    for j in range(weight-1):
        if Gy[i,j]>255:
            Gy[i,j]=255
        elif Gy[i,j]<0:
            Gy[i,j]=0
        else:
            Gy[i,j]=Gy[i,j]
for i in range(height-1):
    for j in range(weight-1):
        dSobel[i,j]=Gx[i,j]+Gy[i,j]
        if dSobel[i,j]>255:
            dSobel[i,j]=255
        elif dSobel[i,j]<0:
            dSobel[i,j]=0 
        else:
            dSobel[i,j]=dSobel[i,j]
#====================模糊===================================#
a=1/9
for i in range(height-1):
    for j in range(weight-1):
        dSobel[i,j]=a*dSobel[i-1,j-1]+a*dSobel[i,j-1]+a*dSobel[i+1,j-1]+a*dSobel[i-1,j]+a*dSobel[i,j]+a*dSobel[i+1,j]+a*dSobel[i-1,j+1]+a*dSobel[i,j+1]+a*dSobel[i+1,j+1]        
#====================正規[0,1]================================#
reg_res=dSobel/255
#print(dSobel)
#=========================二階微分============================#
b=(-1)
for i in range(1,height-1):
    for j in range(1,weight-1):
        conv_res[i,j]=b*img[i-1,j-1]+b*img[i,j-1]+b*img[i+1,j-1]+b*img[i-1,j]+8*img[i,j]+b*img[i+1,j]+b*img[i-1,j+1]+b*img[i,j+1]+b*img[i+1,j+1]
for i in range(height-1):
    for j in range(weight-1):
        if conv_res[i,j]>255:
            conv_res[i,j]=255
        elif conv_res[i,j]<0:
            conv_res[i,j]=0
        else:
            conv_res[i,j]=conv_res[i,j]

#===============combine=======================================#
fin=conv_res*reg_res+img
for i in range(height-1):
    for j in range(weight-1):
        if fin[i,j]>255:
            fin[i,j]=255
        elif fin[i,j]<0:
            fin[i,j]=0
        else:
            fin[i,j]=fin[i,j]
#print(fin)
plt.subplot(2,2,1)
plt.imshow(img,'gray'),plt.title('img')
plt.subplot(2,2,2)
plt.imshow(dSobel,'gray'),plt.title('Sobel')
plt.subplot(2,2,3)
plt.imshow(conv_res,'gray'),plt.title('2Diff')
plt.subplot(2,2,4)
plt.imshow(fin,'gray'),plt.title('Final')
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img,'gray'),plt.title('img')
plt.subplot(1,2,2)
plt.imshow(fin,'gray'),plt.title('Final')