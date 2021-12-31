import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('1.png',0)
height = img.shape[0]
weight = img.shape[1]

kernel=np.ones(shape=(7,7))

def Do_erosion(img,kernel):
    kernel_size = kernel.shape[0]
    erosion_img = np.array(img)
    height = erosion_img.shape[0]
    weight = erosion_img.shape[1]
    eroImg = np.zeros((height,weight))
    kernelcenter = int((kernel_size-1)/2)
    for i in range(kernelcenter, height-kernel_size+1):
        for j in range(kernelcenter, weight-kernel_size+1):
            eroImg[i, j] = np.min(erosion_img[i-kernelcenter:i+kernelcenter,j-kernelcenter:j+kernelcenter])
    return eroImg

def Do_dilation(img, kernel):
    kernel_size = kernel.shape[0]
    dilation_img = np.array(img)
    height = dilation_img.shape[0]
    weight = dilation_img.shape[1]
    dilaImg = np.zeros((height,weight))
    kernelcenter = int((kernel_size-1)/2)
    for i in range(kernelcenter,height-kernel_size+1):
        for j in range(kernelcenter, weight-kernel_size+1):
            dilaImg[i, j] = np.max(dilation_img[i-kernelcenter:i+kernelcenter,j-kernelcenter:j+kernelcenter])
    return dilaImg

erosionImg=Do_erosion(img,kernel)
dilationImg=Do_dilation(img,kernel)
Opening=Do_dilation(erosionImg,kernel)
Closing=Do_erosion(dilationImg,kernel)
Smoothing=Do_erosion(Do_dilation(Opening,kernel),kernel)
Gradient=dilationImg-erosionImg

plt.subplot(3,3,1)
plt.imshow(img,'gray'),plt.title('img')
plt.subplot(3,3,2)
plt.imshow(erosionImg,'gray'),plt.title('Erosion')
plt.subplot(3,3,3)
plt.imshow(dilationImg,'gray'),plt.title('Dilation')
plt.subplot(3,3,4 )
plt.imshow(Opening,'gray'),plt.title('Opening')
plt.subplot(3,3,5 )
plt.imshow(Closing,'gray'),plt.title('Closing')
plt.subplot(3,3,6 )
plt.imshow(Gradient,'gray'),plt.title('Gradient')
plt.subplot(3,3,7 )
plt.imshow(Smoothing,'gray'),plt.title('Smoothing')