import cv2
import numpy as np
import matplotlib.pyplot as plt

def GX(src):
    img = cv2.imread(src,0)
    height = img.shape[0]
    weight = img.shape[1]
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    dSobel = np.zeros((height,weight))
    dSobelx = np.zeros((height,weight))
    dSobely = np.zeros((height,weight))
    Gx = np.zeros(img.shape)
    Gy = np.zeros(img.shape)
    for i in range(height-2):
        for j in range(weight-2):
            Gx[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * sx))
            Gy[i + 1, j + 1] = abs(np.sum(img[i:i + 3, j:j + 3] * sy))
            dSobel[i+1, j+1] = Gx[i+1, j+1]+ Gy[i+1, j+1]
            dSobelx[i+1, j+1] = np.sqrt(Gx[i+1, j+1])
            dSobely[i + 1, j + 1] = np.sqrt(Gy[i + 1, j + 1])
    print(Gx)
    print(Gy)
    return dSobel


src=('1.png')
img = cv2.imread(src,0)
a = np.uint8(GX(src))
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(a,'gray')

plt.show()
