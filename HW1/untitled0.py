from PIL import Image
import pylab as pyl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
#============選取圖片特徵點===============#
plt.figure(1)
plt.subplot(2,2,1)     
im1 = Image.open('1.JPG')
pyl.imshow(im1)
plt.subplot(2,2,2) 
im2 = Image.open('2.JPG')
pyl.imshow(im2)

plt.subplot(2,2,3) 
im3 = Image.open('3.JPG')
pyl.imshow(im3)

im1_rgb=im1.load()
x1 =pyl.ginput(3)

print ('you clicked:',x1)
#print(x1[0][0],x1[0][1])
#print(im1_rgb[x1[0][0],x1[0][1]])
x2 =pyl.ginput(6)
print ('you clicked:',x2)
x2_1=x2[:3]#圖1與圖2的相似座標
x2_2=x2[3:]#圖2與圖3的相似座標

x3 =pyl.ginput(3)
print ('you clicked:',x3)
#=============將座標值調整成3*3array==========================#
def homoadjust(x):
    for i in range(len(x)):
        x[i]=list(x[i])
        x[i].append(1)
    x=np.array(x)
    #x=x.astype(np.int32)
    #print(x)
    return x
homo_x1=homoadjust(x1)
homo_x2_1=homoadjust(x2_1)
homo_x2_2=homoadjust(x2_2)
homo_x3=homoadjust(x3)
#========================找到變換矩陣==================================#
def transmatrix(x,y):
    x=x.T
    y=y.T
    homoarray=(x).dot(np.linalg.inv(y))
    newaff=np.delete(homoarray,2,axis=0)
    return newaff,homoarray

M12,H12=transmatrix(homo_x1,homo_x2_1)

M23,H23=transmatrix(homo_x2_2,homo_x3)
H13=H12.dot(H23)
M13=np.delete(H13,2,axis=0)
print(M12)
#============================================================#
def trim(frame): 
    #crop top 
    if not np.sum(frame[0]): 
        return trim(frame[1:]) 
    #crop top 
    if not np.sum(frame[-1]): 
        return trim(frame[:-2]) 
    #crop top 
    if not np.sum(frame[:,0]): 
        return trim(frame[:,1:]) 
    #crop top 
    if not np.sum(frame[:,-1]): 
        return trim(frame[:,:-2]) 
    return frame
#====================取得座標rgb=============================#
def Getrgb(x,y):
    getrgb=[]
    for i in range(len(x)):
        getrgb.append(y[x[i][0],x[i][1]])
    #print(getrgb)
    return getrgb
#Getrgb(x1,im1_rgb)
#===================bilinear interponation==================#
def bilinear(x,y):
    p = np.array([np.floor(x), np.floor(y),3])#Q11(x1,y1)
    p1 = np.array([p[0], p[1]+1,3])   #Q12(x1,y2)
    p2 = np.array([p[0]+1, p[1],3])   #Q21(x2,y1)
    p3 = np.array([p[0]+1, p[1]+1,3])  #Q22(x2,y2)
    fr1=(p2[0]-x)/(p2[0]-p[0])*p+(x-p[0])/(p2[0]-p[0])*p2
    fr2=(p2[0]-x)/(p2[0]-p[0])*p1+(x-p[0])/(p2[0]-p[0])*p3
    fp=(p1[1]-y)/(p1[1]-p[1])*fr1+(y-p[1])/(p1[1]-p[1])*fr2
    return fp
#========================================================#
cv_im1 = cv2.imread('1.JPG')# 基准图像
cv_im2 = cv2.imread('2.JPG')
cv_im3 = cv2.imread('3.JPG')
rows1, cols1 = cv_im1.shape[:2]
rows2, cols2 = cv_im2.shape[:2]
rows3, cols3 = cv_im3.shape[:2]

warp_im2=cv2.warpAffine(cv_im2,M12,(cols1*3, rows1*2))
warp_im3=cv2.warpAffine(cv_im3,M13,(cols1*3, rows1*2))
warp_im2[0:rows1,0:cols1]=cv_im1
res= np.zeros([rows1*2, cols1*3, 3], np.uint8)
temp= np.zeros([rows1*2, cols1*3, 3], np.uint8)

for row in range(0, rows1*2):
    for col in range(0, cols1*3):
        if (warp_im3[row, col].any() and warp_im2[row, col].any()):
           temp[row,col]=warp_im2[row,col]
  
res=warp_im2-temp+warp_im3     
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
res=trim(res)
# show the result

plt.figure()
plt.imshow(res)
plt.show()

