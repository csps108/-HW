import numpy as np
def bilinear(x,y):
    p1 = np.array([np.floor(x), np.floor(y)])#Q11(x1,y1)
    p2 = np.array([p1[0], p1[1]+1])   #Q12(x1,y2)
    p3 = np.array([p1[0]+1, p1[1]])   #Q21(x2,y1)
    p4 = np.array([p1[0]+1, p1[1]+1])  #Q22(x2,y2)
    fr1=(p3[0]-x)/(p3[0]-p1[0])*p1+(x-p1[0])/(p3[0]-p1[0])*p3
    fr2=(p3[0]-x)/(p3[0]-p1[0])*p2+(x-p1[0])/(p3[0]-p1[0])*p4
    fp=(p2[1]-y)/(p2[1]-p1[1])*fr1+(y-p1[1])/(p2[1]-p1[1])*fr2
    return fp
rows=3
cols=9
a=np.zeros([rows*2,cols*2])
for row in range(rows):
    for col in range(cols):
        if (col!=2):
            aa=bilinear(a[0],a[1])
            print(aa.shape)
            