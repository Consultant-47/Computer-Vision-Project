
import scipy as sp
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.color
import skimage.filters
from skimage import exposure
from skimage import morphology
from skimage.filters import threshold_otsu, threshold_local
from skimage.feature import canny
from skimage.filters import sobel
from skimage.exposure import histogram

# path="S7_DST degrease_100C_0 hrs_2.jpg"
# dest_file="T1.png"

def contact_angle(path, dest_file):
    #converts to grayscale
    image=skimage.io.imread(path)
    gray_image = skimage.color.rgb2gray(image)
    gray_image = skimage.img_as_ubyte(gray_image)

    #finds elevation map and rescales
    elevation_map=sobel(gray_image)
    p2, p98 = np.percentile(elevation_map, (2, 98))
    img_rescale = exposure.rescale_intensity(elevation_map, in_range=(p2, p98))
    img_hc = skimage.img_as_ubyte(img_rescale)
    img_hc = np.subtract(255,img_hc)


    #thresholds and convert to byte
    img_t = img_hc > 1
    img_t=skimage.img_as_ubyte(img_t)

    #removes noise and converts to boolean
    img_dn=skimage.morphology.area_closing(img_t,area_threshold=5000,connectivity=1)
    img_dn=skimage.img_as_bool(img_dn)

    #crop image
    y1,y2=0,0
    x1,x2=0,0
    for i in range(img_dn.shape[1]):
        arr=[]
        flag = False
        for j in range(img_dn.shape[0]):
            if(not img_dn[j][i]):
                arr.append(j)
                flag=True
        if(flag):
            y1=int(sum(arr)/len(arr))
            x1=i
            break
    for i in range(img_dn.shape[1]-1,0,-1):
        arr=[]
        flag = False
        for j in range(img_dn.shape[0]):
            if(not img_dn[j][i]):
                arr.append(j)
                flag=True
        if(flag):
            y2=int(sum(arr)/len(arr))
            x2=i
            break
    y_c=int((y1+y2)/2)
    y_c=y1#-------------
    img_c=img_dn[0:y_c,0:img_dn.shape[1]]

    #fill small holes
    img_c_i=np.empty(img_c.shape, dtype=np.int8)
    for i in range(img_c.shape[0]):
        for j in range(img_c.shape[1]):
            if(not img_c[i][j]):
                img_c_i[i][j]=1
            else:
                img_c_i[i][j]=0
    img_fh=sp.ndimage.binary_fill_holes(img_c_i)

    #run binary erosion
    img=skimage.morphology.binary_erosion(img_fh)
    img=skimage.morphology.binary_erosion(img)
    img=skimage.morphology.binary_erosion(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j]=not img[i][j]

    #finds coordinates of black points
    x=[]
    y=[]
    for j in range(int(img.shape[1]/2),img.shape[1]):
        for i in (range(img.shape[0])):#rev
            if(img[i][j]==False):
                y.append(img.shape[0]-i)
                x.append(j)
                break
    for j in reversed(range(int(img.shape[1]/2))):
        for i in (range(img.shape[0])):#rev   
            if(img[i][j]==False):
                y.append(img.shape[0]-i)
                x.append(j)
                break

    #creates polynomial model
    coeff=np.polyfit(x,y,10)
    curve=np.poly1d(coeff)
    line=np.linspace(min(x),max(x),1000)
    
    #slopes and angles of contact
    x1,x2=min(x),max(x)
    diff=np.poly1d(np.polyder(coeff,m=1))
    slopes=[diff(x1),diff(x2)]
    angle_of_contact_rad=np.arctan(np.abs(slopes))
    angle_of_contact_deg=angle_of_contact_rad*180/np.pi

    #tangents
    tan_1=np.poly1d([slopes[0],curve(x1)-x1*slopes[0]])
    line_1=np.linspace(x1,x1+200,1000)
    tan_2=np.poly1d([slopes[1],curve(x2)-x2*slopes[1]])
    line_2=np.linspace(x2-200,x2,1000)

    #displays in a plot
    im = plt.imread(path)
    #im=img
    fig=plt.figure(dpi=100)#,figsize=(5,5)
    plt.title('Contact Angles are ' + str(round(angle_of_contact_deg[0],2)) + ' & ' + str(round(angle_of_contact_deg[1],2)), backgroundcolor='white')
    plt.axis([0,img.shape[1],y_c-im.shape[0],im.shape[0]])
    plt.axis('off')
    #plt.grid()
    plt.imshow(im, extent=[0,im.shape[1], y_c-im.shape[0],y_c])
    #plt.imshow(im, extent=[0,im.shape[1], 0,im.shape[0]])
    plt.plot(line, curve(line), color='r', linewidth=0.5)
    plt.plot(line_1, tan_1(line_1), color='blue', linewidth=1)
    plt.plot(line_2, tan_2(line_2), color='blue', linewidth=1)
    #plt.scatter(x,y, s=0.01, marker='.',color='g')
    plt.scatter([min(x),max(x)],[curve(min(x)),curve(max(x))], s=5, marker='.',color='y')#---
    #plt.show()
    fig.savefig(dest_file,transparent=False,dpi=1000)

    #skimage.io.imshow(img)
#     skimage.io.imsave(fname="Processed_F_dn.png",arr=img_dn)
#     skimage.io.imsave(fname="Processed_F_cr.png",arr=img_c)
#     skimage.io.imsave(fname="Processed_F_fh.png",arr=img_fh)
#     skimage.io.imsave(fname="Processed_F_er.png",arr=img)

import os

files = [f for f in os.listdir('.') if os.path.isfile(f)]
ctr=0
for f in files:
    if f.endswith('.jpg'):
        contact_angle(f, f.split('.')[0]+"_Processed.png")
        ctr=ctr+1
        print(str(ctr)+' done!')
