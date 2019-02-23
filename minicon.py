import cv2
import sys
import re
import os
import json
import imutils
from IPython.core.debugger import set_trace
import numpy as np
import subprocess
from imutils import contours
from imutils.perspective import four_point_transform
import math

image = cv2.imread("240+39.png" , 0) 
Rem = image.copy()
image = imutils.resize(image, height=150)
blurred = cv2.GaussianBlur(image, (5 , 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)
kernel = np.ones((3,5),np.uint8)
im = cv2.dilate(edged,kernel,iterations = 1)
cv2.imshow("im",im)

contourmask,contours,hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for cnt in contours[1:] :
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(Rem , (x,y) , (x+w,y+h) , (0,0,255) , 2)
    # roi = Rim[y-18:y+h+4, x-10:x+w+13]
    # Reme = roi.copy()
    # Reme = imutils.resize(Reme, height=100)
    # cv2.imwrite( str(w) + "+" +  str(h) + ".png" , roi)
    # txts = text_from_image_file( str(w*h) + ".png" ,'tha')
    # print(txts) 
    # check_str(result,txts)
    # if(w > 230 and w < 250)
    #     image = cv2.imread(str(w) + "+" +  str(h) + ".png" , 0) 

cv2.imshow("rem" , Rem)
cv2.waitKey(0)

