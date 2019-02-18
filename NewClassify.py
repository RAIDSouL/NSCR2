import cv2
import sys
import re
import os
import json
import imutils
import numpy as np
import subprocess
from imutils import contours
from imutils.perspective import four_point_transform

def main() :
    count = 1
    for picture_id in range (4,7):
        
        im = cv2.imread(".//Check Dataset//" + str(picture_id) + ".jpg" , 0)
        im = imutils.resize(im, height=500)
        Rim = im.copy()
        im = cv2.medianBlur(im,9)
        im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
        im = cv2.GaussianBlur(im, (7 , 7), 0)
        im = cv2.Canny(im, 50, 200, 255)
        kernel = np.ones((3,15),np.uint8)
        im = cv2.dilate(im,kernel,iterations = 1)
        kernel = np.ones((1,30),np.uint8)
        im = cv2.erode(im,kernel,iterations = 1)
        _,contours,_ = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for cny in contours[0:] :
            x, y, w, h = cv2.boundingRect(cny)
            cv2.rectangle(Rim , (x,y) , (x+w,y+h) , (0,0,255) , 2)
        cv2.imshow(str(count) , Rim)


        count = count+1

        
    cv2.waitKey(0)

main()