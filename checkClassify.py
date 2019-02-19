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


def main(argv) :
    strTime = ["เช้า","กลางวัน","เย็น","ก่อนนอน","ก่อนอาหาร","หลังอาหาร","หลังอาหารเช้าทันที","หลังอาหารเช้า"]

    image = cv2.imread(argv[0] , 0) 
    image = imutils.resize(image, height=700)
    Rim = image.copy()
    image = cv2.medianBlur(image,9)
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    blurred = cv2.GaussianBlur(image, (7 , 7), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    kernel = np.ones((3,15),np.uint8)
    im = cv2.dilate(edged,kernel,iterations = 1)
    kernel = np.ones((1,30),np.uint8)
    im = cv2.erode(im,kernel,iterations = 1)

    contourmask,contours,hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    fname = argv[0].split(".")[0]
    datalists = []

    hog = cv2.HOGDescriptor((80, 80),(80, 80),(80, 80),(80, 80),40)
    features_train = np.load("features_train.npy")
    label_train = np.load("label_train.npy")
    knn = cv2.ml.KNearest_create()
    knn.train(features_train,cv2.ml.ROW_SAMPLE,label_train)

    isEatingBefore = False
    _isEatBreakfast = False
    _isEatLunch = False
    _isEatDinner = False
    _isEatBedTime =False 

    for cnt in contours[1:] :
        x, y, w, h = cv2.boundingRect(cnt)
        if(w * h > 1000 and w * h < 8000) :
            # cv2.rectangle(Rim , (x-10,y-18) , (x+w+13,y+h+4) , (0,0,255) , 2)
            roi = Rim[y-18:y+h+4, x-10:x+w+13]
            cv2.imwrite( str(w) + "+" + str(h) + ".png" , roi)
            txt = text_from_image_file( str(w) + "+" + str(h)  + ".png",'tha')

            im = roi[0:im.shape[1],0:im.shape[1]]
            im = cv2.resize(im, (80, 80))
            h = hog.compute(im)
            data_train = h.reshape(1,-1)
            _,result,_,_ = knn.findNearest(data_train,3)

            
            if iterative_levenshtein(strTime[0],txt) <= 2:
                if result == 0 :
                    _isEatBreakfast = True
            if iterative_levenshtein(strTime[1],txt) <= 2:
                if result == 0 :
                    _isEatLunch = True
                
            if iterative_levenshtein(strTime[2],txt) <= 2:
                if result == 0 :
                    _isEatDinner = True
                
            if iterative_levenshtein(strTime[3],txt) <= 2:
                if result == 0 :
                    _isEatBedTime = True
                
            if iterative_levenshtein(strTime[4],txt) <= 2:
                if result == 0 :
                    isEatingBefore = True
                
            if iterative_levenshtein(strTime[5],txt) <= 2:
                if result == 0 :
                    isEatingBefore = False
                
            if iterative_levenshtein(strTime[6],txt) <= 2:
                if result == 0 :
                    isEatingBefore = False
                    _isEatBreakfast = True
                
            if iterative_levenshtein(strTime[7],txt) <= 2:
                if result == 0 :isEatingBefore = False
                    _isEatBreakfast = True

    # cv2.imshow("asdfghjk" , Rim)
    # cv2.waitKey(0)


main(sys.argv[1:])