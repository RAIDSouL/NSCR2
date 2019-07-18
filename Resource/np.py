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
strTime = ["sad","life","เย็น","ก่อนนอน","ก่อนอาหาร","หลังอาหาร","หลังอาหารเช้าทันที","หลังอาหารเช้า"]

hog = cv2.HOGDescriptor((80, 80),(80, 80),(80, 80),(80, 80),40)
features_train = np.load("features_train0.npy")
label_train = np.load("label_train0.npy")
knn = cv2.ml.KNearest_create()
knn.train(features_train,cv2.ml.ROW_SAMPLE,label_train)

im = cv2.imread("hog118+50.png" , 0)
im = cv2.resize(im, (80, 80))
image = cv2.medianBlur(im,9)
im = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
ho = hog.compute(im)
data_train = ho.reshape(1,-1)
_,result,_,_ = knn.findNearest(data_train,3)

# svm = cv2.ml.SVM_create()
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.train(features_train,cv2.ml.ROW_SAMPLE,label_train)
# result = svm.predict(data_train)
print(result)