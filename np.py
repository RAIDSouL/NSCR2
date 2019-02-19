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
features_train = np.load("features_train.npy")
label_train = np.load("label_train.npy")
print(features_train)
print(label_train)

#KNN
knn = cv2.ml.KNearest_create()
# set_trace()
knn.train(features_train,cv2.ml.ROW_SAMPLE,label_train)
_,result,_,_ = knn.findNearest(features_train,3)
print(strTime[3])