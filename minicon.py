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

pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")
strTime = ["เช้า","กลางวัน","เย็น","ก่อนนอน","ก่อนอาหาร","หลังอาหาร","หลังอาหารเช้าทันที","หลังอาหารเช้า"]
isEatingBefore = False
_isEatBreakfast = False
_isEatLunch = False
_isEatDinner = False
_isEatBedTime =False 

def iterative_levenshtein(s, t, costs=(1, 1, 1)):
    """ 
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings 
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein 
        distance between the first i characters of s and the 
        first j characters of t
        
        costs: a tuple or a list with three integers (d, i, s)
               where d defines the costs for a deletion
                     i defines the costs for an insertion and
                     s defines the costs for a substitution
    """
    rows = len(s)+1
    cols = len(t)+1
    deletes, inserts, substitutes = costs
    
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = row * deletes
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = col * inserts
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = substitutes
            dist[row][col] = min(dist[row-1][col] + deletes,
                                 dist[row][col-1] + inserts,
                                 dist[row-1][col-1] + cost) # substitution
    # for r in range(rows):
    #     print(dist[r])
    

    return dist[row][col]

def tsplit(string, delimiters):
    """Behaves str.split but supports multiple delimiters."""
    
    delimiters = tuple(delimiters)
    stack = [string,]
    
    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i+j, _substring)
            
    return stack

def cvt_to_JSON(_isPeriod, _isEatBefore,_isEatBreakfast, _isEatLunch, _isEatDinner, _isEatBedTime, _isRoutine, _periodHour) :
    output = {}
    output["isPeriod"] = _isPeriod
    data = {}
    data["isEatingBefore"] = _isEatBefore
    data["isEatBreakfast"] = _isEatBreakfast
    data["isEatLunch"] = _isEatLunch
    data["isEatDinner"] = _isEatDinner
    data["isEatBedTime"] = _isEatBedTime
    output["data"] = data
    conv_json = json.dumps(output, ensure_ascii = False)
    print(conv_json)

def text_from_image_file(image_name,lang):
    output_name = "OutputImg"
    return_code = subprocess.call(['tesseract',image_name,output_name,'-l',lang,'-c','preserve_interword_spaces=1','--psm','7'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    d = open(output_name+'.txt','r',encoding='utf-8')
    str_read = d.read()
    # char_to_remove = temp.split()
    # char_to_remove = re.findall(pattern, temp)
    
    temp = tsplit(str_read,(',', '/', '-', '=',' '))
    ouput = []
    for idx in temp :
        char_to_remove = re.findall(pattern, idx)

        list_with_char_removed = [char for char in idx if not char in char_to_remove]

        
        if len(''.join(list_with_char_removed)) != 0 :
           ouput = ouput + [''.join(list_with_char_removed)]
    return ouput

def check_str(result,txts) :
    global _isEatBreakfast
    global _isEatLunch
    global _isEatDinner
    global isEatingBefore
    global _isEatBedTime
    for txt in txts :
        check_cond = [ ((iterative_levenshtein(idx,txt) <= math.floor((len(idx)-1)/2) ) or txt.find(idx) >= 0) for idx in strTime ]
        if check_cond[0] :
            # print(result)
            # if result == 0 :
            _isEatBreakfast = True
        elif check_cond[1] :
            # print(result)
            # if result == 0 :
            _isEatLunch = True
        elif check_cond[2] :
            # print(result)
            # if result == 0 :
            _isEatDinner = True
        elif check_cond[3] :
            # print(result)
            # if result == 0 :
            _isEatBedTime = True
        elif check_cond[4] :
            # print(result)
            # if result == 0 :
            isEatingBefore = True
        elif check_cond[5] :
            # print(result)
            # if result == 0 :
            isEatingBefore = False
        elif check_cond[6] :
            # print(result)
            # if result == 0 :
            isEatingBefore = False
            _isEatBreakfast = True
        elif check_cond[7] :
            # print(result)
            # if result == 0 :
            isEatingBefore = False
            _isEatBreakfast = True

image = cv2.imread("datatest/418+33.png" , 0) 

image = imutils.resize(image, height=80)
Rem = image.copy()

top,bottom,left,right = [20]*4

blurred = cv2.GaussianBlur(image, (3 , 3), 0)
edged = cv2.Canny(blurred, 50, 200, 255)
kernel = np.ones((2,6),np.uint8)
im = cv2.dilate(edged,kernel,iterations = 3)

img_with_bordor = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
# cv2.imshow("im",img_with_bordor)
# cv2.waitKey(0)
contourmask,contours,hierarchy = cv2.findContours(img_with_bordor,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

hog = cv2.HOGDescriptor((80, 80),(80, 80),(80, 80),(80, 80),40)
features_train = np.load("./version1/features_train0.npy")
label_train = np.load("./version1/label_train0.npy")
knn = cv2.ml.KNearest_create()
knn.train(features_train,cv2.ml.ROW_SAMPLE,label_train)

Rem = cv2.copyMakeBorder(Rem, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
# cv2.imshow("re",Rem)
# cv2.waitKey(0)
# sys.exit()
for cnt in contours[0:] :
    
    x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(Rem , (x-10,y-18) , (x+w+13,y+h+4) , (0,0,255) , 2)

    im = Rem[y:y+h,x:x+h+4]
    # cv2.imwrite( "hog" + str(w) + "+" +  str(h) + ".png" , im)
    # cv2.imshow("rem", im)
    # cv2.waitKey(0)
    im = cv2.resize(im, (80, 80))
    image = cv2.medianBlur(im,9)
    im = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    ho = hog.compute(im)
    data_train = ho.reshape(1,-1)
    _,result,_,_ = knn.findNearest(data_train,4)
    print(result)
    
    if result[0][0] == 0 :
    
        if( h / w > 0.52) :
            roi = Rem[y:y+h, int((x+h)*0.8):x+w]
        else :
            roi = Rem[y:y+h, int((x+h)*0.9):x+w]

        #checkhog

        # roi = Rem[y-18:y+h, x-10:x+w+13]
        # cv2.imshow("roi",roi)
        # cv2.imshow("im", im)
        # cv2.waitKey(0)
        cv2.imwrite( str(w) + "+" +  str(h) + ".png" , roi)
        txts = text_from_image_file( str(w) + "+" +  str(h) + ".png" ,'tha')
        # os.remove(str(w) + "+" +  str(h) + ".png")
        print(txts) 
        check_str(result,txts)

# cv2.imshow("test",Rem)
# cv2.waitKey(0)
cvt_to_JSON(False, isEatingBefore,_isEatBreakfast, _isEatLunch, _isEatDinner, _isEatBedTime, False, "_periodHour")



