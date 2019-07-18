#5-3-62
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
strTime = ["เช้า","กลางวัน","เย็น","ก่อนนอน","ก่อนอาหาร","หลังอาหาร"]
isEatingBefore = False
_isEatBreakfast = False
_isEatLunch = False
_isEatDinner = False
_isEatBedTime =False 
def deepCheck(raw_image,hog,knn):
    # cv2.imshow("asd",raw_image)
    # cv2.waitKey(0)
    # sad = raw_image.copy()
    # cv2.imshow("sda",sad)
    # cv2.waitKey(0)
    blurred = cv2.GaussianBlur(raw_image, (3 , 3), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    kernel = np.ones((2,6),np.uint8)
    im = cv2.dilate(edged,kernel,iterations = 3)
    
    # cv2.imshow("asd",im)
    # cv2.waitKey(0)

    _,contours_2,_ = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours_2 = sorted(contours_2, key=cv2.contourArea, reverse=True)
    
    for cnt_2 in contours_2[0:int(len(contours_2))] :
        x2, y2, w2, h2 = cv2.boundingRect(cnt_2)
        # cv2.rectangle(sad , (x2,y2) , (x2+w2,y2+h2) , (0,0,255) , 2)
        # cv2.imshow("sad",sad)
        # cv2.waitKey(0)
        img_mini_con = raw_image[y2:y2+h2,x2:x2+h2+4]
        img_mini_con = img_mini_con[0:img_mini_con.shape[0],0:img_mini_con.shape[0]]
        img_for_ocr_text = raw_image[y2:y2+h2,x2:x2+w2]
        cv2.imwrite( str(int(w2*h2)) + ".png" , img_for_ocr_text)
        # cv2.imshow("asd",img_for_ocr_text)
        # cv2.imshow("con",img_mini_con)
        # cv2.waitKey(0)
     
        img_mini_con = cv2.resize(img_mini_con, (80,80))
        # img_mini_con = cv2.medianBlur(img_mini_con,9)
        img_mini_con = cv2.adaptiveThreshold(img_mini_con,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
        # cv2.imshow("sdsd" , img_mini_con)
        # cv2.waitKey(0)
        ho = hog.compute(img_mini_con)
        data_train2 = ho.reshape(1,-1)
        _,result2,_,_ = knn.findNearest(data_train2,3)
        # print(result2)
        

        ### only str

        if result2[0][0] == 0 :
            if( h2 / w2 > 0.52) :
                roi = img_for_ocr_text[0:h2, int((h2)*0.7):w2]
                # cv2.imshow("sdss" , roi)
                # cv2.waitKey(0)
            else :
                roi = img_for_ocr_text[0:h2, int((h2)*0.8):w2]
                # cv2.imshow("sdss" , roi)
                # cv2.waitKey(0)
            rand_name_file = np.random.randint(1000 , size = 1)
            cv2.imwrite( str(rand_name_file[0]) + ".png" , roi)
            # set_trace()
            txts2 = text_from_image_file( str(rand_name_file[0]) + ".png" ,'tha')
            # print(txts2)
            os.remove(str(rand_name_file[0]) + ".png")
            check_str(result2[0][0],txts2)


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
    return_code = subprocess.call(['tesseract',image_name,output_name,'-l',lang,'-c','preserve_interword_spaces=1','--psm','6'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

def advance_search_str(i,txt):
    output = []
    for idx in strTime :
        if (len(idx)-i)/2 < 0 :
            output.append(False)
        else :
            if (iterative_levenshtein(idx,txt) <= math.floor((len(idx)-i)/2) ) or txt.find(idx) >= 0 :
                output.append(True) 
            else :
                output.append(False) 
    return output


def check_str(result,txts) :
    global _isEatBreakfast
    global _isEatLunch
    global _isEatDinner
    global isEatingBefore
    global _isEatBedTime

    #Check signed
    if result != 0  :
        return

    for txt in txts :

        # check_cond = [ ((iterative_levenshtein(idx,txt) <= math.floor((len(idx))) ) or txt.find(idx) >= 0) for idx in strTime ]
        # temp = check_cond.copy()

        # max_len_strTime = max([len(i) for i in strTime])
        # for i in range(max_len_strTime-1,-1,-1) :
        #     check_cond = advance_search_str(i,txt)
        #     if (np.sum(check_cond) <= np.sum(temp) ) and np.sum(check_cond) > 0 :
        #         temp = check_cond
        # check_cond = temp

        # if np.sum(check_cond) > 2 :
        #     continue

        # if check_cond[0] :
        #         _isEatBreakfast = True
        # elif check_cond[1] :
        #         _isEatLunch = True
        # elif check_cond[2] :
        #         _isEatDinner = True
        # elif check_cond[3] :
        #         _isEatBedTime = True
        # elif check_cond[4] :
        #         isEatingBefore = True
        # elif check_cond[5] :
        #         isEatingBefore = False
        #         isEatingBefore = False
        check_cond = [ ((iterative_levenshtein(idx,txt) <= math.floor(((len(idx)) /2)) ) or txt.find(idx) >= 0) for idx in strTime ]
        if np.sum(check_cond) > 2 :
            check_cond = [ ((iterative_levenshtein(idx,txt) <= math.floor(((len(idx) -1) /2)) ) or txt.find(idx) >= 0) for idx in strTime ]
        
        if np.sum(check_cond) > 2 :
            continue

        if check_cond[0] :
                _isEatBreakfast = True
        elif check_cond[1] :
                _isEatLunch = True
        elif check_cond[2] :
                # set_trace()
                _isEatDinner = True
        elif check_cond[3] :
                _isEatBedTime = True
        
        check_cond = [ ((iterative_levenshtein(idx,txt) <= math.floor(((len(idx) -1) /2)) ) or txt.find(idx) >= 0) for idx in strTime[-2:] ]
        if np.sum(check_cond) > 1 :
            check_cond = [ ((iterative_levenshtein(idx,txt) <= math.floor(((len(idx) - 3) /2)) ) or txt.find(idx) >= 0) for idx in strTime[-2:] ]
        if np.sum(check_cond) > 1 :
            check_cond = [ ((iterative_levenshtein(idx,txt) <= math.floor(((len(idx) - 5) /2)) ) or txt.find(idx) >= 0) for idx in strTime[-2:] ]


        if check_cond[0] :
                isEatingBefore = True
                
        elif check_cond[1] :
                isEatingBefore = False

def main(argv) :
    try :
        ###imageProcessing###
        Image = cv2.imread(argv[0] , 0)
        Image = imutils.resize(Image , height=700)
        Image = Image[math.floor(Image.shape[0]*0.3):math.floor(Image.shape[0]*0.9),:]
        Image_contour = Image.copy()
        Image_hog = Image.copy()
        Image_padding = Image.copy()

        ###padding parameter###
        top,bottom,left,right = [50]*4

        #Contour
        Image_contour = cv2.medianBlur(Image_contour,9)
        Image_contour = cv2.adaptiveThreshold(Image_contour,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
        blurred = cv2.GaussianBlur(Image_contour, (7 , 7), 0)
        edged = cv2.Canny(blurred, 50, 200, 255)
        kernel = np.ones((3,15),np.uint8)
        Image_contour = cv2.dilate(edged,kernel,iterations = 1)
        kernel = np.ones((1,30),np.uint8)
        Image_contour = cv2.erode(Image_contour,kernel,iterations = 1)
        Image_contour_with_padding = cv2.copyMakeBorder(Image_contour, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        _,contours,_ = cv2.findContours(Image_contour_with_padding,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        Image_padding = cv2.copyMakeBorder(Image_padding, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        datalists = []

        ###HOG###
        hog = cv2.HOGDescriptor((80, 80),(32, 32),(16, 16),(16, 16),40)
        features_train = np.load("./version1/features_Sqr_cir_final.npy")
        label_train = np.load("./version1/label_Sqr_cir_final.npy")
        knn = cv2.ml.KNearest_create()
        knn.train(features_train,cv2.ml.ROW_SAMPLE,label_train)
    
        ###Contour Loop###
        for cnt in contours[0:] :
            
            x, y, w, h = cv2.boundingRect(cnt)
            ################ rectangle if want  #######################
            # cv2.rectangle(Image_padding , (x+h,y-9) , (x+w+13,y+h+4) , (0,0,255) , 2)
            imgsss = Image_padding[y-18:y+h+4 , x-10:x+w+13]
            cv2.imwrite( str(w) + "+" +  str(h) + ".png" , imgsss)
            if w  > 235 :
                Image_mini_con = Image_padding[y-18:y+h+4, x-10:x+w+13]
                # cv2.imshow("sad",Image_mini_con)
                # cv2.waitKey(0)
                Image_mini_con = imutils.resize(Image_mini_con , height = 80)
                Image_mini_con_border = Image_mini_con.copy()

                deepCheck(Image_mini_con,hog,knn)

            else :
                ### Check is image #####
                if w < 2 or h < 2 :
                    continue
                ########## STR ############
                cv2.imwrite( str(w) + "-" +  str(h) + ".png" , Image_padding[y-18:y+h+4 , x-10:x+w+13])
                Image_padding_str = Image_padding[y-9:y+h+4 , x+h:x+w+13]
                cv2.imwrite( str(w) + "+" +  str(h) + ".png" , Image_padding_str)
                txts = text_from_image_file( str(w) + "+" +  str(h) + ".png" ,'tha')
                os.remove(str(w) + "+" +  str(h) + ".png")

                ########## HOG ############
                Image_hog = Image_padding[y-18:y+h+4 , x-10:x+h+4]
                # try : 
                #     cv2.imshow( "Test" , Image_padding[y-5:y+h+4 , x+h:x+w+10])
                #     cv2.waitKey(0)
                # except :
                #     print("Error >> ",x,y,w,h)
                Image_hog = cv2.resize(Image_hog, (80, 80))
                # Image_hog = cv2.medianBlur(Image_hog,9)
                Image_hog = cv2.adaptiveThreshold(Image_hog,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
                ho = hog.compute(Image_hog)
                data_train_hog = ho.reshape(1,-1)
                _,result,_,_ = knn.findNearest(data_train_hog,3)
                # print(txts) 
                # print(result[0][0])
                # if txts == ["ก่อนนอน"] or txts == ["กลางวัน"] :
                #     cv2.imshow("lunch",Image_hog)
                #     cv2.waitKey(0)
                check_str(result[0][0],txts)

    except Exception as e:
        # print(e)
        global isEatingBefore 
        isEatingBefore = False
        global _isEatBreakfast
        _isEatBreakfast = False
        global _isEatLunch 
        _isEatLunch = False
        global _isEatDinner
        _isEatDinner = False 
        global _isEatBedTime
        _isEatBedTime =False
    finally :
        cvt_to_JSON(False, isEatingBefore,_isEatBreakfast, _isEatLunch, _isEatDinner, _isEatBedTime, False, "_periodHour")

main(sys.argv[1:])