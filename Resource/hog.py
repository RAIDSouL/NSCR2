# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython.core.debugger import set_trace
from sklearn.metrics import confusion_matrix
# count = 0
# charlist = "ABCDF"
# colorlist = ["red","green","blue","olive","cyan"]

# #create hog to do the hog

# #1imges = 1 block
# #(size image 50x50 ),(block size),(cell size = 1 cell = 1 block = all img),(number of bin)
# hog = cv2.HOGDescriptor((50,50),(50,50),(50,50),(50,50),9)

# #WinSize, BlockSize, BlockStride, CellSize, NBins

# label_train = np.zeros((25,1))

# for char_id in range(0,5):
#     for im_id in range(1,6):
#         im = cv2.imread(".//Check Dataset/"+charlist[char_id]+"/"+str(im_id)+".bmp",0)

#         #resize imgs are equal 50x50
#         im = im[0:im.shape[1],0:im.shape[1]]
#         im = cv2.resize(im, (50, 50))

#         cv2.imshow("ttt",im)

#         #from data set that border will have องศา clearly , We have to blur to distribute องศา
#         im = cv2.GaussianBlur(im, (3, 3), 0)
#         h = hog.compute(im)

        # if count == 0:
        #     features_train = h.reshape(1,-1)
        # else:
        #     features_train = np.concatenate((features_train,h.reshape(1,-1)),axis = 0)

#         #will get 9 features and 9 row datas

#         #final will get label train
#         label_train[count] = char_id
        
#         count = count+1
#         plt.figure(char_id)
#         plt.plot(h, color=colorlist[char_id])
#         plt.ylim(0,1)
#         plt.figure(5)
#         plt.plot(h,color=colorlist[char_id])
#         plt.ylim(0, 1)

#         #KNN
#         # knn = cv2.ml.KNearest_create()
#         # knn.train(features_train,cv2.ml.ROW_SAMPLE,label_train)

#         #SVM
#         # svm = cv2.ml.SVM_create()
#         # svm.setKernel(cv2.ml.SVM_LINEAR)
#         # svm.train(features_train,cv2.ml.ROW_SAMPLE,label_train)

# print(features_train)
# print(label_train)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import cv2

count = 0
#(size image 50x50 ),(block size),(cell size = 1 cell = 1 block = all img),(number of bin)
hog = cv2.HOGDescriptor((80, 80),(32, 32),(16, 16),(16, 16),40)
charlist = "TF"
label_train = np.zeros((160,1))
colorlist = ["red","green"]
n_sets = [83,79]
for char_id in range(0,2):
    for im_id in range(1,n_sets[char_id]):
        #5 pictures

        #read pictures
        image = cv2.imread("../Checkdataset/"+ charlist[char_id] + "//" + str(im_id) + ".png",0)
        # print("../Checkdataset/"+ charlist[char_id] + "//" + str(im_id) + ".png")
        image = image[0:image.shape[0],0:image.shape[0]]
        # cv2.imshow("image" , image)
        # cv2.waitKey(0)
        image = cv2.resize(image, (80, 80))
        # cv2.imwrite(charlist[char_id] + str(im_id) + ".png",image)
        # image = cv2.medianBlur(image,9)
        # image = cv2.GaussianBlur(image, (7 , 7), 0)
        image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
        # image = cv2.medianBlur(image,7)
        # image = cv2.GaussianBlur(image, (5 , 5), 0)
        # blurred = cv2.GaussianBlur(image, (7 , 7), 0)
        # edged = cv2.Canny(blurred, 50, 200, 255)
        # kernel = np.ones((3,3),np.uint8)
        # im = cv2.dilate(edged,kernel,iterations = 1)
        # cv2.imshow("image" , image)
        # cv2.waitKey(0)
        h = hog.compute(image)
        if count == 0:
            features_train = h.reshape(1,-1)
        else:
            features_train = np.concatenate((features_train,h.reshape(1,-1)),axis = 0)
        label_train[count] = char_id
        count = count+1
        # plt.figure(char_id)
        # plt.plot(h, color=colorlist[char_id])
        # plt.ylim(0,1)
        # plt.figure(3)
        # plt.plot(h,color=colorlist[char_id])
        # plt.ylim(0, 1)


#KNN
knn = cv2.ml.KNearest_create()
# set_trace()
label_train = label_train.astype(int)
# print(label_train.shape)
# print(features_train.shape)
knn.train(features_train,cv2.ml.ROW_SAMPLE,label_train)
_,result,_,_ = knn.findNearest(features_train,3)
# print(result)
cm = confusion_matrix(label_train, result)
print(cm)
np.save("features_Sqr_cir_final" , features_train)
np.save("label_Sqr_cir_final" , label_train)
# print(features_train)
# print(label_train)
