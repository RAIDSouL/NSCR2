# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
import imutils
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

count = 1
x = 0
y = 0

hog = cv2.HOGDescriptor((50,50),(50,50),(50,50),(50,50),9)
charlist = "TF"
label_train = np.zeros((100,1))


for char_id in range(0,2):
    for im_id in range(1,21):
        #5 pictures

        #read pictures
        im = cv2.imread(".//Check dataset/"+ charlist[char_id] + "//" + str(im_id) + ".png",0)
        im = im[0:im.shape[1],0:im.shape[1]]
        im = cv2.resize(im, (50, 50))
        im = cv2.GaussianBlur(im, (3, 3), 0)
        h = hog.compute(im)
        if count == 0:
            features_train = h.reshape(1,-1)
        else:
            features_train = np.concatenate((features_train,h.reshape(1,-1)),axis = 0)

        
        # im = imutils.resize(im, height=50)
        # im = im[y:50,x:x+50]
        # cv2.imwrite( ".//Check Dataset/" + str(char_id) + "//" + str(im_id) +".bmp" , im)
        # cv2.imshow(str(count),im)

        #SHow img training set
        # cv2.moveWindow(str(count), 120*im_id, 100*char_id)
        # count = count+1

# cv2.waitKey(0)
# cv2.destroyAllWindows()

