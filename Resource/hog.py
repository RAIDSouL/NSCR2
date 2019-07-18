import imutils
import numpy as np
import matplotlib.pyplot as plt
import cv2
from IPython.core.debugger import set_trace
import numpy as np
import cv2

count = 0
hog = cv2.HOGDescriptor((80, 80),(80, 80),(80, 80),(80, 80),40)
charlist = "TF"
label_train = np.zeros((40,1))
colorlist = ["red","green"]

for char_id in range(0,2):
    for im_id in range(1,20):
        #5 pictures

        #read pictures
        im = cv2.imread(".//CheckDataset/"+ charlist[char_id] + "//" + str(im_id) + ".png",0)
        im = im[0:im.shape[1],0:im.shape[1]]
        im = cv2.resize(im, (80, 80))
        h = hog.compute(im)
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
knn.train(features_train,cv2.ml.ROW_SAMPLE,label_train)
_,result,_,_ = knn.findNearest(features_train,3)
print(result)

# np.save("features_train" , features_train)
# np.save("label_train" , label_train)
# print(features_train)
# print(label_train)
