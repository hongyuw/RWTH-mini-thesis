import numpy as np
import os
import cv2

DATADIR = "dataset"
CATEGORIES = ['1','2','3','4','5']
IMG_SIZE = 28

#for category in CATEGORIES:
#    path = os.path.join(DATADIR, category)
#    for img in os.listdir(path):
#        #img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
#        img_array = cv2.imread(os.path.join(path,img))
#        break
#    break
#
#print (img_array.shape)
#
#new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
#
#print (new_array.shape)
#
#cv2.imshow('', new_array)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        print ("create_training_data")
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        count = 0
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
                count += 1
                if count > 999:
                    break
            except Exception as e:
                pass

create_training_data()

import random
random.shuffle(training_data)

#for sample in training_data:
#    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
#-1代表图片个数 最后的1代表颜色 
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
#
#pickle_in = open("X.pickle","rb")
#X = pickle.load(pickle_in)
#
#pickle_in = open("y.pickle","rb")
#y = pickle.load(pickle_in)

print(X.shape)  #(2000, 50, 50, 1)
#print(X[1].shape)  #(50, 50, 1)
#print(X[1:].shape)  #(1999, 50, 50, 1)
