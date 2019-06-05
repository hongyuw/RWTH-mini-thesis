#This file loads the data from dataset folder, then do augmentation and create training data
from mxnet import nd
from mxnet.gluon import data as gdata
import numpy as np
import cv2
import os

#directory and parameters
DATADIR = "dataset"  #main folder
CATEGORIES = ['1','2','3','4','5']  #sub folder
IMG_SIZE = 100   #before crop
img_size = 28    #after crop

#augmentation methods
crop = gdata.vision.transforms.RandomResizedCrop(
    (img_size, img_size), scale=(0.8, 1), ratio=(0.9, 1.1))
flip = gdata.vision.transforms.RandomFlipLeftRight()
updown = gdata.vision.transforms.RandomFlipTopBottom()
bright = gdata.vision.transforms.RandomBrightness(0.5)
augs = gdata.vision.transforms.Compose([
    crop,flip,updown,bright])

#create training data and put them in a list
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
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                for j in range(70):
                    new_array = nd.array(img_array)
                    new_array = augs(new_array)
                    new_array = new_array.asnumpy()/255.0   
                    training_data.append([new_array, class_num])
                count += 1
                if count > 70:
                    break
            except Exception as e:
                pass

create_training_data()

import random
random.shuffle(training_data)

#extract data in the list and separate features and label in two pickle files
X = []  #feature
y = []  #label

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, img_size, img_size, 3) #-1 represents data amount

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

print(X.shape)  #(2000, 50, 50, 1)