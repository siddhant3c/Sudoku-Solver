import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from utils import *
from config import *
from preprocess import *

images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)

for x in range(0, noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
print(" ")

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)

#### SPLITTING THE DATA
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses), numOfSamples)
plt.title("Number of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()

img = preProcessing(X_train[30])
img = cv2.resize(img, (300,300))

plt.imshow(img, cmap= "gray")
plt.show()

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

## Adding Depths
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 
                                    X_validation.shape[2], 1)
X_train.shape # :=(6502, 32, 32, 1) ## All are Grayscale Images

dataGen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10
                            )
dataGen.fit(X_train)

## One Hot Encoding
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)