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
from model import *
from eda import *

model = myModel()
print(model.summary())

###TRAINING THE MODEL
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSizeVal), steps_per_epoch=stepsPerEpochVal, epochs=epochsVal, validation_data=(X_validation, y_validation), shuffle=1)

## Plotting
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')

plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

### SAVING THE MODEL 
model.save('models\my_model_cnn.h5')