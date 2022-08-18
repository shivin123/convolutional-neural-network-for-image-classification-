#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#debug mode
#set to 1 to print addtional info
debug_mode=1
#frog
#set to 1 to see a frog
frog=0

#imports

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
print("Done loading imports")

#other info

classes=["airplane","automobile", "bird","cat","deer","dog","frog","horse","ship","truck"]

#checking the data format

datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print("Checking data format")
if debug_mode==1:
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    y_train=y_train.reshape(-1,)
    print(y_train[:4])
print("Done checking data format")

#show image

def plot_sample(x, y, index):
    plt.figure(figsize=(30,4))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])
    
if debug_mode==1:
    print("Image sample")
    if frog==1:
        plot_sample(x_train, y_train, 0)

#normalising the brightness data

x_train=x_train/255
x_test=x_test/255
print("Done normalising")

#convolutional neural network model 2

print("Training model")
model2=models.Sequential([
    
    #CNN layers
    layers.Conv2D(filters=32,activation="relu", kernel_size=(3,3), input_shape=(32,32,3)),
    #layers.MaxPooling2D((2,2)),   #add back this pooling layer to improve performance at the cost of accuracy
    layers.Conv2D(filters=64,activation="relu", kernel_size=(3,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=128,activation="relu", kernel_size=(3,3)),
    layers.MaxPooling2D((2,2)),
    #Dense layers
    layers.Flatten(), #shapeing not needed in the middle 
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax")  #sigmoid replaced with softmax
    ])

model2.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model2.fit(x_train, y_train, epochs=5)

#running test data

model2.evaluate(x_test, y_test)

#classification report

y_pred = model2.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))

# In[ ]:




