#!/usr/bin/env python
# coding: utf-8

# In[1]:


#debug mode
#set to 1 to print addtional info
debug_mode=1


#imports

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
print("Done loading imports")

#other info

Superclasses=["aquatic_mammals","fish","flowers","food_containers","fruit_and_vegetables","household_electrical_devices",
              "household_furniture","insects","large_carnivores","large_man-made_outdoor_things","large_natural_outdoor_scenes",
              "large_omnivores_and_herbivores","medium-sized_mammals","non-insect_invertebrates","people","reptiles",
              "small_mammals","trees","vehicles_1","vehicles_2"]
classes=[['beaver', 'dolphin', 'otter', 'seal', 'whale'],
         ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
         ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
         ['bottles', 'bowls', 'cans', 'cups', 'plates'],
         ['apples', 'mushrooms', 'oranges', 'pears', 'sweet_peppers'],
         ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
         ['bed', 'chair', 'couch', 'table', 'wardrobe'],
         ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
         ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
         ['bridge', 'castle', 'house', 'road', 'skyscraper'],
         ['cloud', 'forest', 'mountain', 'plain', 'sea'],
         ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
         ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
         ['crab', 'lobster', 'snail', 'spider', 'worm'],
         ['baby', 'boy', 'girl', 'man', 'woman'],
         ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
         ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
         ['maple', 'oak', 'palm', 'pine', 'willow'],
         ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
         ['lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']]

epoch_running_count=0

#checking the data format

#datasets.cifar100.load_data()
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
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
    


#normalising the brightness data

x_train=x_train/255
x_test=x_test/255
print("Done normalising")

#convolutional neural network model 2

print("Training model")

#optimizer settings
opt = tf.keras.optimizers.Adam(
    learning_rate=0.0005,
    beta_1=0.8,
    beta_2=0.99,
    epsilon=1e-06,
    amsgrad=True,
    name='Adam',
)

model2=models.Sequential([
    
    #CNN layers
    layers.Conv2D(filters=64,activation="elu", kernel_size=(3,3), padding='same', input_shape=(28, 28, 1)),
    #layers.MaxPooling2D((2,2)),   #add back this pooling layer to improve performance at the cost of accuracy
    layers.Conv2D(filters=64,activation="elu", kernel_size=(3,3)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(filters=128,activation="elu", padding='same', kernel_size=(3,3)),
    layers.Conv2D(filters=128,activation="elu", kernel_size=(3,3)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.50),
    
    #layers.Conv2D(filters=256,activation="elu", padding='same', kernel_size=(3,3)),
    #layers.Conv2D(filters=256,activation="elu", kernel_size=(3,3)),
    #layers.MaxPooling2D((2,2)),
    #layers.Dropout(0.50),
    
    
    #Dense layers
    layers.Flatten(), #shapeing not needed in the middle
    #layers.Dense(256, activation="elu"),
    #layers.Dropout(0.50),
    layers.Dense(128, activation="elu"), #300 * 0.75 = 225
    layers.Dropout(0.50),  #adjust the layer before dropout to account for the number of nodes droped
    #layers.Dense(200, activation="elu"),
    layers.Dense(10, activation="softmax")  #sigmoid replaced with softmax
    ])

model2.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


intial_epoch_count=5  #number of epochs to be run automaticaly

epoch_running_count+=intial_epoch_count

model2.fit(x_train, y_train, epochs=intial_epoch_count) # add back in to validate while training 
#validation_data=(x_test,y_test)

#running test data

model2.evaluate(x_test, y_test)

#classification report

y_pred = model2.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Number of epochs:" + str(epoch_running_count))
print("Classification Report: \n", classification_report(y_test, y_pred_classes))

def continue_training(epoch_count=5):
    global epoch_running_count
    
    epoch_running_count+=epoch_count

    model2.fit(x_train, y_train, epochs=epoch_count)  #validation_data=(x_test,y_test)

    model2.evaluate(x_test, y_test)

    y_pred = model2.predict(x_test)

    y_pred_classes = [np.argmax(element) for element in y_pred]

    print("Number of epochs:" + str(epoch_running_count))

    print("Classification Report: \n", classification_report(y_test, y_pred_classes))
    
#continue_training() #use this in console to continue the training / arg1 is the number of epochs to be run default 


# In[2]:


continue_training(5)


# In[ ]:




