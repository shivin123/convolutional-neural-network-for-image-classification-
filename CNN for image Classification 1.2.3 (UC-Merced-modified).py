#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Imports needed
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from IPython.display import Image as image_jn

img_height = 28
img_width = 28
batch_size = 10

model = keras.Sequential(
    [
    
    #CNN layers
    layers.Conv2D(filters=128,activation="elu", kernel_size=(3,3), padding='same', input_shape=(28,28,3)),
    layers.Conv2D(filters=128,activation="elu", kernel_size=(3,3)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.50),

    #layers.Conv2D(filters=64,activation="elu", kernel_size=(3,3)),
    #layers.Conv2D(filters=64,activation="elu", kernel_size=(3,3)),
    #layers.MaxPooling2D((2,2)),
    #layers.Dropout(0.50),
        
    #Dense layers
    layers.Flatten(), #shapeing not needed in the middle
    layers.Dense(128, activation="elu"), #300 * 0.75 = 225
    layers.Dropout(0.50),  #adjust the layer before dropout to account for the number of nodes droped
    layers.Dense(32, activation="elu"),
    layers.Dense(12, activation="softmax")  #sigmoid replaced with softmax
    ])


ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/cheng/OneDrive/Desktop/pyhton flie/Image processing/CNN/UC-Merced-Modified-2/", #replace with path as approprate
    labels="inferred",
    label_mode="categorical",  # categorical, binary, int
    # class_names=['0', '1', '2', '3', ...]
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=231,
    validation_split=0.1,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/Users/cheng/OneDrive/Desktop/pyhton flie/Image processing/CNN/UC-Merced-Modified-2/", #replace with path as approprate
    labels="inferred",
    label_mode="categorical",  # categorical, binary, int
    # class_names=['0', '1', '2', '3', ...]
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)

self_test = Image.open("C:/Users/cheng/OneDrive/Desktop/pyhton flie/Image processing/CNN/SAT-test-data/agri-1.png")

AUTOTUNE = tf.data.AUTOTUNE

normalization_layer = layers.Rescaling(1./255)

ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))

ds_train = ds_train.cache().prefetch(buffer_size=AUTOTUNE)

ds_validation = ds_validation.map(lambda x, y: (normalization_layer(x), y))

opt = tf.keras.optimizers.Adam(
    learning_rate=0.0005,
    beta_1=0.8,
    beta_2=0.99,
    epsilon=1e-04,
    amsgrad=True,
    name='Adam',
)


model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['Accuracy'])

print("Training")
model.fit(ds_train, epochs=5, verbose=1)

model.evaluate(ds_validation)

model.summary()

model.fit(ds_train, epochs=5, verbose=1)

model.evaluate(ds_validation)







