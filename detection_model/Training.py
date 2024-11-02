# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 07:43:28 2022

@author: Sixtus
"""


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow as tf


from sklearn.model_selection import train_test_split

import pandas
from tqdm import tqdm
#-----------------------------------------
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import numpy as np
from glob import glob


tf.config.list_physical_devices('GPU')
# re-size all the images to this
IMAGE_SIZE = [150, 150]

train_path = 'Dataset/train'
valid_path = 'Dataset/val'



# Here we will be using inceptionv3 and  imagenet weights to train the dataset

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet',
                        include_top=False)
# don't train existing weights
for layer in inception.layers:
    layer.trainable = False
    

# usin global to get our dataset for training  number of output classes
folders = glob('Dataset/train/*')

# our layer
x = Flatten()(inception.output)

#output layer
prediction = Dense(len(folders), activation='softmax')(x)


# create a model object
model = Model(inputs=inception.input, outputs=prediction)

# viewing  the structure of the model and save as text file
with open('modelsummary.txt', 'w') as f:

    model.summary(print_fn=lambda x: f.write(x + '\n'))



#model.summary()



# Compile the model using optimization method Adams
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# Using Image Data Generator to import the images from the 
# Data augmentation start here
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validat_datagen = ImageDataGenerator(rescale = 1./255)


#  providing the target size and batch size for the dataset as image size
training_set = train_datagen.flow_from_directory('Dataset/train',
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

val_set = validat_datagen.flow_from_directory('Dataset/val',
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fit the model
r = model.fit(
  training_set,
  validation_data=val_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(val_set)
)

# saving  model as a h5 file to disk

model.save('model/cabbage_model_inception.h5')
print("model save to disk")


## ploting grpah for loss and accuracy
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='Train acc')
plt.plot(r.history['val_accuracy'], label='Validation acc')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
