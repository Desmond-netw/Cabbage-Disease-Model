# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 18:47:33 2022

@author: Sixtus
"""

#import necessary libraries
#from flask import Flask, render_template, request, redirect
from __future__ import division, print_function

# import Flask and Werkzeug server for the flask
from flask import Flask, redirect, url_for, request, render_template, Response
from werkzeug.utils import secure_filename

# import 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import datetime, time
import tensorflow as tf
import keras

from threading import Thread
tf.keras.utils.save_img

#import keras
#from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array

# GPUs configurations 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.confi.experimental.set_memory_growth(gpu, True)
    
tf.config.list_physical_devices('GPU')


#loading the cabbage model
model = load_model('model/cabbage_model_inception.h5')
print(model)

print('@@ model loaded successfull')


def pred_cab_diseas(cabbage_plant, model):
    test_image = image.load_img(cabbage_plant, target_size = (150,150)) # load image 
    print("@@ Got image for prediction")

    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis= 0) # change dimention 3D TO 4D

    result = model.predict(test_image) # predict diseased cabbage or not
    print ('@@ Raw result = ', result)

    pred = model.predict(test_image)
    pred = np.argmax(result, axis = 1) #get the index of max value
    print (pred)
    if pred == 0:
        pred= " Alternaria spot disease", 'Cabbage-Altenaria-spot.html' # if index 0 burned leaf
    elif pred == 1:
        pred = " Black rot disease", 'Cabbage-Black-rot.html' ## if index 1
    elif pred == 2:
        pred = "Healthy Cabbage Crop", 'Cabbage-healthy.html' #if index 2 fresh leaf
    elif pred == 3:
        pred = "Unidentify Object", 'no_disease.html' # if index 3
    
    return pred
#------------------>> pred_cabbage_disease <<-- end

#------------------>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Camera >>>>>>.


## Creating Flask Instances

app = Flask(__name__)

#render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


    
# get input image from client then predict base on the  respective .html page for solution
@app.route("/predict",methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image'] # fet input
        
        filename = file.filename
        print("@@ Input Posted = ", filename)
        
        file_path = os.path.join('static/user uploads',secure_filename(file.filename))
        file.save(file_path)
        
        print("@@ Predicting class......")    
        
        
        # Make prediction
        pred, output_page = pred_cab_diseas(file_path, model)
        result=pred
        return render_template(output_page, pred_output=result, user_image = file_path)
    return None
        

    #for local system & cloud
if __name__ == "__main__":
    app.run(threaded= True, debug = False ,host='192.168.20.101' , port= 9090)
    #host='192.168.20.101'
   