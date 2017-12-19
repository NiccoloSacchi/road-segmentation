#IMPORTATIONS
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
#this limits the use of the graphics card to make the training more stable
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

import matplotlib.pyplot as plt
from datetime import datetime

# keras functions
from keras import callbacks
from keras import preprocessing
from keras.preprocessing import image

# our libraries
from preprocessing import *

from cnn_models import *
from evaluate import *

import scipy as scipy
from mask_to_submission import *

#LOADING THE DATA
n = 100
imgs, gt_imgs = load_images(n)

#SPLITING IT BETWEEN TEST AND TRAIN
test_ratio = 0.04
train, test = split_train_test(imgs, gt_imgs, test_ratio=test_ratio, seed=1)

#LOADING A NEW MODEL 
folder_name = "model_"+str('{0:%Y-%m-%d_%H%M%S}'.format(datetime.now()))
model_path = "..\\models\\"+folder_name
model = CnnModel(model_n=7, model_path=model_path)

#TRAINING FOR 1000 EPOCHS
num_epochs=10
batch_size=4
_ = model.train(train, test=test, num_epochs=num_epochs, batch_size=batch_size, monitor='loss') 

#SAVING THE MODEL
model.save()

#PREDICTING THE UNKNOWN SET AND CREATING THE SUBMISSION FILE
model.predict_augmented_and_export()