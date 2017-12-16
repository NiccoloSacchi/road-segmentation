#matplotlib inline

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
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

#%load_ext autoreload
#%autoreload 2

# Loaded a set of images
n = 100
imgs, gt_imgs = load_images(n)

# !!! set predict_patch_width in accordance to the model you are using !!!
# the shape of the output of the model depends on the strides parameters 
# (if a layer has stride=2 then each ouput's side is half of the input'side).
# predict_patch_width must be equal to the total reduction of the model, e.g.
# if the model has three layer with stride=2 => the input of the model is 
# reduced by a factor of 2*2*2=8, i.e. the ouptut will be patch-wise with 
# patches 8x8 pixels.
predict_patch_width = 8

X, Y = images_to_XY(imgs, gt_imgs, predict_patch_width=predict_patch_width)

set_ = SimpleNamespace()
set_.X = X
set_.Y = Y

#split the dataset
test_ratio = 0.2

train, test = split_train_test(X, Y, test_ratio=test_ratio, seed=1)
train.X.shape, train.Y.shape, test.X.shape, test.Y.shape 

folder_name = "model_"+str('{0:%Y-%m-%d_%H%M%S}'.format(datetime.now()))


#models = [
#0          model1,
#1          model2,
#2          additional_conv_layer_model,
#3          max_pooling_model,
#4          leaky_relu_model,
#5          decreased_dropout_model,
#6          many_filters_model, 
#7          model_leakyrelu_maxpooling, 
#8          model_relu_maxpooling] 
models_to_train =  [7,8]
set_ = test
batch_sizes =[4,4,4,4,4,4,4,4,4]
for i in models_to_train:
    # generate an unique name for the model (so to avoid overwriting previous models)
    model_path = "..\\models\\"+folder_name
    model = CnnModel(model_n=i, model_path=model_path)
    model.summary()
    num_epochs=2
    batch_size=4
    _ = model.train(train, test=test, num_epochs=num_epochs, batch_size=batch_size, monitor='val_loss') 
    model.save()
    model.plot_history()