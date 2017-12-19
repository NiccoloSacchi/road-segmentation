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

#LOAD THE DATA
n = 100
imgs, gt_imgs = load_images(n)

#SPLIT IT BETWEEN TEST AND TRAIN
test_ratio = 0.04
train, test = split_train_test(imgs, gt_imgs, test_ratio=test_ratio, seed=1)

#LOAD OUR BEST MODEL AND ITS BEST WEIGHTS
folder_name = "model_2017-12-17_015606"
model_path = "..\models\\"+folder_name
model = CnnModel(model_n = 7,model_path=model_path)
model.load() 
model.load_weights("2017-12-19_165405_best-weightsmodel_leakyrelu_maxpooling.hdf5")

#GRID-SEARCH THE THRESHOLD GIVING THE BEST RESULTS
pred_rot4 = model.predict_augmented(train.X, n_rotations=4)
true = predictions_to_class(np.array([gt_img_to_Y(y, predict_patch_width=8) for y in train.Y])).flatten()
_,threshold = grid_search_treshold(pred_rot4[:, :, :, 1].flatten(), true)
print("Threshold : "+str(threshold))

#PREDICT THE UNKNOWN SET AND CREATING THE SUBMISSION FILE
model.predict_augmented_and_export(threshold)
