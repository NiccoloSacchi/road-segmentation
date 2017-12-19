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

#TRAINING THE BASE MODEL FOR 500 EPOCHS
num_epochs=10
batch_size=4
_ = model.train(train, test=test, num_epochs=num_epochs, batch_size=batch_size, monitor='val_loss') 

#ADDING SOME REFINING LAYERS
model.add(
    Conv2D(48, (7, 7),
           padding="same", 
           input_shape=(None, None, 2)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.25)) 

model.add(
    Conv2D(64, (5, 5),
           padding="same"
          ))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.25))

model.add(
    Conv2D(64, (5, 5), 
           padding="same")) 
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.25)) 

model.add(
    Conv2D(2, (5, 5), 
           padding="same"))
model.add(LeakyReLU(alpha=0.1))

model.add(Activation('softmax'))

#STOP THE SYSTEM FROM TRAINING THE FIRST LAYERS ANY FURTHER
for i in range(18):
    model.model.layers[i].trainable = False
    
#AND RE-TRAINING FOR 500 MORE EPOCHS
num_epochs=2
batch_size=4
_ = model.train(train, test=test, num_epochs=num_epochs, batch_size=batch_size, monitor='val_loss') 

#SAVING THE MODEL
model.save()

#GRID-SEARCH THE THRESHOLD GIVING THE BEST RESULTS
pred_rot4 = model.predict_augmented(train.X, n_rotations=4)
true = predictions_to_class(np.array([gt_img_to_Y(y, predict_patch_width=8) for y in train.Y])).flatten()
_,threshold = grid_search_treshold(pred_rot4[:, :, :, 1].flatten(), true)
print("Threshold : "+str(threshold))

#PREDICTING THE UNKNOWN SET AND CREATING THE SUBMISSION FILE
model.predict_augmented_and_export(threshold)
