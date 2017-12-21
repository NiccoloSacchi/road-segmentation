import os
import os.path
import glob
import logging

import numpy as np
from scipy.misc import imread
from scipy import ndimage
from scipy.signal import savgol_filter

from mask_to_submission import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imsave

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

from optparse import OptionParser
import sys

import unet

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.cmap'] = 'gray'

unet.config_logger("/dev/null")

def load_training_data():
    # load the training set

    # set the path of the ground truth and the images
    #---------------------------------------------------------------------------
    impath = "/cvlabdata1/cvlab/datasets_deprelle/roadseg/training/images/*.png"
    gtpath = "/cvlabdata1/cvlab/datasets_deprelle/roadseg/training/groundtruth/"
    #---------------------------------------------------------------------------

    # get the list of the images' path
    #---------------------------------------------------------------------------
    filenames = sorted(glob.glob(impath))
    #---------------------------------------------------------------------------

    # initialize images lists
    #---------------------------------------------------------------------------
    images = []
    labels = []
    #---------------------------------------------------------------------------

    for filename in filenames:

        # get the prefix of the images
        #-----------------------------------------------------------------------
        basename = os.path.basename(filename)
        prefix, _ = os.path.splitext(basename)
        #-----------------------------------------------------------------------

        # select the coresponding gt image
        #-----------------------------------------------------------------------
        gt_basename = prefix + ".png"
        #-----------------------------------------------------------------------

        try:
            # read the image
            #-------------------------------------------------------------------
            image = np.float32(imread(filename) / 255.0)
            #-------------------------------------------------------------------

            # read the label and modify its range
            #-------------------------------------------------------------------
            label = np.int64(imread(os.path.join(gtpath, gt_basename)))

            pos = label >= 125
            neg = label < 125
            label[pos] = 1
            label[neg] = 0
            #-------------------------------------------------------------------

            if image.shape[:2] != label.shape:
                continue

            # add the image and the label to the list
            #-------------------------------------------------------------------

            if(prefix != "satImage_091" and prefix != "satImage_092"):
                if(prefix == "satImage_072" or prefix == "satImage_073"):
                    for i in range(8):
                        print(filename)
                        images.append(image)
                        labels.append(label)
                print(filename)
                images.append(image)
                labels.append(label)
            #-------------------------------------------------------------------

        except FileNotFoundError as e:
            print(e)

    return images, labels

def load_test_data():
    # load the test set

    # set the path of the ground truth and the images
    #---------------------------------------------------------------------------
    impath = "/cvlabdata1/cvlab/datasets_deprelle/roadseg/test/*png"
    #---------------------------------------------------------------------------

    # get the list of the images' path
    #---------------------------------------------------------------------------
    filenames = sorted(glob.glob(impath))
    #---------------------------------------------------------------------------

    # initialize images lists
    #---------------------------------------------------------------------------
    images = []
    #---------------------------------------------------------------------------

    for filename in filenames:
        print(filename)
        try:
            #-------------------------------------------------------------------
            image = np.float32(imread(filename) / 255.0)
            #-------------------------------------------------------------------

            # add the images to the list
            #-------------------------------------------------------------------
            images.append(image)
            #-------------------------------------------------------------------

        except FileNotFoundError as e:
            print(e)

    return images


def select_subset(images, labels, num):
    # select a part of the dataset

    # copy the images and labels
    #---------------------------------------------------------------------------
    temp_images = images.copy()
    temp_labels = labels.copy()
    #---------------------------------------------------------------------------

    # create the images / labels list
    #---------------------------------------------------------------------------
    selec_im = []
    selec_la = []
    #---------------------------------------------------------------------------

    for i in range(num):

        # compute the random index
        #-----------------------------------------------------------------------
        rand_index = np.random.randint(len(temp_images))
        #-----------------------------------------------------------------------

        # select the image and the label corresponding to the random index
        #-----------------------------------------------------------------------
        selec_im.append(temp_images[rand_index])
        selec_la.append(temp_labels[rand_index])
        #-----------------------------------------------------------------------

        # delete the selected image/label from the temporary list
        #-----------------------------------------------------------------------
        del temp_images[rand_index]
        del temp_labels[rand_index]
        #-----------------------------------------------------------------------

    return selec_im, selec_la

def recall_precision(prediction, label, thresh):
    # compute the recall and the precision for one thresholded prediction of
    # the network

    # apply a threshold on the prediction image
    #---------------------------------------------------------------------------
    prediction = np.where(prediction >= thresh, 1, 0)
    #---------------------------------------------------------------------------

    # compute the number of predicted positive pixel and the number of ground
    # truth positive pixel
    #---------------------------------------------------------------------------
    predicted_pos = np.sum(prediction)
    gt_pos = np.sum(label)
    #---------------------------------------------------------------------------

    # compute the per element production of the predicted and the ground truth
    # binary images
    #---------------------------------------------------------------------------
    prod = prediction * label
    #---------------------------------------------------------------------------

    # compute the number pixel postive in both the prediction and the ground
    # truth
    #---------------------------------------------------------------------------
    pred_gt_pos = np.sum(prod)
    #---------------------------------------------------------------------------

    # compute the recall and the precision
    #---------------------------------------------------------------------------
    if(predicted_pos == 0):
        precision = 1
        recall    = 0
    else:
        precision = pred_gt_pos/predicted_pos
        recall     = pred_gt_pos/gt_pos
    #---------------------------------------------------------------------------

    return precision, recall

def compute_recall_precision(prediction, labels_test, thresh_step):
    # compute the recall and precision on all the data test set on the with a
    # thresh between 0 and 1

    # initialize the list for the recall and the precision
    #-----------------------------------------------------------------------
    recallLs    = []
    precisionLs = []
    #-----------------------------------------------------------------------

    for i in range(thresh_step+1):

        # compute the thresh
        #-----------------------------------------------------------------------
        thresh = i/thresh_step
        #-----------------------------------------------------------------------

        # initialize the recall and the precision
        #-----------------------------------------------------------------------
        recall = precision = 0
        #-----------------------------------------------------------------------

        for i in range(len(images_test)):

            # compute the recall and the precision for the prediction
            #-------------------------------------------------------------------
            r , p = recall_precision(prediction[i],labels_test[i],thresh)
            #-------------------------------------------------------------------

            # add the the sum for the mean computation
            #-------------------------------------------------------------------
            recall    = r + recall
            precision = p + precision
            #-------------------------------------------------------------------

        # compute the mean of the recall and the precision
        #-----------------------------------------------------------------------
        recallLs.append(recall / len(images_test))
        precisionLs.append(precision / len(images_test))
        #-----------------------------------------------------------------------

    return recallLs, precisionLs


def draw_prediction(predictions,images):

    if not os.path.exists("results"):
        os.makedirs("results")

    c = np.zeros(images[0].shape)

    for i in range(len(predictions)):
        pred = predictions[i]
        im   = images[i]

        c = np.zeros(im.shape)
        c[:,:,1] = pred

        mask = c > 0.5
        im = im + c
        im[mask] = im[mask]/2

        plt.imsave("results/"+str(i)+"I.png",im)



# initialize the variables
#-------------------------------------------------------------------------------
num_iter    = 15000
save_every  = 5000
print_every = 100
num_points  = 5000
save_path   = 'savedmodels_roadseg_cntr'
#-------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# read the data sets
#-------------------------------------------------------------------------------
logger.info("Loading training set :")
images, labels = load_training_data()
logger.info("Loading test set : ")
tests           = load_test_data()
#-------------------------------------------------------------------------------

# create the classifier
#-------------------------------------------------------------------------------
logger.info("New road segmentatiob training is starting ")
logger.info("Result will be saved in : " + save_path)
unet_config = unet.UNetConfig(steps=5,
                              num_input_channels=3,
                              border_mode='same')

unet_clsf   = unet.UNetClassifier(unet_config).cuda()

trainer, sampler, solver = unet.my_setup(unet_clsf,
                                         images,
                                         labels,
                                         batch_size = 4,
                                         save_path=save_path,
                                         save_every=save_every,
                                         learning_rate=1e-4)
#-------------------------------------------------------------------------------

# train the network
#-------------------------------------------------------------------------------
trainer.train(num_iter, print_every=print_every)
#-------------------------------------------------------------------------------
