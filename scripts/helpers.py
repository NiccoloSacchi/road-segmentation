#!/usr/bin/env python3

import numpy as np
import matplotlib.image as mpimg
from tf_aerial_images import *
import pandas as pd

# each patch is 16*16 pixels
patch_size = 16

def load_images(n=100):
    """ 
    Loads n training images (n=100 loads all the images) both the input images.
    and the groundtruth images. Returns the two list of images. """
    n = min(n, 100) 
    print("Loading " + str(n) + " images")
    
    root_dir = "../dataset/training/"
    
    image_dir = root_dir + "images/"
    images = os.listdir(image_dir)
    
    gt_image_dir = root_dir + "groundtruth/"
    gt_images = os.listdir(gt_image_dir)
    
    return [load_image(image_dir + images[i]) for i in range(n)], [load_image(gt_image_dir + gt_images[i]) for i in range(n)]

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img_patch):
    feat_m = np.mean(img_patch, axis=(0,1)) # average of each color
    feat_v = np.var(img_patch, axis=(0,1))  # variance of each color
    feat = np.append(feat_m, feat_v)
    return feat

def imgs_to_inputs(imgs):
    """ Given a lit of input images, converts them inputs and 
    relative features. 
    X's rows: inputs
    X's column: features """
    X = np.asarray([input for img in imgs for input in img_to_inputs(img)])
    return X
    
def imgs_to_outputs(imgs):
    """ Given a list groundtruth images, splits them into patches and 
    convert each patch into either 0 (background) or 1 (road) """
    Y = np.asarray([input for img in imgs for input in img_to_outputs(img)])
    return Y
    
def img_to_inputs(img):
    """ Given an input image, converts it into 625 inputs (patches) and 
    for each one of them the features. """
    X = np.asarray([ extract_features(patch) for patch in img_crop(img, patch_size, patch_size)])
    return X
    
def img_to_outputs(img):
    """ Given a groundtruth image, splits it into patches and 
    convert each patch into either 0 (background) or 1 (road) """
    Y = [int(value_to_class(np.mean(patch))) for patch in img_crop(img, patch_size, patch_size)]
    return Y

def confusion_matrix(predictions, correct):
    """ Compute the confusion matrix out of the given predictions and correct labels """
    Z = predictions
    Y = correct
    
    FNR = np.sum((Z == 0) & (Y == 1)) / float(len(Z))
    TNR = np.sum((Z == 0) & (Y == 0)) / float(len(Z))

    FPR = np.sum((Z == 1) & (Y == 0)) / float(len(Z))
    TPR = np.sum((Z == 1) & (Y == 1)) / float(len(Z))
    
    confmat = pd.DataFrame(
        data = [[TNR, FNR], [FPR, TPR]],
        index = pd.MultiIndex(
            levels=[['predicted (Z)'], ['0', '1']],
            labels=[[0, 0], [0, 1]]),
        columns = pd.MultiIndex(
            levels=[['actual (Y)'], ['0 (background)', '1 (road)']],
            labels=[[0, 0], [0, 1]]),
    )
    
    return confmat