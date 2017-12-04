#!/usr/bin/env python3

import numpy as np
import matplotlib.image as mpimg
from tf_aerial_images import *
import pandas as pd
from types import SimpleNamespace 
from sklearn import linear_model
import matplotlib.pyplot as plt
from keras.utils import np_utils

PATCH_WIDTH = 16 # each patch is 16*16 pixels

def extend_images(imgs, window_width):
    """ Given a list of images and the size of the window, extends each 
    image by mirroring its border pixels. """
    
    if window_width < PATCH_WIDTH:
        print("Error: the window width should be at least a wide as the patch.")
        return 
    
    if window_width % 2 != 0:
        print("Error: the window width should be even.")
        return 
    
    mirror_width = int((window_width-PATCH_WIDTH)/2)
    
    def extend_image(img):
        new_size = np.array(img.shape)
        new_size[0] = new_size[0] + mirror_width*2
        new_size[1] = new_size[1] + mirror_width*2
        extended_img = np.zeros(new_size)
        # put the original image at the center
        extended_img[mirror_width:-mirror_width, mirror_width:-mirror_width] = img

        # then mirror the 4 sides
        # left 
        extended_img[mirror_width:-mirror_width, :mirror_width] = np.flip(img[:, :mirror_width], 1)
        # right
        extended_img[mirror_width:-mirror_width, -mirror_width:] = np.flip(img[:, -mirror_width:], 1)
        # top 
        extended_img[:mirror_width, mirror_width:-mirror_width] = np.flip(img[:mirror_width, :], 0)
        # bottom 
        extended_img[-mirror_width:, mirror_width:-mirror_width] = np.flip(img[-mirror_width:, :], 0)

        # finally "fill" the corners
        # top-left
        extended_img[:mirror_width, :mirror_width] = np.flip(extended_img[mirror_width:2*mirror_width, :mirror_width], 0)
        # top-right
        extended_img[:mirror_width, -mirror_width:] = np.flip(extended_img[mirror_width:2*mirror_width, -mirror_width:], 0)
        # bottom-left
        extended_img[-mirror_width:, :mirror_width] = np.flip(extended_img[-2*mirror_width:-mirror_width, :mirror_width], 0)
        # bottom-right
        extended_img[-mirror_width:, -mirror_width:] = np.flip(extended_img[-2*mirror_width:-mirror_width, -2*mirror_width:-mirror_width], 0)
        
        return extended_img
    
    return np.array([extend_image(img) for img in imgs])
        

def display_ith_prediction(test, Z, i, window_width):
    """ Given the test (or the train) object, the respective predictions, an index i and the width of
    the window used, display the i-th image and its respective predictions. """
    # just compute some parameters
    mirror_width = int((window_width-PATCH_WIDTH)/2)
    w, h = test.imgs[0].shape[0]-mirror_width*2, test.imgs[0].shape[1]-mirror_width*2
    n_windows_h = int(h / PATCH_WIDTH)
    n_windows_w = int(w / PATCH_WIDTH)
    n_windows = n_windows_w*n_windows_h
    
    # get the i-th image (remove the "mirrors")
    image = test.imgs[i][mirror_width:-mirror_width, mirror_width:-mirror_width]
    # select the predictions from Z
    prediction = Z[i*n_windows:(i+1)*n_windows]
    display_prediction(image, prediction)
    return


def imgs_to_inputs_outputs(imgs, gt_imgs, window_width):
    """ Given a list of images and a window width, convert each image into a list of windows
    where each window is related to a patch that need to be classified (positioned at the 
    center of the image).
    Returns the list of windows and the corresponding list of labels (computed from the 
    list of groundtruth images). 
    """
    
    h, w = imgs[0].shape[0], imgs[0].shape[1]
    
    patch_border = window_width-PATCH_WIDTH
    n_windows_h = (h-patch_border) / PATCH_WIDTH
    n_windows_w = (w-patch_border) / PATCH_WIDTH
    
    if n_windows_h%1 != 0:
        print("Error: the size of the image should be such that when shifting the window with steps as big as the" + 
              " patch width the windows is always completely within the image. Check the height of the image.")
    if n_windows_w%1 != 0:
        print("Error: the size of the image should be such that when shifting the window with steps as big as the" + 
              " patch width the windows is always completely within the image. Check the width of the image.")
        
    n_images = len(imgs)
    n_gt_images = len(gt_imgs)
    if n_images != n_gt_images:
        print("Error: you should pass for each image its groundtruth. Check the length of the passed list of images.")
    
    n_windows_h = int(n_windows_h)
    n_windows_w = int(n_windows_w)
    n_windows = int(n_windows_h*n_windows_w)
    def img_to_inputs_outputs(img, gt_img):
        """ Given an image and its groundtruth, extract the inputs (windows) and the relative outputs """
        # a list of windows of the image
        windows = np.zeros((n_windows, window_width, window_width, img.shape[2]))
        outputs = np.zeros(n_windows)
        # [row_start, col_start] = top-left index of the window
        for i, row_start in enumerate(range(0, h-window_width+1, PATCH_WIDTH)):
            n_rows_windows = i*n_windows_w # number of windows in previous rows
            row_end = row_start + window_width
            for j, col_start in enumerate(range(0, w-window_width+1, PATCH_WIDTH)):
                col_end = col_start + window_width
                windows[n_rows_windows + j, :, :, :] = img[row_start:row_end, col_start:col_end]
                outputs[n_rows_windows + j] = int(value_to_class(np.mean(get_patch(gt_img, i, j))))
        return windows, np_utils.to_categorical(outputs, 2)
   
    X = np.zeros((n_images*n_windows, window_width, window_width, 3)) # array of windows
    Y = np.zeros((n_images*n_windows, 2)) # array of labels (2 classes)
    for n_image in range(n_images):
         # convert each image into a couple of (inputs, ouputs) and then append them to X and Y 
        index = n_image*n_windows
        X[index:index+n_windows], Y[index:index+n_windows] = img_to_inputs_outputs(imgs[n_image], gt_imgs[n_image])
    return X, Y 

def get_patch(img, i, j):
    """ Returns the patch at position (i, j) """
    row_start = i*PATCH_WIDTH
    col_start = j*PATCH_WIDTH
    return img[row_start:row_start+PATCH_WIDTH, col_start:col_start+PATCH_WIDTH]
    
def flatten(matr):
    """ Given a matrix (or an array), flattens it. """
    return [val for row in matr for val in row]

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features_from_patch(img_patch):
    feat_m = np.mean(img_patch, axis=(0,1)) # average of each color
    feat_v = np.var(img_patch, axis=(0,1))  # variance of each color
    feat = np.append(feat_m, feat_v)
    return feat

def imgs_to_inputs(imgs):
    """ Given a list of input images, converts them inputs and 
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
    """ Given an input image, converts it into patches and 
    for each one of them the features. """
    X = np.asarray([extract_features_from_patch(patch) for patch in img_crop(img, PATCH_WIDTH, PATCH_WIDTH)])
    return X
    
def img_to_outputs(img):
    """ Given a groundtruth image, splits it into patches and 
    convert each patch into either 0 (background) or 1 (road) """
    Y = [int(value_to_class(np.mean(patch))) for patch in img_crop(img, PATCH_WIDTH, PATCH_WIDTH)]
    return Y

def stats(predictions, correct):
    """ Compute the confusion matrix out of the given predictions and correct labels """
    Z = predictions
    Y = correct
    
    confusion_matrix = SimpleNamespace()
    
    FN = np.sum((Z == 0) & (Y == 1)) #/ float(len(Z))
    TN = np.sum((Z == 0) & (Y == 0)) #/ float(len(Z))

    FP = np.sum((Z == 1) & (Y == 0)) #/ float(len(Z))
    TP = np.sum((Z == 1) & (Y == 1)) #/ float(len(Z))
    
    confusion_matrix.abs = pd.DataFrame(
        data = [[TN, FN], [FP, TP]],
        index = pd.MultiIndex(
            levels=[['predicted (Z)'], ['0', '1']],
            labels=[[0, 0], [0, 1]]),
        columns = pd.MultiIndex(
            levels=[['actual (Y)'], ['0 (background)', '1 (road)']],
            labels=[[0, 0], [0, 1]]),
    )
    
    # with rates
    try:
        FNR = FN / (TP + FN) # ratio of wrongly "covered" roads
    except:
        FNR = 0
    try:
        TNR = TN / (FP + TN) # ratio of correctly "covered" background
    except:
        TNR = 0
        
    try: 
        FPR = FP / (FP + TN) # ratio of wrongly "covered" background
    except:
        FPR = 0
        
    try:
        TPR = TP / (TP + FN) # ratio of correctly "covered" roads 
    except:
        TPR = 0
    
    confusion_matrix.ratio = pd.DataFrame(
        data = [[TNR, FNR], [FPR, TPR]],
        index = pd.MultiIndex(
            levels=[['predicted (Z)'], ['0', '1']],
            labels=[[0, 0], [0, 1]]),
        columns = pd.MultiIndex(
            levels=[['actual (Y)'], ['0 (background)', '1 (road)']],
            labels=[[0, 0], [0, 1]]),
    )
    
    # compute the F1 score (used in kaggle)
    score = SimpleNamespace()
    score.precision = TP / (TP + FP) # ratio of predicted road that are actually roads
    score.recall = TPR # ratio of road that has been covered
    score.F1 = 2*(score.precision * score.recall) / (score.precision + score.recall)
    
    # show the results
    #display(confusion_matrix.abs)
    #display(confusion_matrix.ratio)

    print("\nRecall =", score.recall)
    print("Precision =", score.precision)
    print("F1 =", score.F1)
    
    #return confusion_matrix, score
    return 


def update_features(curr_X_list, model):
    """ Given a matrix of features (relative to the patches of an image), 
    updates them using the predictions generated by  the given model, i.e. 
    the predicitions of a neigborhood are used as features for those patches.
    curr_X_list: list of current matrix of feaures (one matrix of features
    per image). curr_X_list[i].shape = (#patches per side, #patches per side, #features per patch)
    """
    n_images = len(curr_X_list)
    # get the number of patches per side and the number of features per patch.
    n_patches_side, _, n_features = curr_X_list[0].shape
    
    # use the passed model to extract more information and create new features
    # create 2 features per patch in the neighborhood [prob(background), prob(road)]
    side = 5 # must be odd
    neighborhood_size = side*side # square of side x side patches with the current patch in the middle
    
    # compute the current predictions for each patch (for each image) so to use them as features afterwards
    curr_predictions = np.zeros((n_images, n_patches_side, n_patches_side, 2))
    for im in range(n_images):
        for i in range(n_patches_side):
            for j in range(n_patches_side):
                # get predictions of patch [i, j] of image im
                curr_predictions[im][i, j] =  model.predict_proba(curr_X_list[im][i, j].reshape(1, -1)) 

    # for each patch store only the predictions of the neighborhood
    new_X_list = np.zeros((n_images, n_patches_side, n_patches_side, 2*neighborhood_size + n_features))

    span = int((side-1)/2) # how many patches upward and downward the current one
    # add a 'border' of zeros around the predictions, this will hugely simplify the next for
    pred_wrapper_shape = np.array(curr_predictions[0].shape)
    pred_wrapper_shape[0] += span*2 # add 'span' rows on top and on bottom
    pred_wrapper_shape[1] += span*2 # add 'span' rows on the right and on the left
    predictions = np.zeros(pred_wrapper_shape) 
    for im in range(n_images):
        predictions[span:-span, span:-span] = curr_predictions[im]
        for i in range(n_patches_side):
            for j in range(n_patches_side):
                neighborhood_pred = predictions[i:i+side, j:j+side] # get the predictions of the neighborhood
                # neighborhood_pred.shape = side x side x 2
                # we have to flatten this matrix of arrays to obtain the features of the current patch
                # append to the old features the (flattened) list of predictions
                new_X_list[im][i, j] = np.append(curr_X_list[im][i, j], flatten(flatten(neighborhood_pred)))
       
    return new_X_list
    
def initialize_input_output(imgs, gt_imgs):
    """ Returns a list of matrix of features (one matrix per image) and a list of outputs"""
    n_images = len(imgs)
    
    # get a list of matrix of patches (one matrix of patches per image)
    patches = [img_crop_matr(img) for img in imgs]
    n_patches_side = patches[0].shape[0]
    # get the list of outputs (we don't need to store the relative position here)
    output_patches = flatten([flatten(img_crop_matr(img)) for img in gt_imgs])
    Y = [int(value_to_class(np.mean(patch))) for patch in output_patches]
    
    # extract the initial features (patch-wise)
    n_features = extract_features_from_patch(patches[0][0, 0]).shape[0] # just get the number of features (to initialize the matr)
    X = np.zeros((n_images, n_patches_side, n_patches_side, n_features)) # for each patch 'n_features' features
    for im in range(n_images):
        for i in range(n_patches_side):
            for j in range(n_patches_side):
                X[im][i, j] = extract_features_from_patch(patches[im][i, j])
     
    return X, Y
            
def train_models(imgs, gt_imgs, n_layers=1):
    """ Given the input and the output images train a model iteratively n_layers times.
            1. Extract inputs and features.
            2. Train a model.
            3. Extract inputs and features also considering the predictions of the previously
            trained model. Then repeat from 2.
        Returns a list of models through which an in put image should go through.
    """
    print("Extracting input and output...")
    X_list, Y = initialize_input_output(imgs, gt_imgs)
    # X_list: list of matrices (one matrix of features per image)
    
    # iteratively train a model and update the features
    models = [linear_model.LogisticRegression(C=1e5, class_weight="balanced") for layer in range(n_layers)]
    for layer in range(n_layers): # train a new model for each layer
        print("Training the model", layer, "...")
        # train a model on the current features
        X = flatten([flatten(matr) for matr in X_list]) # convert each matr of features into a list of features
        models[layer].fit(X, Y)
        # update the features using the predictions of the current model
        X_list = update_features(X_list, model=models[layer])
        
    return models

def test_models(imgs, gt_imgs, models=[]):
    """ Given a list of models iterate over them to obtain a final prediction. """
    if len(models)==0:
        print("Error: pass at least a model")
        return
    
    X_list, Y = initialize_input_output(imgs, gt_imgs)
    
    # use all the models except the last one to update the features
    for model in models[:-1]:
        X_list = update_features(X_list, model)
       
    # the last model is used to compute the predictions
    X = flatten([flatten(matr) for matr in X_list])
    Z = models[-1].predict(X)
    
    Y = np.array(Y)
    # show the performance
    stats(Z, Y)
    
    return Z, Y 