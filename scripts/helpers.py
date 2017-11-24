#!/usr/bin/env python3

import numpy as np
import matplotlib.image as mpimg
from tf_aerial_images import *
import pandas as pd
from types import SimpleNamespace 
from sklearn import linear_model
import matplotlib.pyplot as plt

# each patch is 16*16 pixels
patch_size = 16

def display_prediction(img, Z):
    # Display prediction as an image
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 

    patch_size = 16
    w, h = img.shape[0], img.shape[1] # 608, 608 for test set

    # show the i-th image of the test set and the corresponding predictions
    predicted_im = label_to_img(w, h, patch_size, patch_size, Z)
    new_img = make_img_overlay(img, predicted_im)

    plt.imshow(new_img)
    
# Extract patches from a given image
def img_crop_matr(img):
    """ Returns a matrix of patches. """
    is_2d = len(img.shape) < 3
    h, w = img.shape[0], img.shape[1]
    
    # please make sure h and w are divisible by patch_size
    h_patches = int(h/patch_size)
    w_patches = int(w/patch_size)
    
    # a matrix of patches
    if is_2d:
        patches = np.zeros((h_patches, w_patches, patch_size, patch_size))
    else:
        patches = np.zeros((h_patches, w_patches, patch_size, patch_size, 3))

    for i in range(h_patches):
        h_start = patch_size*i
        h_end = patch_size*(i+1)
        for j in range(w_patches):
            w_start = patch_size*j
            w_end = patch_size*(j+1)
            if is_2d:
                patches[i, j] = img[h_start:h_end, w_start:w_end]
            else:
                patches[i, j] = img[h_start:h_end, w_start:w_end, :]
    return patches

def flatten(matr):
    """ Given a matrix (or an array), flattens it. """
    return [val for row in matr for val in row]

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
    X = np.asarray([ extract_features_from_patch(patch) for patch in img_crop(img, patch_size, patch_size)])
    return X
    
def img_to_outputs(img):
    """ Given a groundtruth image, splits it into patches and 
    convert each patch into either 0 (background) or 1 (road) """
    Y = [int(value_to_class(np.mean(patch))) for patch in img_crop(img, patch_size, patch_size)]
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
    FNR = FN / (TP + FN) # ratio of wrongly "covered" roads
    TNR = TN / (FP + TN) # ratio of correctly "covered" background

    FPR = FP / (FP + TN) # ratio of wrongly "covered" background
    TPR = TP / (TP + FN) # ratio of correctly "covered" roads 
    
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
    display(confusion_matrix.abs)
    display(confusion_matrix.ratio)

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