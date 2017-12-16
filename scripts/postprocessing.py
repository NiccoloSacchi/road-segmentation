""" Here all the functions used to improve the prediction. """

import numpy as np
from scipy.ndimage import rotate
from sklearn.metrics import f1_score
from preprocessing import rotate_image

def predictions_to_class(predictions, threshold = 0.5):
    """ Given a list of predictions convert each one of them with 'prediction_to_class' """
    return np.array([prediction_to_class(pred, threshold=threshold) for pred in predictions])

def prediction_to_class(prediction, threshold = 0.5):
    """ Given a prediction convert each entry to a class, depending on the 
        given threshold (predictions[x, y][1] > treshold => it is a road).
        prediction = predicted probabilities of the patches of one 
                image. shape=(H/n, W/n, 2)
    """
    
    final_prediction = np.zeros(prediction.shape[:-1]) # discard the third dimension
    mask_roads = prediction[:, :, 1] > threshold
    final_prediction[mask_roads] = 1.0
    return final_prediction

def take_image_at_center(img, target_shape):
    """ Given an image and a target shape, drops the borders to take the inner image of the given size.
    """
    h_border = img.shape[0]-target_shape[0]
    if h_border%2 == 0:
        border_top = border_bottom = int(h_border/2)
    else:
        border_top = int((h_border-1)/2)
        border_bottom = int(border_top+1)
        
    w_border = img.shape[1]-target_shape[1]
    if w_border%2 == 0:
        border_left = border_right = int(w_border/2)
    else:
        border_left = int((w_border-1)/2)
        border_right = int(border_left+1)
    
    return img[border_top:img.shape[0]-border_bottom, border_left:img.shape[1]-border_right]    

def grid_search_treshold(predictions, true):
    """ Given the list of predictions and the list of true labels find 
        the treshold that minimizes the F1 score.
        predictions: probabilities of each batch. shape=(#batches,)
        true: classes of each batch. shape=(#batches,)"""
    F1_best = 0 
    treshold_best = 0
    for treshold in np.arange(0, 1, 0.01):
        pred = predictions > treshold
        f1 = f1_score(true, pred, pos_label=1, average='binary', sample_weight=None)
        if f1 > F1_best:
            F1_best = f1
            treshold_best = treshold
    return F1_best, treshold_best