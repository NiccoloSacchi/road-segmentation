""" Here all the functions used to improve the prediction. """

import numpy as np

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