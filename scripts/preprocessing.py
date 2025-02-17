""" All function used to load and process the data before feeding it to the model:
    load data, reshape data, augment data"""

import os
import numpy as np
import matplotlib.image as mpimg
from types import SimpleNamespace 
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate

PATCH_WIDTH = 16

def load_images_to_predict(start,end):
    n = end-start 
    print("Loading " + str(n) + " images to predict")
    
    root_dir = "../dataset/test_set_images/"
    for i in range(start+1,end+1):
        print("loading image " + root_dir + "test_"+str(i)+"/test_"+str(i)+".png");
    
    return  np.array([load_image(root_dir + "test_"+str(i)+"/test_"+str(i)+".png") for i in range(start+1,end+1)])
    

def load_images_range(start=0, end=100):
    """ 
    Loads end-start training images (both sat and groundtruth images) and returns the two list of images. """
    print("Loading images " + str(start) + " to "+str(end))
    
    root_dir = "../dataset/training/"
    
    image_dir = root_dir + "images/"
    images = os.listdir(image_dir)
    
    gt_image_dir = root_dir + "groundtruth/"
    gt_images = os.listdir(gt_image_dir)
    
    return  np.array([load_image(image_dir + images[i]) for i in range(start,end)]), \
            np.array([load_image(gt_image_dir + gt_images[i]) for i in range(start,end)])
    
def load_images(n=100):
    """ 
    Loads n training images (n=100 loads all the images) both the input images.
    and the groundtruth images. Returns the two list of images. """
    n = min(n, 100) 
    print("Loading " + str(n) + " images")
    
    root_dir = "../dataset/training/"
    
    image_dir = root_dir + "images/"
    images = sorted(os.listdir(image_dir))
    
    gt_image_dir = root_dir + "groundtruth/"
    gt_images = sorted(os.listdir(gt_image_dir))
    
    return  np.array([load_image(image_dir + images[i]) for i in range(n)]), \
            np.array([load_image(gt_image_dir + gt_images[i]) for i in range(n)])

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_crop_matr(img, patch_width=PATCH_WIDTH):
    """ Returns a matrix of patches of the given width. """
    is_2d = len(img.shape) < 3
    h, w = img.shape[0], img.shape[1]
    
    # please make sure h and w are divisible by PATCH_WIDTH
    h_patches = int(h/patch_width)
    w_patches = int(w/patch_width)
    
    # a matrix of patches
    if is_2d:
        patches = np.zeros((h_patches, w_patches, patch_width, patch_width))
    else:
        patches = np.zeros((h_patches, w_patches, patch_width, patch_width, 3))

    for i in range(h_patches):
        h_start = patch_width*i
        h_end = patch_width*(i+1)
        for j in range(w_patches):
            w_start = patch_width*j
            w_end = patch_width*(j+1)
            if is_2d:
                patches[i, j] = img[h_start:h_end, w_start:w_end]
            else:
                patches[i, j] = img[h_start:h_end, w_start:w_end, :]
    return patches

def gt_img_to_Y(gt_img, predict_patch_width=PATCH_WIDTH):
        """ Reshape the groundtruth images: given the patch width convert each patch into either 
            0 or 1 and return the obtained matrix (groundtruth of smaller size) after converting 
            it to categorical (1 -> [0, 1], 0 -> [1, 0]). """
        gt_matr = img_crop_matr(gt_img, predict_patch_width)
        Y = np.zeros((gt_matr.shape[0], gt_matr.shape[1], 2)) # keep width and heigth but convert the patch to a class
        
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j, 1] = value_to_class(np.mean(gt_matr[i, j]))
                Y[i, j, 0] = 1-Y[i, j, 1]
        return Y#np_utils.to_categorical(Y, 2).astype('float32') #np_utils.to_categorical(Y, 2) ot np.expand_dims(Y, axis=3)
    
def images_to_XY(imgs, gt_imgs, predict_patch_width=PATCH_WIDTH):
    """ Convert the images to the required format so that they can be fed to the cnn. 
        predict_patch_width: is the with of the patch you want to predict (it must be 
        coherent with the otuput of the cnn). """
    
    def gt_imgs_to_Y(gt_imgs, patch_width=PATCH_WIDTH):
        """ Reshape the groundtruth images: given the patch width convert each patch into either 
            0 or 1 and return the obtained matrix (groundtruth of smaller size) after converting 
            it to categorical (1 -> [0, 1], 0 -> [1, 0]). """
        if len(gt_imgs>0):
            gt_matr = np.array([img_crop_matr(gt_img, patch_width) for gt_img in gt_imgs])
            Y = np.zeros((gt_matr.shape[0], gt_matr.shape[1], gt_matr.shape[2])) # keep width and heigth but convert the patch to a class
            for im in range(Y.shape[0]):
                for i in range(Y.shape[1]):
                    for j in range(Y.shape[2]):
                        Y[im, i, j] = value_to_class(np.mean(gt_matr[im, i, j]))
            return Y #np_utils.to_categorical(Y, 2) ot np.expand_dims(Y, axis=3)
        return np.array(None)

    X = imgs
    Y = gt_imgs_to_Y(gt_imgs, predict_patch_width)
    return X, Y
    
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def rotate_image(img, degrees, reshape=True):
    """ Efficiently rotate an imagefrom sklearn.metrics import f1_score
        img: images that has to be rotated
        degrees: rotation of the image
    """
    # transform degrees in a positive number in [0, 360]
    while degrees < 0:
        degrees = degrees+360
    while degrees > 360:
        degrees = degrees-360
        
    if degrees == 0:
        return img
    if (degrees % 90) == 0:
        # much faster with numpy
        return np.rot90(img, k=int(degrees/90), axes=(0, 1))
    return rotate(img, degrees, axes=(0,1), reshape=reshape, order=1, mode="reflect")
    
    
def split_train_test(X, Y, test_ratio=0.8, seed=1):
    """ Given a list of images and respective groundtruth images, splits them 
    into a train set and test set. 
    Use a fixed seed to guarantee reproducibility! """
    
    train = SimpleNamespace()
    test = SimpleNamespace()
    train.X, test.X, train.Y, test.Y = train_test_split(X, Y, test_size=test_ratio, random_state=seed)
    return train, test