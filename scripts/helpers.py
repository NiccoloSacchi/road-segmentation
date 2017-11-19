import numpy as np
import matplotlib.image as mpimg
from PIL import Image

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    if len(gt_img.shape) == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        w = gt_img.shape[0]
        h = gt_img.shape[1]
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

# each patch is 16*16 pixels
patch_size = 16
# percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.25 

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
    Y = [int(np.mean(patch)>foreground_threshold) for patch in img_crop(img, patch_size, patch_size)]
    return Y
