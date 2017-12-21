
import os
import logging

import numpy as np

import torch
from torch.autograd import Variable

def numpytorch(f):
    
    def newf(*args, **kwargs):
        
        newargs = []
        numpy_mode = False
        for arg in args:
            if isinstance(arg, np.ndarray):
                numpy_mode = True
                arg = Variable(torch.from_numpy(arg).cuda())
            newargs.append(arg)
        
        for k, arg in kwargs.items():
            if isinstance(arg, np.ndarray):
                numpy_mode = True
                arg = Variable(torch.from_numpy(arg).cuda())
                kwargs[k] = arg
        
        res = f(*newargs, **kwargs)
        
        if numpy_mode and isinstance(res, Variable):
            res = res.data.cpu().numpy()
        
        return res
    
    return newf

def labels_to_probabilities(labels, num_labels):
    
    probs = np.zeros((len(labels), num_labels) + labels.shape[1:], dtype=np.float32)
    a = np.arange(len(labels))
    probs[(a[:, None, None], labels) + np.ix_(*map(range, labels.shape[1:]))] = 1
    return probs

def pad_for_unet(array, unet_config, mode='reflect', padding_for='blocks'):
    """
    Pads an array for U-Net training and prediction. Detects if the array has
    more than one channel and does not pad over those dimensions.
    
    `padding_for` can be one of ['blocks', 'input', 'ouput'].
    """
    
    ndims = unet_config.ndims
    array_shape = array.shape[:ndims] # Array shape ignoring channels
    
    if padding_for == 'blocks':
        margin = unet_config.margin()
        pad_width = [(margin, margin)] * ndims
    elif padding_for == 'input':
        pad_width, _ = unet_config.in_out_pad_widths(array_shape)
    elif padding_for == 'output':
        _, pad_width = unet_config.in_out_pad_widths(array_shape)
    else:
        raise ValueError("Unknown `padding_for` value '{}'".format(padding_for))
    
    if array.ndim > ndims:
        pad_width = pad_width + [(0, 0)] * (array.ndim - ndims)
    
    return np.pad(array, pad_width, mode)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class BinCounter(object):
    """Counter of elements in NumPy arrays."""
    
    def __init__(self, minlength=0, x=None, weights=None):
        
        self.minlength = minlength
        self.counts = np.zeros(minlength, dtype=np.int_)
        
        if x is not None:
            self.update(x, weights)
    
    def update(self, x, weights=None):
        if weights is not None:
            weights = weights.flatten()
        
        current_counts = np.bincount(np.ravel(x), weights=weights, minlength=self.minlength)
        current_counts[:len(self.counts)] += self.counts
        
        self.counts = current_counts
    
    @property
    def frequencies(self):
        return self.counts / np.float_(np.sum(self.counts))
    

def config_logger(log_file):
    """
    Basic configuration of the logging system. Support logging to a file.
    """
    
    class MyFormatter(logging.Formatter):
        
        info_format = "\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s"
        error_format = "\x1b[31;1m%(asctime)s [%(name)s] [%(levelname)s]\x1b[0m %(message)s"
        
        def format(self, record):
            
            if record.levelno > logging.INFO:
                self._style._fmt = self.error_format
            else:
                self._style._fmt = self.info_format
            
            return super(MyFormatter, self).format(record)
    
    rootLogger = logging.getLogger()
    
    fileHandler = logging.FileHandler(log_file)
    fileFormatter = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s]> %(message)s")
    fileHandler.setFormatter(fileFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleFormatter = MyFormatter()
    consoleHandler.setFormatter(consoleFormatter)
    rootLogger.addHandler(consoleHandler)
    
    rootLogger.setLevel(logging.INFO)
