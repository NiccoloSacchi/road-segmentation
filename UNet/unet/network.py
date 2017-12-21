
from itertools import chain
from operator import mul

import logging

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from .utils import numpytorch

logger = logging.getLogger(__name__)


#UTILS
#-----------------------------------------------------------------------
def crop_slices(shape1, shape2):
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2)
                    for sh1, sh2 in zip(shape1, shape2)]
    return slices

def crop_and_merge(tensor1, tensor2):
    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)

    return torch.cat((tensor1[slices], tensor2), 1)
#-----------------------------------------------------------------------

class UNetConfig(object):

    def __init__(self,
                 steps=4,
                 first_layer_channels=64,
                 num_classes=2,
                 num_input_channels=1,
                 two_sublayers=True,
                 ndims=2,
                 border_mode='valid'):

        logger.info("Creating network...")

        # checking border mode for the convolution ops
        #-----------------------------------------------------------------------
        if border_mode not in ['valid', 'same']:
            raise ValueError("`border_mode` not in ['valid', 'same']")
        #-----------------------------------------------------------------------

        # parameters of the network
        #-----------------------------------------------------------------------
        #number of steps of the network

        self.steps = steps
        # number of channels after the first conv op
        self.first_layer_channels = first_layer_channels
        # number of channel of the input
        self.num_input_channels = num_input_channels
        # number of classe (here 0 neg, 1 pos, 2 ignore)
        self.num_classes = num_classes
        # using two subkayers
        self.two_sublayers = two_sublayers
        # number of dimensions
        self.ndims = ndims
        # border mode for conv ops
        self.border_mode = border_mode
        #-----------------------------------------------------------------------

class UNetLayer(nn.Module):

    def __init__(self, num_channels_in, num_channels_out, ndims,
                 two_sublayers=True, border_mode='valid'):
        """"Building the elementary layer of the network"""

        super(UNetLayer, self).__init__()

        # using two subkayers
        #-----------------------------------------------------------------------
        self.two_sublayers = two_sublayers
        #-----------------------------------------------------------------------

        # init of the ops
        #-----------------------------------------------------------------------
        conv_op = nn.Conv2d
        drop_op = nn.Dropout2d
        norm_op = nn.BatchNorm2d
        #-----------------------------------------------------------------------

        # padding setting
        #-----------------------------------------------------------------------
        if border_mode == 'valid':
            padding = 0
        elif border_mode == 'same':
            padding = 1
        else:
            raise ValueError("unknown border_mode `{}`".format(border_mode))
        #-----------------------------------------------------------------------

        # building sequential element
        #-----------------------------------------------------------------------
        conv1 = conv_op(num_channels_in, num_channels_out,
                        kernel_size=3, padding=padding)
        relu1 = nn.ReLU()

        conv2 = conv_op(num_channels_out, num_channels_out,
                        kernel_size=3, padding=padding)
        relu2 = nn.ReLU()
        norm2 = norm_op(num_channels_out)
        drop2 = drop_op(0.1)
        #-----------------------------------------------------------------------

        # forming the sequence
        #-----------------------------------------------------------------------
        self.unet_layer = nn.Sequential(conv1, relu1, norm2,
                                        conv2, relu2, norm2, drop2)
        #-----------------------------------------------------------------------


    def forward(self, x):
        # forward founction of the layer
        return self.unet_layer(x)

class UNet(nn.Module):
    """The U-Net network"""

    def __init__(self, unet_config=None):

        super(UNet, self).__init__()

        # lloking for a config
        #-----------------------------------------------------------------------
        if unet_config is None:
            unet_config = UNetConfig()
        #-----------------------------------------------------------------------

        # laoding config
        #-----------------------------------------------------------------------
        self.config = unet_config
        ndims = self.config.ndims
        #-----------------------------------------------------------------------

        # init the ops
        #-----------------------------------------------------------------------
        self.max_pool = nn.MaxPool2d(2)
        ConvLayer = nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose2d
        #-----------------------------------------------------------------------

        # building first layer
        #-----------------------------------------------------------------------
        first_layer_channels = self.config.first_layer_channels
        two_sublayers = self.config.two_sublayers
        layer1 = UNetLayer(self.config.num_input_channels,
                           first_layer_channels,
                           ndims=ndims,
                           two_sublayers=two_sublayers,
                           border_mode=self.config.border_mode)
        #-----------------------------------------------------------------------

        # building down layers
        #-----------------------------------------------------------------------
        down_layers = [layer1]
        for i in range(1, self.config.steps + 1):
            lyr = UNetLayer(first_layer_channels * 2**(i - 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            two_sublayers=two_sublayers,
                            border_mode=self.config.border_mode)

            down_layers.append(lyr)
        #-----------------------------------------------------------------------

        # building up layers
        #-----------------------------------------------------------------------
        up_layers = []
        for i in range(self.config.steps - 1, -1, -1):
            # Up-convolution
            upconv = ConvTransposeLayer(in_channels=first_layer_channels * 2**(i+1),
                                        out_channels=first_layer_channels * 2**i,
                                        kernel_size=2,
                                        stride=2)

            lyr = UNetLayer(first_layer_channels * 2**(i + 1),
                            first_layer_channels * 2**i,
                            ndims=ndims,
                            two_sublayers=two_sublayers,
                            border_mode=self.config.border_mode)

            up_layers.append((upconv, lyr))
        #-----------------------------------------------------------------------

        # building final layers
        #-----------------------------------------------------------------------
        final_layer = ConvLayer(in_channels=first_layer_channels,
                                out_channels=self.config.num_classes,
                                kernel_size=1)

        #-----------------------------------------------------------------------

        # saving the layers
        #-----------------------------------------------------------------------
        self.down_layers = down_layers
        self.up_layers = up_layers
        self.down = nn.Sequential(*down_layers)
        self.up = nn.Sequential(*chain(*up_layers))
        self.final_layer = final_layer
        #-----------------------------------------------------------------------


    @numpytorch
    def forward(self, input):
        """forward function"""

        # going through the first layer
        #-----------------------------------------------------------------------
        x  = self.down_layers[0](input)
        #-----------------------------------------------------------------------

        # init the list
        #-----------------------------------------------------------------------
        down_outputs = [x]
        #-----------------------------------------------------------------------

        # going through the down layers
        #-----------------------------------------------------------------------
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)
        #-----------------------------------------------------------------------

        # going through the up layers
        #-----------------------------------------------------------------------
        for (upconv_layer, unet_layer), down_output in zip(self.up_layers, down_outputs[-2::-1]):
            x = upconv_layer(x)
            x = crop_and_merge(down_output, x)
            x = unet_layer(x)
        #-----------------------------------------------------------------------

        # going through the final layer
        #-----------------------------------------------------------------------
        x = self.final_layer(x)
        #-----------------------------------------------------------------------

        return x


class UNetClassifier(UNet):
    """UNet used as a classifier."""

    def __init__(self, unet_config=None):

        super(UNetClassifier, self).__init__(unet_config)

        # init of the softmax
        #-----------------------------------------------------------------------
        self.softmax = nn.Softmax2d()
        #-----------------------------------------------------------------------


    @staticmethod
    def unet_cross_entropy_labels(pred, labels, weights=None):

        aux  = pred.gather(1, labels[:, None])

        # computing the cross entropy
        #-----------------------------------------------------------------------
        loss = -aux.log()
        #-----------------------------------------------------------------------

        # ignoring the pixel that should be ignored
        #-----------------------------------------------------------------------
        if weights is not None:
            loss = loss * weights
            loss = loss[weights != 0.0]
        #-----------------------------------------------------------------------

        return loss.mean()

    @numpytorch
    def linear_output(self, input):
        """getting the ouput of the network"""
        return super(UNetClassifier, self).forward(input)

    @numpytorch
    def forward(self, input):
        """applying softmax to the output of the network"""
        x = self.linear_output(input)
        x = self.softmax(x)
        return x

    @numpytorch
    def loss(self, input, labels, weights=None):
        """computing the loss of the network"""
        pred = self(input)
        return self.unet_cross_entropy_labels(pred, labels, weights)
