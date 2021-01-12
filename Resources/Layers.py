import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Utils as myUT

import os
import math
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from collections import OrderedDict

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = T.mul(x, x)
        tmp1 = T.rsqrt(T.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1

class res_cnn_block(nn.Module):
    def __init__(self, id, depth, chnls, kern, pad, act, pnrm, conv_layer):
        super(res_cnn_block, self).__init__()

        layers = []
        for d in range(1,depth+1):
            layers.append(( "res_{}_conv_{}".format(id,d), conv_layer( in_channels=chnls, out_channels=chnls,
                                                                       kernel_size=kern, stride=1, padding=pad ) ))
            layers.append(( "res_{}_actv_{}".format(id,d+1), act ))
            if pnrm: layers.append(( "res_{}_pnrm_{}".format(id,d+1), PixelNorm() ))

        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, data):
        return self.layers(data) + data

def mlp_creator( name, n_in=1, n_out=None, d=1, w=256,
                       act_h=nn.ReLU(), act_o=None, drpt=0, lnrm=False,
                       custom_size=None, return_list=False, equal=False ):
    """ A function used by many of the project algorithms to contruct a
        simple and configurable MLP.
        By default the function returns the full nn sequential model, but if
        return_list is set to true then the output will still be in list form
        to allow final layer configuration by the caller.
        The custom_size argument is a list for creating streams with varying
        layer width. If this is set then the width and depth parameters
        will be ignored.
    """
    layers = []
    widths = []

    lin_layer = EqualizedLinear if equal else nn.Linear

    ## Generating an array to use as the widths
    widths.append( n_in )
    if custom_size is not None:
        d = len( custom_size )
        widths += custom_size
    else:
        widths += d*[w]

    ## Creating the "hidden" layers in the stream
    for l in range(1, d+1):
        layers.append(( "{}_lin_{}".format(name, l), lin_layer(widths[l-1], widths[l]) ))
        layers.append(( "{}_act_{}".format(name, l), act_h ))
        if drpt>0:
            layers.append(( "{}_drp_{}".format(name, l), nn.Dropout(p=drpt) ))
        if lnrm:
            layers.append(( "{}_nrm_{}".format(name, l), nn.LayerNorm(widths[l]) ))

    ## Creating the "output" layer of the stream if applicable which is sometimes
    ## Not the case when creating base streams in larger arcitectures
    if n_out is not None:
        layers.append(( "{}_lin_out".format(name), lin_layer(widths[-1], n_out) ))
        if act_o is not None:
            layers.append(( "{}_act_out".format(name), act_o ))

    ## Return the list of features or...
    if return_list:
        return layers

    ## ... convert the list to an nn, then return
    return nn.Sequential(OrderedDict(layers))

def cnn_creator( name, data_chnls, cnn_layers, act, pnrm, upscl=False, equal=False ):
    ## CNN layers are specified in order: C,K,S,P

    ## We want to ignore one of datapoints, either first or last
    d = len(cnn_layers)
    layers = []

    conv_layer = EqualizedConv2d if equal else nn.Conv2d

    for l, (c,k,p,res,pl) in enumerate(cnn_layers):
        if upscl: l = l + 1
        ## Working out how the channels mesh
        in_ch  = data_chnls if (not upscl and l==0) else cnn_layers[l-1][0]
        out_ch = data_chnls if (upscl and l==d)     else cnn_layers[l][0]

        ## If upscaling the first thing we do is a nearest neighbour resize
        if upscl and pl>0:
            layers.append(( "{}_upsm_{}".format(name, l), nn.Upsample(scale_factor=pl, mode='nearest') ))

        ## If it wants a residual block, then we use that instead of a single convolution
        if res>0:
            if in_ch != out_ch:
                print("\n\n\n Warning! Can not change chanel size on residual block! \n\n\n")
            layers.append(( "{}_res_{}".format(name, l), res_cnn_block(l, res, in_ch, k, p, act, pnrm, conv_layer) ))

        ## We add the normal convolution layer
        else:
            layers.append(( "{}_conv_{}".format(name, l), conv_layer( in_channels=in_ch, out_channels=out_ch,
                                                                      kernel_size=k, stride=1, padding=p,
                                                                      padding_mode="replicate" ) ))

            ## We dont put an activation or normalisation in the final layer of a upscaling net
            if upscl and l==d:
                continue

            ## We add the non-linearity
            layers.append(( "{}_actv_{}".format(name, l+1), act ))

            ## We add the batch normalisation
            if pnrm:
               layers.append(( "{}_pnrm_{}".format(name, l+1), PixelNorm() ))

        ## We now add the pooling layer for the conv_net
        if not upscl and pl>0:
            layers.append(( "{}_avgp_{}".format(name, l), nn.AvgPool2d( kernel_size=pl, stride=pl ) ))

    layer_nn = nn.Sequential(OrderedDict(layers))

    return layer_nn

class GRF(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_=1):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = - lambda_ * grads
        return dx, None

class GRL(nn.Module):
    def __init__(self, lambda_=1):
        super(GRL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GRF.apply(x, self.lambda_)

class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(EqualizedLinear, self).__init__()

        ## Initialising the parameters
        self.weight = nn.Parameter(T.randn(out_dim, in_dim))
        self.bias = nn.Parameter(T.zeros(out_dim)) if bias else None

        ## Calculate the scale to be applied each forward pass
        fan_in = in_dim
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        return F.linear(x, self.weight*self.scale, self.bias)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class EqualizedConv2d(nn.Module):
    def __init__( self, in_channels, out_channels, kernel_size,
                  stride=1, padding=0, bias=True ):
        super(EqualizedConv2d, self).__init__()

        ## Initialising the parameters
        self.weight = nn.Parameter(T.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(T.zeros(out_channels)) if bias else None

        self.padding = padding
        self.stride = stride

        ## Calculate the scale to be applied each forward pass
        fan_in = in_channels * kernel_size**2
        self.scale = np.sqrt(2) / np.sqrt(fan_in)


    def forward(self, x):
        return F.conv2d( x, self.weight*self.scale, self.bias,
                         self.stride, self.padding )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )
