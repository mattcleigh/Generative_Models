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

class conv_block(nn.Module):
    def __init__(self, data_shape, depth, c_out, kern, pad, act, nrm, pool, upscl):
        super(conv_block, self).__init__()

        ## The list the contains the sequence of operations
        layers = []

        ## Upscaling and start of block
        if upscl and pool>0:
            layers.append( nn.Upsample(scale_factor=pool, mode="nearest" ) )

        ## The convolutional layers with activations and normalisation
        for d in range(1,depth+1):
            c_in = data_shape[0] if d==1 else c_out
            layers.append( nn.Conv2d( in_channels=c_in, out_channels=c_out,
                                      kernel_size=kern, stride=1, padding=pad ) )
            if act is not None: layers.append( act )
            if nrm: layers.append( PixelNorm() )
            # if nrm and not upscl: layers.append( nn.InstanceNorm2d(c_out) )

        ## Downscaling at end of block
        if not upscl and pool>0:
            layers.append( nn.AvgPool2d( kernel_size=pool, stride=pool ) )

        self.block = nn.Sequential(*layers)

        ## Save the excpected input and the output shapes of the block
        self.input_shape = data_shape
        with T.no_grad():
            self.output_shape = list( self.block( T.ones( size=[1]+self.input_shape) ).shape[1:] )

    def forward(self, input):
        return self.block(input)

class res_block(conv_block):
    def __init__(self, *args):
        super(res_block, self).__init__(*args)

        ## Create the residual path CNN
        c_in  = self.input_shape[0]
        c_out = self.output_shape[0]
        self.fmap_change = nn.Conv2d( c_in, c_out, 1 ) if c_in != c_out else nn.Identity()

    def forward(self, input):
        layer_out = self.block(input)
        skip = F.interpolate(input, size=layer_out.shape[2:], mode="bilinear", align_corners=False) ## mode="nearest", align_corners=False)
        skip = self.fmap_change(skip)
        return (layer_out + skip) / np.sqrt(2)

class skip_block(conv_block):
    def __init__(self, *args, toRBGout = [], do_hidden=False):
        super(skip_block, self).__init__(*args)

        ## Create the to RBG layer CNN
        c_in = self.output_shape[0]
        self.to_rgb = nn.Conv2d( c_in, toRBGout, 1 ) if do_hidden else nn.Identity()
        self.do_hidden = do_hidden

    def forward(self, input):

        ## Work out what inputs are available
        if type(input) is tuple:
            old_img, old_hidden = input
        else:
            old_hidden = input
            old_img = None

        ## Get the block outputs
        new_hidden = self.block(old_hidden)
        new_img = self.to_rgb(new_hidden)

        ## Add the old image if present
        if old_img is not None:
            new_img += F.interpolate(old_img, size=new_img.shape[2:], mode="nearest" )

        ## Return the results
        if self.do_hidden:
            return new_img, new_hidden
        else:
            return new_img

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
        block = []

        block.append( lin_layer(widths[l-1], widths[l]) )
        block.append( act_h )
        if drpt>0: block.append( nn.Dropout(p=drpt) )
        if lnrm:   block.append( nn.LayerNorm(widths[l]) )

        block_nn = nn.Sequential(*block)
        layers.append(("dense_block_{}".format(l), block_nn))

    ## Creating the "output" layer of the stream if applicable which is sometimes
    ## Not the case when creating base streams in larger arcitectures
    if n_out is not None:
        block = []
        block.append( lin_layer(widths[-1], n_out) )
        if act_o is not None:
            block.append( act_o )
        block_nn = nn.Sequential(*block)
        layers.append(("dense_block_out", block_nn))

    ## Return the list of features or...
    if return_list:
        return layers

    ## ... convert the list to an nn, then return
    return nn.Sequential(OrderedDict(layers))

def cnn_creator( name, data_shape, cnn_layers, act, nrm, upscl, x_chnls=None ):
    ## CNN layers are specified in order:
    ## channel, kernel, padding, depth, pooling, residual

    ## We want to ignore one of datapoints, either first or last
    d = len(cnn_layers)
    layers = []

    for l, (c,k,p,n,pl,res) in enumerate(cnn_layers):
        block = []

        ## Calculate the desired number of output chanels
        out_ch = x_chnls if (upscl and l==d-1) else cnn_layers[l+upscl][0]

        ## Dont put an act, norm, in the first(last) layer of an encoding(decoding) network
        flag = (not upscl and l!=0) or (upscl and l!=d-1)
        do_nrm  = nrm if flag else False
        do_actv = act if flag else None
        do_hidden = True if flag else False

        args = [data_shape, n, out_ch, k, p, do_actv, do_nrm, pl, upscl ]

        ## Use the block creator
        if res and not upscl:
            block_nn = res_block( *args )
        elif res and upscl:
            block_nn = skip_block( *args, toRBGout=x_chnls, do_hidden=do_hidden )
        else:
            block_nn = conv_block( *args )

        ## We update the shape of the data for the new layer
        data_shape = block_nn.output_shape

        ## Add the block to the layer list
        layers.append(block_nn)

    layer_nn = nn.Sequential(*layers)

    return layer_nn, data_shape

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
