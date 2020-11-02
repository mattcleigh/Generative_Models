import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Plotting as myPL

import matplotlib.pyplot as plt

from itertools import count
from collections import OrderedDict

import torch as T
import torch.nn as nn
import torch.nn.functional as F

def mlp_creator( name, n_in=1, n_out=None, d=1, w=256,
                       act_h=nn.ReLU(), act_o=None, drpt=0, lnrm=False,
                       custom_size=None, return_list=False ):
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

    ## Generating an array to use as the widths
    widths.append( n_in )
    if custom_size is not None:
        d = len( custom_size )
        widths += custom_size
    else:
        widths += d*[w]

    ## Creating the "hidden" layers in the stream
    for l in range(1, d+1):
        layers.append(( "{}_lin_{}".format(name, l), nn.Linear(widths[l-1], widths[l]) ))
        layers.append(( "{}_act_{}".format(name, l), act_h ))
        if drpt>0:
            layers.append(( "{}_drp_{}".format(name, l), nn.Dropout() ))
        if lnrm:
            layers.append(( "{}_nrm_{}".format(name, l), nn.LayerNorm(widths[l]) ))

    ## Creating the "output" layer of the stream if applicable which is sometimes
    ## Not the case when creating base streams in larger arcitectures
    if n_out is not None:
        layers.append(( "{}_lin_out".format(name), nn.Linear(widths[-1], n_out) ))
        if act_o is not None:
            layers.append(( "{}_act_out".format(name), act_o ))

    ## Return the list of features or...
    if return_list:
        return layers

    ## ... convert the list to an nn, then return
    return nn.Sequential(OrderedDict(layers))

def cnn_creator( name, data_chnls, cnn_layers, act, brnm, tpose = False ):
    ## CNN layers are specified in order: C,K,S,P

    ## We want to ignore one of datapoints, either first or last
    d = len(cnn_layers)
    conv_type = nn.ConvTranspose2d if tpose else nn.Conv2d
    layers = []

    for l, (c,k,s,p) in enumerate(cnn_layers):
        if tpose: l = l + 1
        ## Working out how the channels mesh
        in_ch  = data_chnls if (not tpose and l==0) else cnn_layers[l-1][0]
        out_ch = data_chnls if (tpose and l==d)     else cnn_layers[l][0]

        layers.append(( "{}_conv2d_{}".format(name, l), conv_type( in_channels=in_ch, out_channels=out_ch,
                                                                   kernel_size=k, stride=s, padding=p ) ))
        if not tpose or l<d-1:
            if brnm:
                layers.append(( "{}_bnm_{}".format(name, l+1), nn.BatchNorm2d(out_ch) ))
            layers.append(( "{}_act_{}".format(name, l+1), act ))

    return nn.Sequential(OrderedDict(layers))


def calc_cnn_out_dim( x_dims, cnn_layers ):
    """ A function to return the exact number of outputs from a CNN
    """
    print("\nChecking Data-CNN-MLP Compatibility:")
    n_in = x_dims[-1]
    for c,k,s,p in cnn_layers:
        n_in = ( n_in + 2*p - k ) / s + 1
        print(" - ", n_in)

    assert( n_in.is_integer() ), "Incompatible layer/kernel sizes"

    ## Finnaly we square the output and multiply by the final channel number
    ## This is because all channels are flattened into one array
    n_out = int(n_in*n_in*c)
    print(" - - dimension of CNN output = ", n_out)
    return n_out
