import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Networks as myNN
from Resources import Plotting as myPL

import numpy as np
import matplotlib.pyplot as plt

from itertools import count
from collections import OrderedDict

import torch as T
import torch.nn as nn
import torch.nn.functional as F

def get_random_samples( shape, flag ):
    if flag == "gauss":
        return T.randn( shape )

    if flag == "disk":
        theta = 2*np.pi*T.rand(shape[0])
        length = T.sqrt(T.rand(shape[0]))
        x, y = length * T.cos(theta), length * T.sin(theta)
        return T.cat( (x.unsqueeze(1), y.unsqueeze(1)), 1 )

    if flag == "square":
        return 2 * T.rand( shape ) - 1

    if flag == "star10":
        theta = 2*np.pi*T.rand(shape[0])
        length = T.sqrt(T.rand(shape[0]))
        x, y = length * T.cos(theta), length * T.sin(theta)

        ctheta = T.randint(0, 10, (shape[0],)) * np.pi / 5
        clen = 0.5 * T.rand(shape[0]) + 0.5
        cx, cy = clen * T.cos(ctheta), clen * T.sin(ctheta)

        tx = x * np.pi/20 + cx
        ty = y * np.pi/20 + cy

        return T.cat( (tx.unsqueeze(1), ty.unsqueeze(1)), 1 )

def calc_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = T.linalg.norm(grad.data, ord=2)
        total_norm += param_norm.item()*param_norm.item()
    return np.sqrt(total_norm)

def calc_cnn_out_dim( x_dims, cnn_layers ):
    """ A function to return the exact number of outputs from a CNN
    """
    print("\nChecking Data-CNN-MLP Compatibility:")
    n_in = x_dims[-1]
    print("Input image = {}x{}x{}".format(*x_dims))
    for l, (c,k,p,n,pl,res) in enumerate(cnn_layers, 1):
        n_in = ( n_in + 2*p - k ) / 1 + 1
        print(" --> Conv {} out = {}x{}x{}".format(l,c,int(n_in),int(n_in)) )
        if pl>0:
            n_in = ( n_in - pl ) / pl + 1
            print(" --> Pool {} out = {}x{}x{}".format(l,c,int(n_in),int(n_in)) )

    assert( n_in.is_integer() ), "Incompatible layer/kernel sizes"

    ## Finnaly we square the output and multiply by the final channel number
    ## This is because all channels are flattened into one array
    n_out = int(n_in*n_in*c)
    print("Total network output = {}x{}x{} = {}".format(c,int(n_in),int(n_in),n_out))
    return n_out

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val
