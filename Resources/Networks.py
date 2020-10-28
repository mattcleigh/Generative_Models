import sys
home_env = '../'
sys.path.append(home_env)

import os
import math
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class VAE_Network(nn.Module):
    """ A simple and symmetric Gaussian VAE network. The width and depth are
        applied to both the encoder and decoder network.
    """

    def __init__(self, name, chpt_dir,
                       x_dims, z_dims, c_dims,
                       layer_sizes, activ, clamp_out ):
        super(VAE_Network, self).__init__()

        ## Defining the network features
        self.name      = name
        self.chpt_dir  = chpt_dir
        self.full_nm   = os.path.join(self.chpt_dir, self.name)
        self.x_dims    = x_dims
        self.z_dims    = z_dims
        self.c_dims    = c_dims
        self.clamp_out = clamp_out

        ## If the data has a sinle dimentions then we are using a symmetric MLP
        ## Where both the input and output dimentions are equal to the data
        if len(x_dims)==1:
            self.conv_net = False
            enc_ins  = x_dims[0]

        ## If the data dimensions indicates that we need a convolutional architecture then we prepare
        ## A CNN encoder and a t-CNN decoder, the overall structure is still symmetric
        else:
            self.conv_net = True
            channels = [x_dims[0]] + [32,16]
            kernels  = [6,4]
            strides  = [2,2]
            enc_ins  = calc_cnn_out_dim( x_dims, channels, kernels, strides )

        ## Defining the CNN network segments
        if self.conv_net:
            self.cnv_enc = cnn_creator( "cnv_enc", channels, kernels, strides, activ )
            channels.reverse(), kernels.reverse(), strides.reverse()
            self.cnv_dec = cnn_creator( "cnv_dev", channels, kernels, strides, activ, tpose=True )

        ## Defining the MLP network segments
        self.mlp_enc = mlp_creator( "mlp_enc", n_in=enc_ins, n_out=2*z_dims, custom_size=layer_sizes, act_h=activ )
        layer_sizes.reverse()
        self.mlp_dec = mlp_creator( "mlp_dec", n_in=z_dims+c_dims, n_out=enc_ins, custom_size=layer_sizes, act_h=activ )

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        print("\n\nNetwork structure:")
        print(self)
        print("\n")

    def encode(self, data):

        ## The information is passed through the encoding conv net if need be and then flattened
        ## We save the output size as it will be used to reshape the mlp output for the t-CNN
        if self.conv_net:
            data = self.cnv_enc(data)
            self.reshape_size = data.shape[1:]
            data = data.view(data.size(0), -1)

        ## The information is propagated through the mlp endocer network
        latent_stats = self.mlp_enc(data)

        ## The output is split into the seperate means and (log) stds components
        means, log_stds = T.chunk(latent_stats, 2, dim=-1)

        ## We now sample in the latent space based on these statistics
        gaussian_dist = T.distributions.Normal( means, log_stds.exp() )
        z = gaussian_dist.rsample()

        ## We return the latent space sample and the stats of the distribution
        return z, means, log_stds

    def decode(self, z, c_info=None):

        ## We add in our conditional information if we want to
        if self.c_dims > 0:
            z = T.cat((z, c_info), 1)

        ## The information is passed through the mlp decoder
        recon = self.mlp_dec(z)

        ## The information is unflattened and passed through the decoding conv net if need be
        if self.conv_net:
            recon = recon.view(recon.size(0), *self.reshape_size)
            recon = self.cnv_dec(recon)

        ## The output is clamped as required
        if self.clamp_out:
            recon = T.sigmoid(recon)

        return recon

    def forward(self, data, c_info=None):
        latent_samples, latent_means, latent_stds = self.encode(data)
        reconstruction = self.decode(latent_samples, c_info)
        return reconstruction, latent_means, latent_stds

    def save_checkpoint(self, flag=""):
        T.save(self.state_dict(), self.full_nm+"_"+flag)

    def load_checkpoint(self, flag=""):
        self.load_state_dict(T.load(self.full_nm+"_"+flag))

def mlp_creator( name, n_in=1, n_out=None, d=1, w=256,
                       act_h=nn.ReLU(), act_o=None, l_nrm=False,
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
        if l_nrm:
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


def cnn_creator( name, channels, kernels, strides, act, tpose = False ):

    ## Calculate the depth of the network
    d = len(channels) - 1
    layers = []

    ## Check that all settings match the depth
    if d != len(kernels):
        print("\n\n Conv net specifications do not have equal length!\n\n")
        return 1

    ## The layer type
    if tpose:
        layer = nn.ConvTranspose2d
    else:
        layer = nn.Conv2d

    for l in range(d):

        ## We add the convulional layer
        layers.append(( "{}_conv2d_{}".format(name, l+1),
                        layer( in_channels=channels[l], out_channels=channels[l+1],
                                   kernel_size=kernels[l], stride=strides[l], padding=0 )
                     ))

        ## We do not add an activation function in the final layer of a transposed conv net
        if not tpose or l != d-1:
            layers.append(( "{}_act_{}".format(name, l+1), act ))

    return nn.Sequential(OrderedDict(layers))


def calc_cnn_out_dim( x_dims, channels, kernels, strides, padding=0 ):
    """ A function to return the exact number of outputs from a CNN
    """
    n_in = x_dims[-1]

    ## We now loop through the layers and calculate the number of outputs
    for k,s in zip(kernels, strides):
        n_in = ( n_in + 2*padding - k ) / s + 1

    ## Finnaly we square the output and multiply by the final channel number
    ## This is because all channels are flattened into one array
    n_out = int(n_in*n_in*channels[-1])

    return n_out
