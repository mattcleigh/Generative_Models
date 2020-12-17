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

class AE_Network(nn.Module):
    """ A simple and symmetric Gaussian VAE network. The width and depth are
        applied to both the encoder and decoder network.
    """

    def __init__( self, name, var, do_cnn,
                  x_dims, z_dims, c_dims, clamp_out,
                  act, mlp_layers, cnn_layers,
                  drpt, lnrm, bnrm ):
        super(AE_Network, self).__init__()

        ## Defining the network features
        self.__dict__.update(locals())
        self.resize = None

        ## We resize the mlp input/ouptut dimension to match the data or the cnn output
        mlp_dim = myUT.calc_cnn_out_dim( x_dims, cnn_layers ) if self.do_cnn else x_dims[0]

        ## Defining the CNN and mlp encoder network
        if self.do_cnn: self.cnv_enc = myUT.cnn_creator( "cnv_encoder", x_dims[0], cnn_layers, act, bnrm )
        self.mlp_enc = myUT.mlp_creator( "mlp_encoder", n_in=mlp_dim+c_dims, n_out=(1+var)*z_dims,
                                         custom_size=mlp_layers, act_h=act, drpt=drpt, lnrm=lnrm )

        ## Reversing the layer structure so that the network is symmetrical
        mlp_layers.reverse(), cnn_layers.reverse()

        ## Defining the MLP and t-CNN decoder network
        self.mlp_dec = myUT.mlp_creator( "mlp_decoder", n_in=z_dims+c_dims, n_out=mlp_dim,
                                         custom_size=mlp_layers, act_h=act, drpt=drpt, lnrm=lnrm )
        if self.do_cnn: self.cnv_dec = myUT.cnn_creator( "cnv_decoder", x_dims[0], cnn_layers, act, bnrm, tpose=True )

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        print("\n\nNetwork structure: {}".format(self.name))
        print(self)
        print("\n")

    def encode(self, data, c_info=None):
        ## The information is passed through the encoding conv net if need be and then flattened
        ## We save the output size as it will be used to reshape the mlp output for the t-CNN
        if self.do_cnn:
            data = self.cnv_enc(data)
            data = data.view(data.size(0), -1)

        ## We may now add conditional information
        if c_info is not None and self.c_dims>0:
            data = T.cat((data, c_info), 1)

        ## The information is propagated through the mlp endocer network
        z_outs = self.mlp_enc(data)

        ## If we are dealing with a variational autoencoder we need to sample
        if self.var:
            ## The output is split into the seperate means and (log) stds components
            z_means, z_log_stds = T.chunk(z_outs, 2, dim=-1)
            ## We sample the latent space based on these statistics
            gaussian_dist = T.distributions.Normal( z_means, z_log_stds.exp() )
            z_values = gaussian_dist.rsample()
            ## We return the latent space sample and the stats of the distribution
            return z_values, z_means, z_log_stds

        return z_outs, None, None

    def decode(self, z, c_info=None):
        ## We add in our conditional information if specified
        if c_info is not None and self.c_dims>0:
            z = T.cat((z, c_info), 1)

        ## The information is passed through the mlp decoder
        recon = self.mlp_dec(z)

        ## The information is unflattened and passed through the decoding conv net if specified
        if self.do_cnn:
            if self.resize is None:
                data = self.cnv_enc(T.zeros([1]+self.x_dims).to(self.device))
                self.resize = data.shape[1:]
            recon = recon.view(recon.size(0), *self.resize)
            recon = self.cnv_dec(recon)

        ## The output is clamped as required
        if self.clamp_out:
            recon = T.sigmoid(recon)

        return recon

    def forward(self, data, c_info=None):
        z_values, z_means, z_log_stds = self.encode(data, c_info)
        recons = self.decode(z_values, c_info)
        return recons, z_values, z_means, z_log_stds

    def save_checkpoint(self, flag=""):
        T.save(self.state_dict(), self.full_nm+"_"+flag)

    def load_checkpoint(self, flag=""):
        self.load_state_dict(T.load(self.full_nm+"_"+flag))

class DIS_Network(nn.Module):
    """ A discriminator neural network for a GAN setup.
    """

    def __init__( self, name, do_cnn,
                  x_dims, c_dims,
                  act, mlp_layers, cnn_layers,
                  drpt, lnrm, brnm ):
        super(DIS_Network, self).__init__()

        ## Defining the network features
        self.__dict__.update(locals())

        ## We resize the mlp input/ouptut dimension to match the data or the cnn output
        mlp_dim = myUT.calc_cnn_out_dim( x_dims, cnn_layers ) if self.do_cnn else x_dims[0]

        ## Defining the network structure
        if self.do_cnn: self.cnv_net = myUT.cnn_creator( "cnv_net", x_dims[0], cnn_layers, act, brnm )
        self.mlp_net = myUT.mlp_creator( "mlp_net", n_in=mlp_dim+c_dims, n_out=1, custom_size=mlp_layers,
                                         act_h=act, drpt=drpt, lnrm=lnrm )

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        print("\n\nNetwork structure: {}".format(self.name))
        print(self)
        print("\n")

    def forward(self, input, c_info=None):

        ## The input is passed through the cnn and flattened
        if self.do_cnn:
            input = self.cnv_net(input)
            input = input.view(input.size(0), -1)

        ## We may now add conditional information
        if c_info is not None and self.c_dims>0:
            input = T.cat((input, c_info), 1)

        ## The flattened tensor is then passed through the mlp
        output = self.mlp_net(input)

        return output

    def save_checkpoint(self, flag=""):
        T.save(self.state_dict(), self.name+"_"+flag)

    def load_checkpoint(self, flag=""):
        self.load_state_dict(T.load(self.name+"_"+flag))

class BIBAE_Cluster(nn.Module):
    """ A cluster of a deterministic AE, with two adversaries. The IO and the LS discriminator. Both are optional and
        without them this is just reduced to a simple autoencoder.
    """

    def __init__( self, name ):
        super(BIBAE_Cluster, self).__init__()
        self.__dict__.update(locals())
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.GRLambda = 0.0

    def setup_AE( self, *args):
        self.AE_net = AE_Network(*args)

    def setup_LSD( self, GRLambda, *args):
        self.LSD_lambda = GRLambda
        self.LSD_net = DIS_Network(*args)

    def setup_IOD( self, GRLambda, *args):
        self.IOD_lambda = GRLambda
        self.IOD_net = DIS_Network(*args)

    def forward( self, data, c_info=None, train_LSD=False, train_IOD=False ):

        ## Increasing the effect of the adversaries
        if self.GRLambda < 1.0 and (train_IOD or train_LSD):
            self.GRLambda += 1e-2
        elif self.GRLambda > 1.0:
            self.GRLambda = 1.0

        ## First we pass the data through the AE network
        recons, z_values, z_means, z_log_stds = self.AE_net(data, c_info )

        ## If there is an LSD network then we pass it through after a GRL
        LSD_real = 0
        LSD_fake = 0
        if hasattr(self, "LSD_net") and train_LSD:

            ## Generate numbers in a given pattern
            n_samples = myUT.get_random_samples( z_values.shape, "disk" ).to(self.device)

            LSD_real = self.LSD_net( n_samples, None )
            LSD_fake = self.LSD_net( GRL(self.GRLambda)(z_values), None )

        ## Then if there is an IOD network then we pass it through after a GRL
        IOD_real = 0
        IOD_fake = 0
        if hasattr(self, "IOD_net") and train_IOD:
            IOD_real = self.IOD_net( data, c_info )
            IOD_fake = self.IOD_net( GRL(self.GRLambda)(recons), c_info )

        ## The output of all stages are returned
        return recons, z_values, z_means, z_log_stds, LSD_real, LSD_fake, IOD_real, IOD_fake

    def save_checkpoint(self, flag=""):
        T.save(self.state_dict(), self.name+"_"+flag)

    def load_checkpoint(self, flag=""):
        self.GRLambda = 1.0
        self.load_state_dict(T.load(self.name+"_"+flag))

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
        dx = -lambda_ * grads
        return dx, None

class GRL(nn.Module):
    def __init__(self, lambda_=1):
        super(GRL, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GRF.apply(x, self.lambda_)
