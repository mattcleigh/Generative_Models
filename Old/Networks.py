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

from collections import OrderedDict

class VAE_Network(nn.Module):
    """ A simple and symmetric Gaussian VAE network. The width and depth are
        applied to both the encoder and decoder network.
    """

    def __init__( self, name, net_dir, do_cnn,
                  x_dims, z_dims, c_dims, clamp_out,
                  act, mlp_layers, drpt, lnrm,
                  cnn_layers, bnrm ):
        super(VAE_Network, self).__init__()

        ## Defining the network features
        self.__dict__.update(locals())
        self.full_nm = os.path.join(self.net_dir, self.name)
        self.resize = None

        ## We resize the mlp input/ouptut dimension to match the data or the cnn output
        mlp_dim = myUT.calc_cnn_out_dim( x_dims, cnn_layers ) if self.do_cnn else x_dims[0]

        ## Defining the CNN and mlp encoder network
        if self.do_cnn: self.cnv_enc = myUT.cnn_creator( "cnv_enc", x_dims[0], cnn_layers, act, bnrm )
        self.mlp_enc = myUT.mlp_creator( "mlp_enc", n_in=mlp_dim+c_dims, n_out=2*z_dims, custom_size=mlp_layers, act_h=act, drpt=drpt, lnrm=lnrm )

        ## Reversing the layer structure so that the network is symmetrical
        mlp_layers.reverse(), cnn_layers.reverse()

        ## Defining the MLP and t-CNN decoder network
        self.mlp_dec = myUT.mlp_creator( "mlp_dec", n_in=z_dims+c_dims, n_out=mlp_dim, custom_size=mlp_layers, act_h=act, drpt=drpt, lnrm=lnrm )
        if self.do_cnn: self.cnv_dec = myUT.cnn_creator( "cnv_dev", x_dims[0], cnn_layers, act, bnrm, tpose=True )

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
            if self.resize is None: self.resize = data.shape[1:]
            data = data.view(data.size(0), -1)

        ## We may now add conditional information
        if c_info is not None:
            data = T.cat((data, c_info), 1)

        ## The information is propagated through the mlp endocer network
        z_stats = self.mlp_enc(data)

        ## The output is split into the seperate means and (log) stds components
        z_means, z_log_stds = T.chunk(z_stats, 2, dim=-1)

        ## We sample the latent space based on these statistics
        gaussian_dist = T.distributions.Normal( z_means, z_log_stds.exp() )
        z_samples = gaussian_dist.rsample()

        ## We return the latent space sample and the stats of the distribution
        return z_samples, z_means, z_log_stds

    def decode(self, z, c_info=None):
        ## We add in our conditional information if specified
        if c_info is not None:
            z = T.cat((z, c_info), 1)

        ## The information is passed through the mlp decoder
        recon = self.mlp_dec(z)

        ## The information is unflattened and passed through the decoding conv net if specified
        if self.do_cnn:
            recon = recon.view(recon.size(0), *self.resize)
            recon = self.cnv_dec(recon)

        ## The output is clamped as required
        if self.clamp_out:
            recon = T.sigmoid(recon)

        return recon

    def forward(self, data, c_info=None):
        z_samples, z_means, z_log_stds = self.encode(data, c_info)
        reconstruction = self.decode(z_samples, c_info)
        return reconstruction, z_samples, z_means, z_log_stds

    def save_checkpoint(self, flag=""):
        T.save(self.state_dict(), self.full_nm+"_"+flag)

    def load_checkpoint(self, flag=""):
        self.load_state_dict(T.load(self.full_nm+"_"+flag))

class DIS_Network(nn.Module):
    """ A discriminator neural network for a GAN setup.
    """

    def __init__( self, name, net_dir, do_cnn,
                  x_dims, c_dims,
                  act, mlp_layers, drpt, lnrm,
                  cnn_layers=[], brnm=False ):
        super(DIS_Network, self).__init__()

        ## Defining the network features
        self.__dict__.update(locals())
        self.full_nm = os.path.join(self.net_dir, self.name)

        ## We resize the mlp input/ouptut dimension to match the data or the cnn output
        mlp_dim = myUT.calc_cnn_out_dim( x_dims, cnn_layers ) if self.do_cnn else x_dims[0]

        ## Defining the network structure
        if self.do_cnn: self.cnv_net = myUT.cnn_creator( "cnv_net", x_dims[0], cnn_layers, act, brnm )
        self.mlp_net = myUT.mlp_creator( "mlp_net", n_in=2*mlp_dim+c_dims, n_out=1, custom_size=mlp_layers, act_h=act, act_h=act, drpt=drpt, lnrm=lnrm )

        ## Moving the network to the device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        print("\n\nNetwork structure: {}".format(self.name))
        print(self)
        print("\n")

    def forward(self, fakes, reals, c_info=None):

        ## The real data and the generated data randomly suffled
        combined    = T.cat( (reals.unsqueeze(0), fakes.unsqueeze(0) ), 0 )
        identities  = T.LongTensor(len(reals)).random_(0, 2).to(self.device)
        batch_idxes = list(range(len(reals)))
        input1 = combined[ identities,   batch_idxes]
        input2 = combined[ 1-identities, batch_idxes]

        ## Both inputs are passed through the same CNN layers and flattened
        if self.do_cnn:
            input1 = self.cnv_net(input1)
            input2 = self.cnv_net(input2)
            input1 = input1.view(input1.size(0), -1)
            input2 = input2.view(input2.size(0), -1)

        ## The inputs are joined together for the mlp
        input_joined = T.cat((input1, input2), 1)

        ## We may now add conditional information
        if c_info is not None:
            input_joined = T.cat((input_joined, c_info), 1)

        ## The flattened tensor is then passed through the mlp
        output = self.mlp_net(input_joined)

        return output, identities

    def save_checkpoint(self, flag=""):
        T.save(self.state_dict(), self.full_nm+"_"+flag)

    def load_checkpoint(self, flag=""):
        self.load_state_dict(T.load(self.full_nm+"_"+flag))

class VAELoss(nn.Module):
    def __init__(self, do_mse, KLD_weight):
        super(VAELoss, self).__init__()

        if do_mse:
            self.rec_loss = nn.MSELoss(reduction="sum")
        else:
            self.rec_loss = nn.BCELoss(reduction="sum")

        self.KLD_weight = KLD_weight

    def forward(self, y_pred, y_true, means, log_stds):

        ## Calculate the reconstruction error
        rec_err = self.rec_loss( y_pred, y_true) / len(y_pred)

        ## Calculate the KL Divergence loss
        kld_div = - 0.5 * T.sum( 1 + 2*log_stds - means*means - (2*log_stds).exp() )

        ## Combine and return
        return rec_err + self.KLD_weight * kld_div
