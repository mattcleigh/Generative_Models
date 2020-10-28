import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Networks as myNN
from Resources import Datasets as myDS

import os
import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from collections import OrderedDict, deque

class Agent(object):
    def __init__(self,
                 name,
                 net_dir,
                 \
                 dataset_name,
                 data_dims, flatten_data,
                 latent_dims,
                 condit_dims, targ_onehot,
                 clamp_out,
                 \
                 layer_sizes, active,
                 lr,
                 \
                 channels, kernels,
                 strides, padding,
                 \
                 reg_weight,
                 \
                 batch_size, n_workers,
                 ):

        ## Setting all class variables
        self.__dict__.update(locals())

        ## Getting the training and testing dataloaders
        self.train_loader, self.test_loader = myDS.load_dataset( dataset_name, batch_size, n_workers )

        ## The VAE network itself
        self.VAE = myNN.VAE_Network( name, net_dir, data_dims, latent_dims,
                                     condit_dims, layer_sizes, active, clamp_out,
                                     channels, kernels, strides, padding )

        ## The optimiser and reconstruction loss function
        self.VAE_optimiser = optim.Adam( self.VAE.parameters(), lr=lr )
        self.rec_loss_fn = nn.MSELoss( reduction="sum" )

        ## Running history of the test and training losses per epoch, for graphing
        self.epochs_trained = 0
        self.trn_loss_hist  = deque( maxlen=100 )
        self.tst_loss_hist  = deque( maxlen=100 )

        ## A list of examples to save for use in visualisation of the reconstruction and the latent space
        self.recon_data     = T.stack([self.test_loader.dataset[i][0] for i in range(4) ])
        self.recon_targets  = T.tensor([self.test_loader.dataset[i][1] for i in range(4) ], dtype=T.int64)
        self.latent_means   = 0
        self.latent_targets = 0

    def save_models(self, flag=""):
        self.VAE.save_checkpoint(flag)

    def load_models(self, flag=""):
        self.VAE.load_checkpoint(flag)

    def prepare_conditional(self, targets):
        """ This is used to modify the targets into conditional information,
            particularly for one-hot encoding categorial varaibles (MNIST, CIFAR)
            This function is called even if the no conditional information is used.
        """
        if self.condit_dims == 0:
            return None

        condit = targets.to(self.VAE.device)
        if self.targ_onehot:
            condit = F.one_hot(condit, num_classes = self.condit_dims)
        return condit

    def prepare_data(self, data):
        """ This is used to modify the input data for the network,
            particularly for flattening images when using an MLP
        """
        data = data.to(self.VAE.device)
        if self.flatten_data:
            data = data.view( data.size(0), -1 )
        return data

    def train(self):
        """ This function performs one epoch of training on data provided by the train_loader
        """
        self.VAE.train()

        tot_loss = 0
        for (data, targets) in tqdm(self.train_loader, desc="Training", ncols=60, unit=""):

            ## We zero out the gradients, as required for each pytorch train loop
            self.VAE_optimiser.zero_grad()

            ## We prepare the input and conditional data (flatten, one-hot, move to device, etc)
            data   = self.prepare_data(data)
            condit = self.prepare_conditional(targets)

            ## We calculate the reconstructed output and latent stats of the VAE
            reconstructions, means, log_stds = self.VAE(data, condit)

            ## We calculate the total loss of the batch by combining the reconstruction error and the regulariation
            rec_loss = self.rec_loss_fn(reconstructions, data)
            kld_loss = - 0.5 * T.sum( 1 + 2*log_stds - means*means - (2*log_stds).exp() )
            loss = rec_loss + self.reg_weight * kld_loss

            ## Perform the gradient descent step
            loss.backward()
            self.VAE_optimiser.step()

            ## Update the running loss
            tot_loss += loss.item()

        ## Update the epoch deque and the epoch counter
        self.trn_loss_hist.append( tot_loss / len(self.train_loader.dataset) )
        self.epochs_trained += 1

    def test(self):
        """ This function performs one epoch of testing on data provided by the test_loader.
            It is basically the same as the train functin above but without the gradient desc
        """
        self.VAE.eval()
        with T.no_grad():
            tot_loss = 0
            for (data, targets) in tqdm(self.test_loader, desc="Testing ", ncols=60, unit=""):
                data  = self.prepare_data(data)
                condit = self.prepare_conditional(targets)

                reconstructions, means, log_stds = self.VAE(data, condit)

                rec_loss = self.rec_loss_fn(reconstructions, data)
                kld_loss = - 0.5 * T.sum( 1 + 2*log_stds - means*means - (2*log_stds).exp() )
                loss = rec_loss + self.reg_weight * kld_loss

                tot_loss += loss.item()

        self.tst_loss_hist.append( tot_loss / len(self.test_loader.dataset) )

        ## Graphs and visualisations are based on the final batch in the test set
        self.latent_means = means.cpu()
        self.latent_targets = targets

    def visualise_recreation(self):
        """ This function returns reconstructions of 4 selected examples and is called
            during training after each epoch. Good for images.
        """
        with T.no_grad():
            data = self.prepare_data( self.recon_data )
            condit_info = self.prepare_conditional(self.recon_targets)
            reconstructions, *_ = self.VAE(data, condit_info)
            return reconstructions.view(self.recon_data.shape).cpu()
