import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Networks as myNN
from Resources import Datasets as myDS
from Resources import Plotting as myPL

import os
import numpy as np
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from itertools import count
from collections import OrderedDict, deque

class BIB_AE(object):
    def __init__(self, name, save_dir ):

        self.name = name
        self.save_dir = save_dir

    def setup_training(self, burn_in, disc_range ):
        """ A function which sets up some variables used in training, including
            the targeted range of discrinimator accuracies.
            This function is required only for trianing.
        """
        self.epochs_trained = 0
        self.burn_in   = burn_in
        self.has_iodis = hasattr(self, "IODIS_Net")
        self.has_lzdis = hasattr(self, "LZDIS_Net")
        self.disc_max = max(disc_range)
        self.disc_min = min(disc_range)
        self.run_trn_loss = np.zeros(4)
        self.run_trn_acc  = np.zeros(2)
        self.run_tst_loss = np.zeros(4)
        self.run_tst_acc  = np.zeros(2)

    def initiate_dataset( self, dataset_name,
                          data_dims, flatten_data, clamped,
                          target_dims, targ_onehot,
                          n_workers, batch_size ):
        """ Calls the load_dataset method to give the cluster pytorch dataloaders
            for both the training and the test set, also loads some examples to use for plotting.
            This is required for the program to execute
        """
        self.data_dims    = data_dims
        self.flatten_data = flatten_data
        self.clamped      = clamped
        self.target_dims  = target_dims
        self.targ_onehot  = targ_onehot

        ## Update the name of the model to include the dataset it is working on
        self.name += "_" + dataset_name

        ## Getting the training and testing dataloaders
        self.train_loader, self.test_loader = myDS.load_dataset( dataset_name, batch_size, n_workers )

        ## A list of examples to save for use in visualisation of the reconstruction and the latent space
        idxes = np.random.randint(0,1000,4)
        self.vis_data = T.stack([self.train_loader.dataset[i][0] for i in idxes ])
        self.vis_targets = T.tensor([self.train_loader.dataset[i][1] for i in idxes ], dtype=T.int64)
        self.latent_means = 0
        self.latent_targets = 0

        ## Checking if we will need a CNN structure for all later networks or if an MLP will be fine
        self.do_cnn = not( len(data_dims)==flatten_data==1 )

    def initiate_VAE( self, do_mse, latent_dims, KLD_weight,
                      lr, act,
                      mlp_layers, drpt, lnrm,
                      cnn_layers, bnrm ):
        """ Give the cluster both a generator network, loss function, and optimiser.
            This is required for the program to execute
        """

        self.latent_dims = latent_dims
        self.KLD_weight = KLD_weight

        ## The VAE network
        self.VAE_Net = myNN.VAE_Network( self.name+"_VAE", self.save_dir, self.do_cnn,
                                         self.data_dims, self.latent_dims, self.target_dims, self.clamped,
                                         act, mlp_layers, drpt, lnrm,
                                         cnn_layers, bnrm )

        ## The VAE reconstruction loss function and optimiser
        self.VAErec_loss_fn = myNN.VAELoss(do_mse)
        self.VAE_optimiser  = optim.Adam( self.VAE_Net.parameters(), lr=lr )

        ## Running history of the test and training VAE losses (wont include discriminator losses!)
        self.VAE_trn_hist = deque( maxlen=100 )
        self.VAE_tst_hist = deque( maxlen=100 )

    def initiate_IO_Disc( self, weight, lr, act,
                          mlp_layers, drpt, lnrm,
                          cnn_layers, bnrm ):
        """ Give the cluster both an adversarial discriminator network for the input and output.
            This is NOT required for the program to execute.
        """
        self.IODIS_weight = weight

        ## The Discriminator network
        self.IODIS_Net = myNN.DIS_Network(  self.name+"_IODIS", self.save_dir, self.do_cnn,
                                            self.data_dims, self.target_dims,
                                            act, mlp_layers, drpt, lnrm,
                                            cnn_layers, bnrm )

        ## The discriminator optimiser
        self.IODIS_optimiser= optim.Adam( self.IODIS_Net.parameters(), lr=lr )

        ## Running history of the test and training VAE losses (wont include discriminator losses!)
        self.IODIS_trn_hist = deque( maxlen=100 )
        self.IODIS_tst_hist = deque( maxlen=100 )

    def initiate_LZ_Disc( self, weight, lr, act,
                          mlp_layers, drpt, lnrm ):
        """ Give the cluster an adversarial network for the latent space.
        """
        self.LZDIS_weight = weight

        ## The Discriminator network
        self.LZDIS_Net = myNN.DIS_Network( self.name+"_LZDIS", self.save_dir, False,
                                           [self.latent_dims], 0,
                                           act, mlp_layers, drpt, lnrm )

        ## The discriminator  optimiser
        self.LZDIS_optimiser= optim.Adam( self.LZDIS_Net.parameters(), lr=lr )

        ## Running history of the test and training VAE losses (wont include discriminator losses!)
        self.LZDIS_trn_hist = deque( maxlen=100 )
        self.LZDIS_tst_hist = deque( maxlen=100 )

    def save_models(self, flag=""):
        self.VAE_Net.save_checkpoint(flag)
        if self.has_iodis: self.IODIS_Net.save_checkpoint(flag)
        if self.has_lzdis: self.LZDIS_Net.save_checkpoint(flag)

    def load_models(self, flag=""):
        self.VAE_Net.load_checkpoint(flag)
        if self.has_iodis: self.IODIS_Net.load_checkpoint(flag)
        if self.has_lzdis: self.LZDIS_Net.load_checkpoint(flag)

    def prepare_conditional(self, targets):
        """ This is used to modify the targets into conditional information,
            particularly for one-hot encoding categorial varaibles (MNIST, CIFAR)
            This function is called even if the no conditional information is used.
        """
        if self.target_dims == 0:
            return None

        cond_info = targets.to(self.VAE_Net.device)
        if self.targ_onehot:
            cond_info = F.one_hot(cond_info, num_classes = self.target_dims)
        return cond_info

    def prepare_data(self, data):
        """ This is used to modify the input data for the network,
            particularly for flattening images when using an MLP
        """
        data = data.to(self.VAE_Net.device)
        if self.flatten_data:
            data = data.view( data.size(0), -1 )
        return data

    def change_mode(self, mode):
        if mode=="train":
            self.VAE_Net.train()
            if self.has_iodis: self.IODIS_Net.train()
            if self.has_lzdis: self.LZDIS_Net.train()
        else:
            self.VAE_Net.eval()
            if self.has_iodis: self.IODIS_Net.eval()
            if self.has_lzdis: self.LZDIS_Net.eval()

    def vae_step( self, data, cond_info, train=True ):

        if train:
            self.VAE_optimiser.zero_grad()

        reconstructions, z_samples, z_means, z_log_stds = self.VAE_Net(data, cond_info)
        rec_loss, kld_loss = self.VAErec_loss_fn( reconstructions, data, z_means, z_log_stds )
        kld_loss *= self.KLD_weight
        loss = rec_loss + kld_loss
        if train:
            loss.backward()
            self.VAE_optimiser.step()

        return rec_loss.item(), kld_loss.item()

    def iodis_step( self, data, cond_info, train=True ):

        ## We train the discriminator on real and fake data
        if train:
            self.IODIS_optimiser.zero_grad()

        recn, *_ = self.VAE_Net(data, cond_info)

        real_outs = self.IODIS_Net( data, cond_info )
        fake_outs = self.IODIS_Net( recn, cond_info )

        ## We calculate the accuracy of the discrinimator
        real_acc = T.round(real_outs).sum()
        fake_acc = T.round(1-fake_outs).sum()
        acc = (real_acc+fake_acc)/(2*len(data))

        ## We only perform gradient descent if the accuracy isnt too high
        if train and acc<=self.disc_max:
            loss = ( - T.log(real_outs) - T.log(1-fake_outs) ).mean()
            loss.backward()
            self.IODIS_optimiser.step()

        ## We now train the generator using fake data if the accuracy isnt too low
        gloss = T.zeros(1)
        if train and acc>=self.disc_min:
            self.VAE_optimiser.zero_grad()
            recn, *_ = self.VAE_Net(data, cond_info)
            fake_outs = self.IODIS_Net( recn, cond_info )
            gloss = -T.log(fake_outs).mean() * self.IODIS_weight
            gloss.backward()
            self.VAE_optimiser.step()

        return acc.item(), gloss.item()

    def lzdis_step( self, data, cond_info, train=True ):

        if train:
            self.LZDIS_optimiser.zero_grad()

        _, z_samples, _, _ = self.VAE_Net(data, cond_info)
        n_samples = T.normal( 0, 1, z_samples.shape ).to(self.LZDIS_Net.device)

        n_outs = self.LZDIS_Net( n_samples, None )
        z_outs = self.LZDIS_Net( z_samples, None )

        n_acc = T.round(n_outs).sum()
        z_acc = T.round(1-z_outs).sum()
        acc = (n_acc+z_acc)/(2*len(data))

        if train and acc<=self.disc_max:
            loss = ( - T.log(n_outs) - T.log(1-z_outs) ).mean()
            loss.backward()
            self.LZDIS_optimiser.step()

        gloss = T.zeros(1)
        if train and acc>=self.disc_min:
            self.VAE_optimiser.zero_grad()
            _, z_samples, _, _ = self.VAE_Net(data, cond_info)
            z_outs = self.LZDIS_Net( z_samples, None )
            gloss = -T.log(z_outs).mean() * self.LZDIS_weight
            gloss.backward()
            self.VAE_optimiser.step()


        return acc.item(), gloss.item()

    def training_epoch(self):
        """ This function performs one epoch of training on data provided by the train_loader
        """
        ## Put all both networks into training mode
        self.change_mode( "train" )

        ## We collect running losses/accuracies for each model
        self.run_trn_loss = np.zeros(4)
        self.run_trn_acc = np.zeros(2)

        ## Now we cycle through each minibatch
        for (data, targets) in tqdm(self.train_loader, desc="Training", ncols=80, unit=""):

            ## We prepare the input and conditional data (flatten, one-hot, move to device, etc)
            data = self.prepare_data(data)
            cond_info = self.prepare_conditional(targets)

            ## We do the usual VAE training step, based on MSE or BCE from inputs to outputs
            rec_loss, kld_loss = self.vae_step(data, cond_info)
            self.run_trn_loss[0] += rec_loss
            self.run_trn_loss[1] += kld_loss

            ## If we are still in the burn in period, then we dont turn on discriminators
            if self.epochs_trained<self.burn_in:
                continue

            ## We check if we have an IO discriminator
            if self.has_iodis:
                iod_acc, iod_loss = self.iodis_step(data, cond_info)
                self.run_trn_loss[2] += iod_loss
                self.run_trn_acc[0] += iod_acc

            ## We check if we have an LZ discriminator
            if self.has_lzdis:
                lzd_acc, lzd_loss = self.lzdis_step(data, cond_info)
                self.run_trn_loss[3] += lzd_loss
                self.run_trn_acc[1] += lzd_acc

        ## At the end of the epoch we update the running stats
        self.VAE_trn_hist.append( self.run_trn_loss[0] / len(self.train_loader) )
        if self.has_iodis: self.IODIS_trn_hist.append( self.run_trn_acc[0] / len(self.train_loader) )
        if self.has_lzdis: self.LZDIS_trn_hist.append( self.run_trn_acc[1] / len(self.train_loader) )

        ## The epoch counter is incremented
        self.epochs_trained += 1

    def testing_epoch(self):
        """ This function performs one epoch of testing on data provided by the test_loader.
            It is basically the same as the train functin above but without the gradient desc
        """
        with T.no_grad():
            self.change_mode( "eval" )
            self.run_tst_loss = np.zeros(4)
            self.run_tst_acc = np.zeros(2)

            for (data, targets) in tqdm(self.test_loader, desc="Testing ", ncols=80, unit=""):

                data = self.prepare_data(data)
                cond_info = self.prepare_conditional(targets)
                rec_loss, kld_loss = self.vae_step(data, cond_info, train=False)
                self.run_tst_loss[0] += rec_loss
                self.run_tst_loss[1] += kld_loss

                if self.epochs_trained<self.burn_in:
                    continue

                if self.has_iodis:
                    iod_acc, iod_loss = self.iodis_step(data, cond_info, train=False)
                    self.run_tst_loss[2] += iod_loss
                    self.run_tst_acc[0] += iod_acc

                if self.has_lzdis:
                    lzd_acc, lzd_loss = self.lzdis_step(data, cond_info, train=False)
                    self.run_tst_loss[3] += lzd_loss
                    self.run_tst_acc[1] += lzd_acc

            self.VAE_tst_hist.append( self.run_tst_loss[0] / len(self.test_loader) )
            if self.has_iodis: self.IODIS_tst_hist.append( self.run_tst_acc[0] / len(self.test_loader) )
            if self.has_lzdis: self.LZDIS_tst_hist.append( self.run_tst_acc[1] / len(self.test_loader) )

            ## We also update the information used for visualisation based on the last batch
            self.latent_means = self.VAE_Net( data, cond_info )[2].cpu()
            self.latent_targets = targets

    def visualise_recreation(self):
        """ This function returns reconstructions of 4 selected examples and is called
            during training after each epoch. Good for images.
        """
        with T.no_grad():
            data = self.prepare_data( self.vis_data )
            cond_info = self.prepare_conditional(self.vis_targets)
            reconstructions, *_ = self.VAE_Net(data, cond_info)
            return reconstructions.view(self.vis_data.shape).cpu()

    def run_training_loop( self, load_flag, vis_z, dim_red ):
        """ This is the main training loop
        """
        plt.ion()

        ## For loading previous states of the network
        if load_flag is not None:
            self.load_models( load_flag )

        ## Creating the plots for the visualisation
        rp = myPL.recreation_plot( self.vis_data, self.name )
        if vis_z>0:
            zp = myPL.latent_plot( self.name, dim_red )

        ## Creating the loss/accuracy plots
        vae_loss_plot = myPL.loss_plot( self.VAE_Net.name )
        all_loss_plot = myPL.loss_contribution_plot( self.VAE_Net.name )
        if self.has_iodis: iodis_loss_plot = myPL.loss_plot( self.IODIS_Net.name )
        if self.has_lzdis: lzdis_loss_plot = myPL.loss_plot( self.LZDIS_Net.name )

        ## We run the training loop indefinetly
        for epoch in count(1):
            print( "\nEpoch: {}".format(epoch) )

            ## Run the test/train cycle
            self.testing_epoch()
            self.training_epoch()

            ## Update the visualisation graphs
            rp.update( self.visualise_recreation() )
            if vis_z>0 and epoch%vis_z==0:
                zp.update( self.latent_targets, self.latent_means )

            ## Normalise the runnin loss scores
            self.run_trn_loss = np.abs(self.run_trn_loss) / self.run_trn_loss[0]
            print( "Loss Contributions: ")
            print(" - Rec: ", self.run_trn_loss[0] )
            print(" - KLD: ", self.run_trn_loss[1] )
            if self.has_iodis: print(" - IOD: ", self.run_trn_loss[2] )
            if self.has_lzdis: print(" - LZD: ", self.run_trn_loss[3] )

            ## Update the loss/accuracy plots
            vae_loss_plot.update( self.VAE_tst_hist, self.VAE_trn_hist )
            all_loss_plot.update( self.run_trn_loss )
            if self.has_iodis: iodis_loss_plot.update( self.IODIS_tst_hist, self.IODIS_trn_hist )
            if self.has_lzdis: lzdis_loss_plot.update( self.LZDIS_tst_hist, self.LZDIS_trn_hist )

            ## We save the latest version of the networks
            self.save_models("latest")
