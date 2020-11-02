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

    def setup_training(self, burn_in, disc_ranges ):
        """ A function which sets up some variables used in training, including
            the targeted range of discrinimator accuracies.
            This function is required only for trianing.
        """
        self.train_target = "VAE"
        self.epochs_trained = 0
        self.burn_in   = burn_in
        self.disc_max  = max(disc_ranges)
        self.disc_min  = min(disc_ranges)
        self.has_iodis = hasattr(self, "IODIS_Net")
        self.has_lzdis = hasattr(self, "LZDIS_Net")

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
        self.clamped     = clamped
        self.target_dims  = target_dims
        self.targ_onehot  = targ_onehot

        ## Update the name of the model to include the dataset it is working on
        self.name += "_" + dataset_name
        ## Getting the training and testing dataloaders
        self.train_loader, self.test_loader = myDS.load_dataset( dataset_name, batch_size, n_workers )

        ## A list of examples to save for use in visualisation of the reconstruction and the latent space
        idxes = np.random.randint(0,1000,4)
        self.vis_data     = T.stack([self.test_loader.dataset[i][0] for i in idxes ])
        self.vis_targets  = T.tensor([self.test_loader.dataset[i][1] for i in idxes ], dtype=T.int64)
        self.latent_means   = 0
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
        self.VAErec_loss_fn = myNN.VAELoss(do_mse, KLD_weight)
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

        ## The discriminator loss function and optimiser
        self.IODIS_loss_fn = nn.CrossEntropyLoss()
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

        ## The discriminator loss function and optimiser
        self.LZDIS_loss_fn = nn.CrossEntropyLoss()
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

    def set_train_target(self):
        """ We work out which network needs training based on how the discriminators are doing.
        """

        ## First we see if we are still in the burn in phase of training
        if self.epochs_trained<self.burn_in:
            self.train_target = "VAE"
            return 0

        # ## Do we have an IO Discriminator to train
        # if self.has_iodis:
        #     disc_acc = self.IODIS_trn_hist[-1]
        #     if   disc_acc>self.disc_max and self.train_target=="IODIS": self.train_target = "VAE"
        #     elif disc_acc<self.disc_min and self.train_target=="VAE":   self.train_target = "IODIS"
        #
        # ## Do we have an LZ Discriminator to train
        # if self.has_lzdis:
        #     disc_acc = self.LZDIS_trn_hist[-1]
        #     if   disc_acc>self.disc_max and self.train_target=="LZDIS": self.train_target = "VAE"
        #     elif disc_acc<self.disc_min and self.train_target=="VAE":   self.train_target = "LZDIS"

        if self.has_iodis:
            disc_acc = self.IODIS_trn_hist[-1]
            if disc_acc<self.disc_min:
                self.train_target = "IODIS"
                return 0
            elif disc_acc>self.disc_max:
                self.train_target = "VAE"
                return 0
        if self.has_lzdis:
            disc_acc = self.LZDIS_trn_hist[-1]
            if disc_acc<self.disc_min:
                self.train_target = "LZDIS"
                return 0
            elif disc_acc>self.disc_max:
                self.train_target = "VAE"
                return 0

        self.train_target = np.random.choice(["IODIS","LZDIS","VAE"])

    def training_epoch(self):
        """ This function performs one epoch of training on data provided by the train_loader
        """
        ## Put all both networks into training mode
        self.change_mode( "train" )

        ## We collect running losses/accuracies for each model
        run_vae_loss  = 0
        run_iodis_acc = 0
        run_lzdis_acc = 0

        ## We work out which network is being trained this epoch
        self.set_train_target()
        print( "Training Target: ", self.train_target )

        ## Now we cycle through each minibatch
        for (data, targets) in tqdm(self.train_loader, desc="Training", ncols=80, unit=""):

            ## We prepare the input and conditional data (flatten, one-hot, move to device, etc)
            data     = self.prepare_data(data)
            cond_info = self.prepare_conditional(targets)

            ## We calculate the reconstructed output and latent stats and error of the VAE
            reconstructions, z_samples, z_means, z_log_stds = self.VAE_Net(data, cond_info)
            vae_loss = self.VAErec_loss_fn( reconstructions, data, z_means, z_log_stds )
            run_vae_loss += vae_loss.item()

            ## We check if we are out of the burn in period
            if self.epochs_trained>=self.burn_in:

                ## We check if we have an IO discriminator loss to add
                if self.has_iodis:
                    iodis_output, identities = self.IODIS_Net( reconstructions, data, cond_info )
                    iodis_loss = self.IODIS_loss_fn(iodis_output,identities)

                    ## We see how many the discriminator got correct
                    pred = T.argmax(iodis_output.data, 1)
                    run_iodis_acc += (pred==identities).sum().item()

                    ## And we modify the generator's loss function
                    vae_loss -= self.IODIS_weight * T.log(iodis_loss)

                ## We check if we have an latent space discriminator loss to add
                if self.has_lzdis:
                    norm_samples = T.normal(0, 1, size=z_samples.shape).to(self.LZDIS_Net.device)
                    lzdis_output, identities = self.LZDIS_Net( z_samples, norm_samples, None )
                    lzdis_loss = self.LZDIS_loss_fn(lzdis_output, identities)

                    ## We see how many the discriminator got correct
                    pred = T.argmax(lzdis_output.data, 1)
                    run_lzdis_acc += (pred==identities).sum().item()

                    ## And we modify the generator's loss function
                    vae_loss -= self.LZDIS_weight * T.log(lzdis_loss)

            ## Now we perform gradient descent on the target
            if self.train_target == "VAE":
                self.VAE_optimiser.zero_grad()
                vae_loss.backward()
                self.VAE_optimiser.step()

            if self.train_target == "IODIS":
                self.IODIS_optimiser.zero_grad()
                iodis_loss.backward()
                self.IODIS_optimiser.step()

            if self.train_target == "LZDIS":
                self.LZDIS_optimiser.zero_grad()
                lzdis_loss.backward()
                self.LZDIS_optimiser.step()

        ## At the end of the epoch we update the running stats
        self.VAE_trn_hist.append( run_vae_loss / len(self.train_loader) )
        if self.has_iodis: self.IODIS_trn_hist.append( run_iodis_acc / len(self.train_loader.dataset) )
        if self.has_lzdis: self.LZDIS_trn_hist.append( run_lzdis_acc / len(self.train_loader.dataset) )

        ## The epoch counter is incremented
        self.epochs_trained += 1

    def testing_epoch(self):
        """ This function performs one epoch of testing on data provided by the test_loader.
            It is basically the same as the train functin above but without the gradient desc
        """
        with T.no_grad():
            self.change_mode( "eval" )
            run_vae_loss  = 0
            run_iodis_acc = 0
            run_lzdis_acc = 0
            for (data, targets) in tqdm(self.test_loader, desc="Testing", ncols=80, unit=""):

                data      = self.prepare_data(data)
                cond_info = self.prepare_conditional(targets)

                reconstructions, z_samples, z_means, z_log_stds = self.VAE_Net(data, cond_info)
                vae_loss = self.VAErec_loss_fn( reconstructions, data, z_means, z_log_stds )
                run_vae_loss += vae_loss.item()

                if self.epochs_trained>self.burn_in:

                    if self.has_iodis:
                        iodis_output, identities = self.IODIS_Net( reconstructions, data, cond_info )
                        iodis_loss = self.IODIS_loss_fn(iodis_output,identities)
                        pred = T.argmax(iodis_output.data, 1)
                        run_iodis_acc += (pred==identities).sum().item()

                    if self.has_lzdis:
                        norm_samples = T.normal(0, 1, size=z_samples.shape).to(self.LZDIS_Net.device)
                        lzdis_output, identities = self.LZDIS_Net( z_samples, norm_samples, None )
                        lzdis_loss = self.LZDIS_loss_fn(lzdis_output, identities)
                        pred = T.argmax(lzdis_output.data, 1)
                        run_lzdis_acc += (pred==identities).sum().item()

            self.VAE_tst_hist.append( run_vae_loss / len(self.test_loader) )
            if self.has_iodis: self.IODIS_tst_hist.append( run_iodis_acc / len(self.test_loader.dataset) )
            if self.has_lzdis: self.LZDIS_tst_hist.append( run_lzdis_acc / len(self.test_loader.dataset) )

            ## We also update the information used for visualisation based on the last batch
            self.latent_means = z_means.cpu()
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
        if self.has_iodis: iodis_loss_plot = myPL.loss_plot( self.IODIS_Net.name )
        if self.has_lzdis: lzdis_loss_plot = myPL.loss_plot( self.LZDIS_Net.name )

        ## We run the training loop indefinetly
        for epoch in count(1):
            print( "\nEpoch: {}".format(epoch) )

            ## Run the test/train cycle
            self.training_epoch()
            self.testing_epoch()

            ## Update the visualisation graphs
            rp.update( self.visualise_recreation() )
            if vis_z>0 and epoch%vis_z==0:
                zp.update( self.latent_targets, self.latent_means )

            ## Update the loss/accuracy plots
            vae_loss_plot.update( self.VAE_tst_hist, self.VAE_trn_hist )
            if self.has_iodis: iodis_loss_plot.update( self.IODIS_tst_hist, self.IODIS_trn_hist )
            if self.has_lzdis: lzdis_loss_plot.update( self.LZDIS_tst_hist, self.LZDIS_trn_hist )

            ## We save the latest version of the networks
            self.save_models("latest")
