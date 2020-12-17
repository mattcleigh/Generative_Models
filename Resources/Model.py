import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Networks as myNN
from Resources import Datasets as myDS
from Resources import Plotting as myPL
from Resources import Utils    as myUT

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

class BIBAE_Agent(object):
    def __init__(self, name, save_dir ):

        self.name = os.path.join( save_dir, name )
        self.cluster = myNN.BIBAE_Cluster( self.name+"_BIBAE" )

        self.has_KLD    = False
        self.has_LSD    = False
        self.has_IOD    = False

        self.train_LSD  = False
        self.train_IOD  = False

        self.KLD_weight = 0
        self.LSD_weight = 0
        self.IOD_weight = 0

        self.AE_use_cond  = False
        self.IOD_use_cond = False

    def save_models(self, flag=""):
        self.cluster.save_checkpoint(flag)

    def load_models(self, flag=""):
        self.cluster.load_checkpoint(flag)

    def initiate_dataset( self, dataset_name,
                          data_dims, flatten_data, clamped,
                          class_dims, class_onehot,
                          n_workers, batch_size ):
        """ Calls the load_dataset method to give the cluster pytorch dataloaders
            for both the training and the test set, also loads some examples to use for plotting.
            This is required for the program to execute
        """
        self.data_dims     = data_dims
        self.flatten_data  = flatten_data
        self.clamped       = clamped
        self.class_dims    = class_dims
        self.class_onehot  = class_onehot

        ## Update the name of the model to include the dataset it is working on
        self.name += "_" + dataset_name

        ## Getting the training and testing dataloaders
        self.train_loader, self.test_loader, self.unorm_trans = myDS.load_dataset( dataset_name, batch_size, n_workers )

        ## A list of examples to save for use in visualisation of the reconstruction and the latent space
        idxes = np.random.randint(0,len(self.train_loader.dataset),4)
        self.vis_data    = T.stack( [self.train_loader.dataset[i][0] for i in idxes ])
        self.vis_classes = T.tensor([self.train_loader.dataset[i][1] for i in idxes ], dtype=T.int64)
        self.latent_classes = 0

        ## Checking if we will need a CNN structure for all later networks or if an MLP will be fine
        self.do_cnn = not( len(data_dims)==flatten_data==1 )

    def initiate_AE( self, variational, KLD_weight, cyclical,
                     latent_dims, use_cond,
                     act, mlp_layers, cnn_layers,
                     drpt, lnrm, bnrm ):
        """ This initialises the autoencoder
        """

        ## The latent space dimension is needed throughout the training steps
        self.has_KLD = variational
        self.KLD_weight  = KLD_weight
        self.cyclical    = cyclical
        self.latent_dims = latent_dims
        self.AE_use_cond = use_cond
        c_dims = self.class_dims if use_cond else 0

        ## Giving the cluster an autoencoding network
        self.cluster.setup_AE( self.name+"_AE", variational, self.do_cnn,
                               self.data_dims, self.latent_dims, c_dims, self.clamped,
                               act, mlp_layers, cnn_layers,
                               drpt, lnrm, bnrm )

        ## The AE reconstruction loss function and optimiser
        self.AE_loss_fn = nn.MSELoss(reduction="mean")

        ## Running history of the test and training AE losses
        self.AE_trn_hist = deque( maxlen=100 )
        self.AE_tst_hist = deque( maxlen=100 )
        self.KLD_trn_hist = deque( maxlen=100 )
        self.KLD_tst_hist = deque( maxlen=100 )

    def initiate_LSD( self, GRLambda, weight,
                      act, mlp_layers, drpt, lnrm ):
        """ Give the cluster an adversarial network for the latent space.
        """

        ## The discriminator contribution to the total loss function
        self.has_LSD = True
        self.train_LSD = False
        self.LSD_weight = weight

        ## The Discriminator network
        self.cluster.setup_LSD( GRLambda, self.name+"_LSD", False,
                                [self.latent_dims], 0,
                                act, mlp_layers, [], drpt, lnrm, False )

        ## The discriminar reconstruction loss function and optimiser
        self.LSD_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        ## Running history of the test and training LSD losses
        self.LSD_trn_hist = deque( maxlen=100 )
        self.LSD_tst_hist = deque( maxlen=100 )

    def initiate_IOD( self, GRLambda, weight, use_cond,
                      act, mlp_layers, cnn_layers,
                      drpt, lnrm, bnrm ):
        """ Give the cluster both an adversarial discriminator network for the input and output.
        """

        ## The discriminator contribution to the total loss function
        self.has_IOD      = True
        self.train_IOD    = False
        self.IOD_weight   = weight
        self.IOD_use_cond = use_cond
        c_dims = self.class_dims if use_cond else 0

        ## Giving the cluster the IO discriminator network
        self.cluster.setup_IOD( GRLambda, self.name+"_IOD", self.do_cnn,
                                self.data_dims, c_dims,
                                act, mlp_layers, cnn_layers,
                                drpt, lnrm, bnrm )

        ## The discriminar reconstruction loss function and optimiser
        self.IOD_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        ## Running history of the test and training IOD losses
        self.IOD_trn_hist = deque( maxlen=100 )
        self.IOD_tst_hist = deque( maxlen=100 )

    def setup_training(self, lr, burn_in, change_lr, clip_grad ):
        """ A function which sets up some variables used in training, including
            the targeted range of discrinimator accuracies.
            This function is required only for trianing.
        """
        self.optimiser = optim.Adam( self.cluster.parameters(), lr=lr, betas=(0.5,0.99) )
        self.change_lr = change_lr
        self.burn_in = burn_in
        self.clip_grad = clip_grad
        self.epochs_trained = 0

    def prepare_conditional(self, classes):
        """ This is used to modify the classes into conditional information,
            particularly for one-hot encoding categorial varaibles (MNIST, CIFAR)
            This function is called even if the no conditional information is used.
        """

        ## We check if conditional information is used by any of our networks
        if self.AE_use_cond==self.IOD_use_cond==False:
            return None

        cond_info = classes.to(self.cluster.device)
        if self.class_onehot:
            cond_info = F.one_hot(cond_info, num_classes = self.class_dims)
        return cond_info

    def prepare_data(self, data):
        """ This is used to modify the input data for the network,
            particularly for flattening images when using an MLP
        """
        data = data.to(self.cluster.device)
        if self.flatten_data:
            data = data.view( data.size(0), -1 )
        return data

    def set_discrinimator_flags(self):

        ## We check if we have passed the burn in period
        if self.epochs_trained == self.burn_in:
            if self.has_LSD: self.train_LSD = True
            if self.has_IOD: self.train_IOD = True
            if self.change_lr is not None:
                self.optimiser.param_groups[0]['lr'] = self.change_lr


    def training_epoch(self):
        """ This function performs one epoch of training on data provided by the train_loader
        """
        ## Put all networks into training mode
        self.cluster.train()

        ## We collect running losses for each model
        running_loss = np.zeros(4)

        ## Now we cycle through each minibatch
        for (data, classes) in tqdm(self.train_loader, desc="Training", ncols=80, unit=""):

            ## We zero out the gradients
            self.optimiser.zero_grad()

            ## We prepare the input and conditional data (flatten, one-hot, move to device, etc)
            data = self.prepare_data(data)
            cond_info = self.prepare_conditional(classes)

            ## We forward propagate the data through the entire cluster
            (recons, z_values, z_means, z_log_stds,
                     LSD_real, LSD_fake, IOD_real, IOD_fake) = self.cluster(data, cond_info, self.train_LSD, self.train_IOD)

            ## Generate class labels for discriminators and placeholder loss
            KLD_loss = T.tensor(0.0)
            LSD_loss = T.tensor(0.0)
            IOD_loss = T.tensor(0.0)
            if (self.train_LSD or self.train_IOD):
                ones  = T.ones(  [len(recons),1], dtype=T.float32, device=self.cluster.device)
                zeros = T.zeros( [len(recons),1], dtype=T.float32, device=self.cluster.device)

            ## Now we calculate actual loss terms
            AE_loss = self.AE_loss_fn( recons, data )
            if self.has_KLD:   KLD_loss = 0.5 * T.mean( z_means*z_means + (2*z_log_stds).exp() - 2*z_log_stds - 1 )
            if self.train_LSD: LSD_loss = 0.5 * ( self.LSD_loss_fn( LSD_real, ones) + self.LSD_loss_fn( LSD_fake, zeros) )
            if self.train_IOD: IOD_loss = 0.5 * ( self.IOD_loss_fn( IOD_real, ones) + self.IOD_loss_fn( IOD_fake, zeros) )

            ## The loss funcitons are combined for the whole network, and grad desc is performed
            Total_loss = AE_loss + self.KLD_weight*KLD_loss + self.LSD_weight*LSD_loss + self.IOD_weight*IOD_loss
            Total_loss.backward()

            ## Clipping on the absolute value of the gradients
            if self.clip_grad > 0:
                nn.utils.clip_grad_value_(self.cluster.parameters(), self.clip_grad)
            self.optimiser.step()

            ## We update the running losses for plotting
            running_loss += np.array([ AE_loss.item(), KLD_loss.item(), LSD_loss.item(), IOD_loss.item() ])

        ## At the end of the epoch we update the stats
        self.AE_trn_hist.append( running_loss[0] / len(self.train_loader) )
        if self.has_KLD: self.KLD_trn_hist.append( running_loss[1] / len(self.train_loader) )
        if self.has_LSD: self.LSD_trn_hist.append( running_loss[2] / len(self.train_loader) )
        if self.has_IOD: self.IOD_trn_hist.append( running_loss[3] / len(self.train_loader) )

        ## The epoch counter is incremented
        self.epochs_trained += 1

    def testing_epoch(self):
        """ This function performs one epoch of testing on data provided by the test_loader.
            It is basically the same as the train functin above but without the gradient desc
        """
        with T.no_grad():
            self.cluster.eval()
            running_loss = np.zeros(4)

            for (data, classes) in tqdm(self.test_loader, desc="Testing ", ncols=80, unit=""):

                data = self.prepare_data(data)
                cond_info = self.prepare_conditional(classes)

                (recons, z_values, z_means, z_log_stds,
                         LSD_real, LSD_fake, IOD_real, IOD_fake) = self.cluster(data, cond_info, self.train_LSD, self.train_IOD)

                KLD_loss = T.tensor(0.0)
                LSD_loss = T.tensor(0.0)
                IOD_loss = T.tensor(0.0)
                if (self.train_LSD or self.train_IOD):
                    ones  = T.ones(  [len(recons),1], dtype=T.float32, device=self.cluster.device)
                    zeros = T.zeros( [len(recons),1], dtype=T.float32, device=self.cluster.device)

                AE_loss = self.AE_loss_fn( recons, data )
                if self.has_KLD:   KLD_loss = 0.5 * T.mean( z_means*z_means + (2*z_log_stds).exp() - 2*z_log_stds - 1 )
                if self.train_LSD: LSD_loss = 0.5 * ( self.LSD_loss_fn( LSD_real, ones) + self.LSD_loss_fn( LSD_fake, zeros) )
                if self.train_IOD: IOD_loss = 0.5 * ( self.IOD_loss_fn( IOD_real, ones) + self.IOD_loss_fn( IOD_fake, zeros) )

                running_loss += np.array([ AE_loss.item(), KLD_loss.item(), LSD_loss.item(), IOD_loss.item() ])

            ## At the end of the epoch we update the stats
            self.AE_tst_hist.append( running_loss[0] / len(self.test_loader) )
            if self.has_KLD: self.KLD_tst_hist.append( running_loss[1] / len(self.test_loader) )
            if self.has_LSD: self.LSD_tst_hist.append( running_loss[2] / len(self.test_loader) )
            if self.has_IOD: self.IOD_tst_hist.append( running_loss[3] / len(self.test_loader) )

            ## We also update the information used for visualisation based on the last batch
            self.latent_values  = z_means.cpu() if self.has_KLD else z_values.cpu()
            self.latent_classes = classes

    def update_cyclical(self):
        if self.cyclical is None:
            return 0

        ratio = self.epochs_trained / self.cyclical[1]
        self.KLD_weight = self.cyclical[0] * np.clip( 2*(ratio%1), 0, 1 )

    def visualise_recreation(self):
        """ This function returns reconstructions of 4 selected examples and is called
            during training after each epoch. Good for images.
        """
        with T.no_grad():
            data = self.prepare_data( self.vis_data )
            cond_info = self.prepare_conditional(self.vis_classes)
            recons, *_ = self.cluster.AE_net(data, cond_info)
            return recons.view(self.vis_data.shape).cpu()

    def run_training_loop( self, load_flag, vis_z, dim_red ):
        """ This is the main training loop
        """
        plt.ion()

        ## For loading previous states of the network
        if load_flag is not None:
            self.load_models( load_flag )

        ## Creating the plots for the visualisation
        rp = myPL.recreation_plot( self.vis_data, self.name, self.unorm_trans )
        if vis_z>0:
            zp = myPL.latent_plot( self.name, dim_red )

        ## Creating the loss/accuracy plots
        AE_loss_plot = myPL.loss_plot( self.cluster.AE_net.name )
        if self.has_KLD: KLD_loss_plot = myPL.loss_plot( self.cluster.AE_net.name+"_KLD" )
        if self.has_LSD: LSD_loss_plot = myPL.loss_plot( self.cluster.LSD_net.name )
        if self.has_IOD: IOD_loss_plot = myPL.loss_plot( self.cluster.IOD_net.name )

        ## We run the training loop indefinetly
        for epoch in count(1):
            print( "\nEpoch: {}".format(epoch) )

            ## We run some checks on the network configuration
            self.set_discrinimator_flags()
            self.update_cyclical()

            ## Run the test/train cycle
            self.testing_epoch()
            self.training_epoch()

            ## Update the visualisation graphs
            rp.update( self.visualise_recreation() )
            if vis_z>0 and epoch%vis_z==0:
                zp.update( self.latent_classes, self.latent_values )

            ## Printing the KLD weight
            print( "Annealing Schedule:" )
            print( " - KLD Weight : ", self.KLD_weight )

            ## Printing the GRL influence
            print( "Avesary Impact:" )
            print( " - GRL Lambda: ", self.cluster.GRLambda )

            ## Print out the loss scores
            print( "Loss Contributions: ")
            print(" - Rec: ", self.AE_trn_hist[-1] )
            if self.has_KLD: print(" - KLD: ", self.KLD_weight * self.KLD_trn_hist[-1] )
            if self.has_LSD: print(" - LSD: ", self.LSD_weight * self.LSD_trn_hist[-1] )
            if self.has_IOD: print(" - IOD: ", self.IOD_weight * self.IOD_trn_hist[-1] )

            ## Update the loss/accuracy plots
            AE_loss_plot.update( self.AE_trn_hist, self.AE_tst_hist )
            if self.has_KLD: KLD_loss_plot.update( self.KLD_trn_hist, self.KLD_tst_hist )
            if self.has_LSD: LSD_loss_plot.update( self.LSD_trn_hist, self.LSD_tst_hist )
            if self.has_IOD: IOD_loss_plot.update( self.IOD_trn_hist, self.IOD_tst_hist )

            ## We save the latest version of the networks
            self.save_models("latest")
