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
        self.bibae_net = myNN.BIBAE_Network( self.name+"_BIBAE" )

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
        self.bibae_net.save_checkpoint(flag)

    def load_models(self, flag=""):
        self.bibae_net.load_checkpoint(flag)

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
        self.bibae_net.rename(self.name)

        ## Getting the training and testing dataloaders as well as the post processing transformation
        self.train_loader, self.test_loader, self.unorm_trans = myDS.load_dataset( dataset_name, batch_size, n_workers )

        ## A list of examples to save for use in visualisation of the reconstruction and the latent space
        idxes = np.random.randint(0,len(self.test_loader.dataset),6)
        self.vis_data = T.stack( [self.test_loader.dataset[i][0] for i in idxes ])

        ## Class values are needed for conditional information
        if isinstance(self.test_loader.dataset[0][1], int):
            self.vis_classes = T.tensor([self.test_loader.dataset[i][1] for i in idxes ])
        else:
            self.vis_classes = T.stack([self.test_loader.dataset[i][1] for i in idxes ])

        ## Checking if we will need a CNN structure for all later networks or if an MLP will be fine
        self.do_cnn = not( len(data_dims)==flatten_data==1 )

    def initiate_AE( self, variational, KLD_weight, cyclical,
                     latent_dims, use_cond,
                     act, mlp_layers, cnn_layers,
                     drpt, lnrm, pnrm ):
        """ This initialises the autoencoder
        """

        ## The latent space dimension is needed throughout the training steps
        self.has_KLD = variational
        self.KLD_weight = KLD_weight
        self.cyclical    = cyclical
        self.latent_dims = latent_dims
        self.AE_use_cond = use_cond
        c_dims = self.class_dims if use_cond else 0

        ## Giving the cluster an autoencoding network
        self.bibae_net.setup_AE( self.name+"/AE", variational, self.do_cnn,
                               self.data_dims, self.latent_dims, c_dims, self.clamped,
                               act, mlp_layers, cnn_layers,
                               drpt, lnrm, pnrm )

    def initiate_LSD( self, on_at, kill_KLD, GRLambda, weight, flag, loss_type,
                      act, mlp_layers, drpt, lnrm ):
        """ Give the cluster an adversarial network for the latent space.
        """

        ## The discriminator contribution to the total loss function
        self.has_LSD    = True
        self.train_LSD  = False
        self.LSD_on_at  = on_at
        self.kill_KLD   = kill_KLD
        self.LSD_weight = weight

        ## The Discriminator network
        self.bibae_net.setup_LSD( GRLambda, loss_type, flag, self.name+"/LSD", False,
                                [self.latent_dims], 0,
                                act, mlp_layers, [], drpt, lnrm, False )

    def initiate_IOD( self, on_at, GRLambda, weight, loss_type, use_cond,
                      act, mlp_layers, cnn_layers,
                      drpt, lnrm, pnrm ):
        """ Give the cluster both an adversarial discriminator network for the input and output.
        """

        ## The discriminator contribution to the total loss function
        self.has_IOD      = True
        self.train_IOD    = False
        self.IOD_on_at    = on_at
        self.IOD_weight   = weight
        self.IOD_use_cond = use_cond
        c_dims = self.class_dims if use_cond else 0

        ## Giving the cluster the IO discriminator network
        self.bibae_net.setup_IOD( GRLambda, loss_type, self.name+"/IOD",
                                  self.do_cnn, self.data_dims, c_dims,
                                  act, mlp_layers, cnn_layers,
                                  drpt, lnrm, pnrm )

    def setup_training(self, load_flag, lr, change_lr, clip_grad ):
        """ A function which sets up some variables used in training, including
            the targeted range of discrinimator accuracies.
            This function is required only for trianing.
        """
        self.optimiser = optim.Adam( self.bibae_net.parameters(), lr=lr, betas=(0.9,0.99) )
        self.change_lr = change_lr
        self.clip_grad = clip_grad
        self.epochs_trained = 0

        self.trn_hist = deque(maxlen=100)
        self.tst_hist = deque(maxlen=100)

        ## For loading previous states of the network
        if load_flag is not None:
            self.load_models( load_flag )

    def prepare_conditional(self, classes):
        """ This is used to modify the classes into conditional information,
            particularly for one-hot encoding categorial varaibles (MNIST, CIFAR)
            This function is called even if the no conditional information is used.
        """

        ## We check if conditional information is used by any of our networks
        if self.AE_use_cond==self.IOD_use_cond==False:
            return None

        cond_info = classes.to(self.bibae_net.device)
        if self.class_onehot:
            cond_info = F.one_hot(cond_info, num_classes = self.class_dims)
        return cond_info

    def prepare_data(self, data):
        """ This is used to modify the input data for the network,
            particularly for flattening images when using an MLP
        """
        data = data.to(self.bibae_net.device)
        if self.flatten_data:
            data = data.view( data.size(0), -1 )
        return data

    def set_discrinimator_flags(self):
        change = False
        if self.has_LSD:
            if self.epochs_trained == self.LSD_on_at:
                self.train_LSD = True
                change = True
            if self.kill_KLD:
                if self.epochs_trained == self.LSD_on_at+2:
                    self.KLD_weight = 0

        if self.has_IOD:
            if self.epochs_trained == self.IOD_on_at:
                self.train_IOD = True
                change = True

        if change:
            print("Reducing ADAM beta to 0 and chaning to new learning rate")
            self.optimiser.param_groups[0]["betas"] = (0,0.99)
            self.optimiser.param_groups[0]["lr"] = self.change_lr


    def training_epoch(self, show_every = 0):
        """ This function performs one epoch of training on data provided by the train_loader
        """
        ## Put all networks into training mode
        self.bibae_net.train()

        ## We collect running losses for each model
        running_loss = np.zeros(4)

        ## Now we cycle through each minibatch
        for i, (data, classes) in enumerate(tqdm(self.train_loader, desc="Training", ncols=80, unit="")):

            ## We zero out the gradients
            self.optimiser.zero_grad()

            ## We prepare the input and conditional data (flatten, one-hot, move to device, etc)
            data = self.prepare_data(data)
            cond_info = self.prepare_conditional(classes)

            ## We forward propagate the data through the BibAE network and calculate losses
            recons, z_values, AE_loss, KLD_loss, LSD_loss, IOD_loss = self.bibae_net(data, cond_info, self.train_LSD, self.train_IOD)

            ## The loss funcitons are combined for the whole network, and grad desc is performed
            Total_loss = AE_loss + self.KLD_weight*KLD_loss + self.LSD_weight*LSD_loss + self.IOD_weight*IOD_loss
            Total_loss.backward()

            ## Clipping on the absolute value of the gradients
            if self.clip_grad > 0:
                nn.utils.clip_grad_value_(self.bibae_net.parameters(), self.clip_grad)
            self.optimiser.step()

            ## We update the running losses for plotting
            running_loss += np.array([ AE_loss.item(), KLD_loss.item(), LSD_loss.item(), IOD_loss.item() ])

            ## Displaying intermittent results
            if show_every > 0:
                if i % show_every == 0:
                    self.rec_plot.update( self.visualise_recreation() )

        ## At the end of the epoch we update the stats
        running_loss = (running_loss / len(self.train_loader) ).tolist()
        self.trn_hist.append( running_loss )

        ## The epoch counter is incremented
        self.epochs_trained += 1

    def testing_epoch(self):
        """ This function performs one epoch of testing on data provided by the test_loader.
            It is basically the same as the train functin above but without the gradient desc
        """
        # with T.no_grad():
        self.bibae_net.eval()

        running_loss = np.zeros(4)
        for (data, classes) in tqdm(self.test_loader, desc="Testing ", ncols=80, unit=""):
            data = self.prepare_data(data)
            cond_info = self.prepare_conditional(classes)
            recons, z_values, AE_loss, KLD_loss, LSD_loss, IOD_loss = self.bibae_net(data, cond_info, self.train_LSD, self.train_IOD)
            running_loss += np.array([ AE_loss.item(), KLD_loss.item(), LSD_loss.item(), IOD_loss.item() ])
        running_loss = (running_loss / len(self.test_loader) ).tolist()
        self.tst_hist.append( running_loss )
        self.latent_values = z_values.detach().cpu()
        self.latent_classes = classes

    def update_cyclical(self):
        if self.cyclical is None:
            return 0
        ratio = self.epochs_trained / self.cyclical[1]
        self.KLD_weight = self.cyclical[0] * np.clip( 2*(ratio%1), 0, 1 )

    def visualise_recreation(self):
        """ This function returns reconstructions of 6 selected examples and is called
            during training after each epoch. Good for images.
        """
        with T.no_grad():
            self.bibae_net.eval()
            data = self.prepare_data( self.vis_data )
            cond_info = self.prepare_conditional(self.vis_classes)
            recons, *_ = self.bibae_net.AE_net(data, cond_info)
            self.bibae_net.train()
            return recons.view(self.vis_data.shape).cpu()

    def run_training_loop( self, vis_z, dim_red, show_every = 0, sv_evry = 20 ):
        """ This is the main training loop
        """
        plt.ion()

        ## Creating the plots for the visualisation
        self.rec_plot = myPL.recreation_plot( self.vis_data, self.name, self.unorm_trans )
        if vis_z>0: lat_plot = myPL.latent_plot( self.name, dim_red )

        ## Creating the loss/accuracy plots
        AE_loss_plot = myPL.loss_plot( self.bibae_net.AE_net.name )
        if self.has_KLD: KLD_loss_plot = myPL.loss_plot( self.bibae_net.AE_net.name+"_KLD" )
        if self.has_LSD: LSD_loss_plot = myPL.loss_plot( self.bibae_net.LSD_net.name )
        if self.has_IOD: IOD_loss_plot = myPL.loss_plot( self.bibae_net.IOD_net.name )

        ## We run the training loop indefinetly
        for epoch in count(1):
            print( "\nEpoch: {}".format(epoch) )

            ## We run some checks on the network configuration
            self.set_discrinimator_flags()
            self.update_cyclical()

            ## Run the test/train cycle
            self.testing_epoch()
            self.training_epoch(show_every)

            ## Update the visualisation graphs
            self.rec_plot.update( self.visualise_recreation() )
            if vis_z>0 and epoch%vis_z==0:
                lat_plot.update( self.latent_classes, self.latent_values )

            ## Update the loss/accuracy plots
            trn_arr = np.array(self.trn_hist)
            tst_arr = np.array(self.tst_hist)
            AE_loss_plot.update( trn_arr[:,0], tst_arr[:,0] )
            if self.has_KLD: KLD_loss_plot.update( trn_arr[:,1], tst_arr[:,1] )
            if self.has_LSD: LSD_loss_plot.update( trn_arr[:,2], tst_arr[:,2] )
            if self.has_IOD: IOD_loss_plot.update( trn_arr[:,3], tst_arr[:,3] )

            ## Printing the KLD weight
            print( "Annealing Schedule:" )
            print( " - KLD Weight : ", self.KLD_weight )

            ## Print out the loss scores
            print( "Loss Contributions: ")
            print(" - Rec: ", self.trn_hist[-1][0] )
            if self.has_KLD: print(" - KLD: ", self.KLD_weight * self.tst_hist[-1][1] )
            if self.has_LSD: print(" - LSD: ", self.LSD_weight * self.tst_hist[-1][2] )
            if self.has_IOD: print(" - IOD: ", self.IOD_weight * self.tst_hist[-1][3] )

            ## We save the latest version of the networks
            self.save_models("latest")
            if epoch%sv_evry==0:
                self.save_models(str(self.epochs_trained))
