import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Utils as myUT
from Resources import Model

import torch.nn as nn

def main():

    ## Initialise the model
    model = Model.BIB_AE( name = "VAE_GAN", save_dir = "Saved_Models" )

    ## Load up the dataset
    model.initiate_dataset( dataset_name = "MNIST",
                            data_dims = [1,28,28], flatten_data = False, clamped = True,
                            target_dims = 0, targ_onehot = True,
                            n_workers = 12, batch_size = 1024 )

    ## Initialise the generative VAE
    model.initiate_VAE( do_mse = False, latent_dims = 2, KLD_weight = 0,
                        lr = 1e-4, act = nn.ELU(),
                        mlp_layers = [256, 128], drpt = 0.0, lnrm = False,
                        cnn_layers = [ [32,4,1,0], ## C,K,S,P
                                       [32,5,2,0],
                                       [32,5,2,0] ],
                        bnrm = False )

    ## Initialise the IO Discriminator
    model.initiate_IO_Disc( weight = 5e-2, lr = 1e-5, act = nn.ELU(),
                            mlp_layers = [32], drpt = 0.0, lnrm = False,
                            cnn_layers = [ [16,4,2,0],
                                           [16,5,2,0] ],
                            bnrm = False )

    # ## Initialise the Latent (Z) Discriminator
    model.initiate_LZ_Disc( weight = 1e-1, lr = 1e-5, act = nn.ELU(),
                            mlp_layers = [64,64], drpt = 0.0, lnrm = False )

    ## Setup up the parameters for training the networks
    model.setup_training( burn_in = 1, disc_range = [0.80,0.95] )

    ## Run the training loop
    model.run_training_loop( load_flag = "latest", vis_z = 1, dim_red = "None" )

if __name__ == '__main__':
    main()
