import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Utils as myUT
from Resources import Model

import torch.nn as nn

def main():

    ## Initialise the model
    model = Model.BIBAE_Agent( name = "VAE_GAN", save_dir = "Saved_Models" )

    ## Load up the dataset
    model.initiate_dataset( dataset_name = "CelebA",
                            data_dims = [3,64,64], flatten_data = False, clamped = False,
                            class_dims = 40, class_onehot = False,
                            n_workers = 12, batch_size = 128 )

    ## Initialise the generative VAE
    model.initiate_AE( variational = True, KLD_weight = 5e-2, cyclical = None,
                       latent_dims = 256, use_cond = False,
                       act = nn.LeakyReLU(0.2),
                       mlp_layers = [512],
                       cnn_layers = [ [16,3,1,0,0], ## Chan,Kern,Pad,Residual,Pool
                                      [32,3,1,0,2],
                                      [64,3,1,0,2],
                                      [128,3,1,0,2],
                                      [256,3,1,0,2],
                                      [256,3,1,2,2],
                                      [256,3,1,2,2] ],
                       drpt = 0.0, lnrm = False, pnrm = True )

    ## Initialise the Latent Space Discriminator
    # model.initiate_LSD( on_at = 0, kill_KLD = False, GRLambda = 1, weight = 1,
    #                     flag = "gauss", loss_type = "BCE",
    #                     act = nn.LeakyReLU(0.2),
    #                     mlp_layers = [512,512,256],
    #                     drpt = 0.2, lnrm = False )

    # Initialise the IO Discriminator
    model.initiate_IOD( on_at = 0, GRLambda = 5e-1,
                        weight = 1, use_cond = True,
                        loss_type = "EMD",
                        act = nn.LeakyReLU(0.2),
                        mlp_layers = [64,64,64],
                        cnn_layers = [ [16,3,1,0,0],  ## Chan,Kern,Pad,Residual,Pool
                                       [32,3,1,0,2],
                                       [64,3,1,0,2],
                                       [128,3,1,0,2],
                                       [256,3,1,0,2],
                                       [256,3,1,1,2],
                                       [256,3,1,1,2] ],
                        drpt = 0.2, lnrm = False, pnrm = False )

    ## Setup up the parameters for training the networks
    model.setup_training( load_flag = "latest", lr = 1e-3, change_lr = 1e-4, clip_grad = 0 )

    ## Run the training loop
    model.run_training_loop( vis_z = 1, dim_red = "PCA", show_every = 20, sv_evry = 50 )

if __name__ == '__main__':
    main()
