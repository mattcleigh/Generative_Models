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
    model.initiate_dataset( dataset_name = "Friend_Faces",
                            data_dims = [3,90,90], flatten_data = False, clamped = False,
                            class_dims = 10, class_onehot = True,
                            n_workers = 6, batch_size = 64 )

    ## Initialise the generative VAE
    model.initiate_AE( variational = True, KLD_weight = 1e-6, cyclical = None,
                       latent_dims = 64,   use_cond = False,
                       act = nn.LeakyReLU(0.2),
                       mlp_layers = [512],
                       cnn_layers = [ [32,4,1,0], ## C,K,S,P
                                      [32,3,2,0],
                                      [64,3,2,0],
                                      [128,3,2,0],
                                      [256,4,2,0],
                                      [512,4,2,0], ],
                       drpt = 0.0, lnrm = True, bnrm = True )

    ## Initialise the Latent Space Discriminator
    # model.initiate_LSD( GRLambda = 1, weight = 1,
    #                     act = nn.LeakyReLU(),
    #                     mlp_layers = [64,64,64],
    #                     drpt = 0.2, lnrm = False )

    ## Initialise the IO Discriminator
    model.initiate_IOD( GRLambda = 1, weight = 1, use_cond = False,
                        act = nn.LeakyReLU(0.2),
                        mlp_layers = [256],
                        cnn_layers = [ [32,4,2,0], ## C,K,S,P
                                       [32,4,2,0],
                                       [32,3,2,0],
                                       [32,4,2,0], ],
                        drpt = 0.5, lnrm = False, bnrm = True )

    ## Setup up the parameters for training the networks
    model.setup_training( lr = 5e-4, burn_in = 10, change_lr = 3e-4, clip_grad = 0 )

    ## Run the training loop
    model.run_training_loop( load_flag = "latest", vis_z = 1, dim_red = "PCA" )

if __name__ == '__main__':
    main()
