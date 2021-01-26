import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Utils as myUT
from Resources import Model

import torch.nn as nn

def main():

    ## Inita
    ## Initialise the Latent Space Discriminator
    # model.initiate_LSD( on_at = 0, kill_KLD = False, GRLambda = 1, weight = 1,
    #                     shape = "gauss", loss_type = "BCE",
    #                     act = nn.LeakyReLU(0.2),
    #                     mlp_layers = [512,512,256],
    #                     drpt = 0.2, lnrm = False )

    # Initialise the IO Discriminator
    model.initiate_IOD( on_at = 0, GRLambda = 1,
                        weight = 1, use_cond = False,
                        loss_type = "NST",
                        act = nn.LeakyReLU(0.2),
                        mlp_layers = [128],
                        cnn_layers = [ [16,1,0,1,0,1], ## Chan,Kern,Pad,NConv,Pool,Res
                                       [32,3,1,2,2,1],
                                       [64,3,1,2,2,1],
                                       [128,3,1,2,2,1],
                                       [256,3,1,2,2,1],
                                       [256,3,1,2,2,1],
                                       [256,3,1,2,2,1] ],
                        drpt = 0.0, lnrm = False, nrm = False )

    ## Setup up the parameters for training the networks
    model.setup_training( load_flag = "latest", lr = 5e-4, change_lr = 1e-4, clip_grad = 0 )

    ## Run the training loop
    model.run_training_loop( vis_z = 1, dim_red = "PCA", show_every = 10, sv_evry = 10 )

if __name__ == '__main__':
    main()
