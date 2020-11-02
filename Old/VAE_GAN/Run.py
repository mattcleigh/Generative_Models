import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Utils as myUT

import torch.nn as nn
from VAE_GAN import Agent

def main():

    ################ USER INPUT ################

    alg_name  = "VAE_GAN"
    dataset   = "CIFAR"
    load_flag = None
    patience  = 1e6
    vis_z     = 0
    dim_red   = "None" ## None, PCA or TSNE, the latter is much slower but can give better results

    agent = Agent(
                    name    = "{}_{}".format(alg_name, dataset),
                    net_dir = "Saved_Models",
                    \
                    dataset_name = dataset,
                    data_dims    = [3,32,32], flatten_data = False,
                    latent_dims  = 16,
                    condit_dims  = 10, targ_onehot = True,
                    clamp_out    = True,
                    \
                    \
                    burn_in = 50, swap_time = 10,
                    \
                    ## The VAE network structure
                    V_lr  = 1e-4,
                    V_act = nn.ELU(),
                    V_mlp_layers = [512,256,128], V_drpt = 0.0, V_lnrm = True,
                    V_cnn_layers = [ [16,4,2,1], ## C,K,S,P
                                     [16,4,2,1],
                                     [16,4,1,1],
                                     [16,4,1,1] ],
                    \
                    ## The Discriminator network structure
                    D_lr  = 5e-6,
                    D_act = nn.ELU(),
                    D_mlp_layers = [10], D_drpt = 0.0, D_lnrm = True,
                    D_cnn_layers = [ [16,4,2,1], ## C,K,S,P
                                     [16,4,2,1] ],
                    \
                    reg_weight = 1, gan_weight = 5,
                    \
                    batch_size = 1024, n_workers = 12,
                 )

    ############################################

    myUT.Fit( agent, load_flag, patience, vis_z, dim_red )


if __name__ == '__main__':
    main()
