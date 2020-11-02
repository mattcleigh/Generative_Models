import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Utils as myUT

import torch.nn as nn
from VAE import Agent

def main():

    ################ USER INPUT ################

    alg_name  = "VAE"
    dataset   = "MNIST"
    load_flag = None
    patience  = 50
    vis_z     = 5
    dim_red   = "TSNE" ## None, PCA or TSNE, the latter is much slower but can give better results

    agent = Agent(
                    name    = "{}_{}".format(alg_name, dataset),
                    net_dir = "Saved_Models",
                    \
                    dataset_name = dataset,
                    data_dims    = [1,28,28], flatten_data = False,
                    latent_dims  = 10,
                    condit_dims  = 0, targ_onehot = True,
                    clamp_out    = True,
                    \
                    lr  = 1e-3,
                    act = nn.ELU(),
                    mlp_layers = [256,128], drpt = 0.0, lnrm = False,
                    cnn_layers = [ [16,4,2,1], ## C,K,S,P
                                   [16,4,2,1],
                                   [16,4,1,1] ],
                    \
                    reg_weight = 1,
                    \
                    batch_size = 1024, n_workers = 12,
                 )

    ############################################

    myUT.Fit( agent, load_flag, patience, vis_z, dim_red )

if __name__ == '__main__':
    main()
