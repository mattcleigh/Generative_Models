import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Utils as myUT

import torch.nn as nn
from VAE import Agent

def main():

    ################ USER INPUT ################

    alg_name = "VAE"
    dataset_name = "MNIST"

    load_flag = None

    patience = 50

    vis_z_space = 1
    dim_reduction = "None" ## None, PCA or TSNE, the latter is much slower but can give better results

    agent = Agent(
                     name    = "MLP" + "_{}_{}".format(alg_name, dataset_name),
                     net_dir = "Saved_Models",
                     \
                     dataset_name = dataset_name,
                     data_dims = [784], flatten_data = True,
                     latent_dims = 2,
                     condit_dims = 0, targ_onehot = True,
                     clamp_out = True,
                     \
                     layer_sizes = [256,64,32], active = nn.ELU(),
                     grad_clip = 40,
                     lr = 1e-3,
                     \
                     reg_weight = 1,
                     \
                     batch_size = 1024, n_workers = 12,
                 )

    ############################################

    myUT.Fit( agent, load_flag, patience, vis_z_space, dim_reduction )


if __name__ == '__main__':
    main()
