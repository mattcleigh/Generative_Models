import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Utils as myUT

import torch.nn as nn
from VAE import Agent

def main():

    ################ USER INPUT ################

    alg_name = "VAE"
    dataset_name = "CelebA"

    load_flag = None

    patience = 50

    vis_z_space = 50
    dim_reduction = "TSNE" ## None, PCA or TSNE, the latter is much slower but can give better results

    agent = Agent(
                     name    = "CNN" + "_{}_{}".format(alg_name, dataset_name),
                     net_dir = "Saved_Models",
                     \
                     dataset_name = dataset_name,
                     data_dims = [3,32,32], flatten_data = False,
                     latent_dims = 16,
                     condit_dims = 0, targ_onehot = True,
                     clamp_out = True,
                     \
                     layer_sizes = [128], active = nn.ELU(),
                     lr = 1e-4,
                     \
                     channels = [64,64,128,128], kernels = [4,4,4,4],
                     strides = [2,2,2,2], padding = 1,
                     \
                     reg_weight = 0.1,
                     \
                     batch_size = 1024, n_workers = 6,
                 )

    ############################################

    myUT.Fit( agent, load_flag, patience, vis_z_space, dim_reduction )


if __name__ == '__main__':
    main()
