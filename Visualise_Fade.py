import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Model
from Resources import Utils as myUT
from Resources import Plotting as myPL

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torchvision as TV
import torch.nn.functional as F

def main():

    steps = 50

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

    ## Initialise the model
    model.load_models( "latest" )
    model.bibae_net.eval()

    plt.ion()
    fig = plt.figure( figsize=(9,3) )
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax_S = fig.add_subplot(1,3,1)
    ax_M = fig.add_subplot(1,3,2)
    ax_D = fig.add_subplot(1,3,3)

    real_D = model.test_loader.dataset[0][0].to(model.bibae_net.device)
    im_S = ax_S.imshow( myPL.trans(real_D, model.unorm_trans) )
    im_M = ax_M.imshow( myPL.trans(real_D, model.unorm_trans) )
    im_D = ax_D.imshow( myPL.trans(real_D, model.unorm_trans) )

    with T.no_grad():
        while True:

            ## The destination image becomes the new source
            real_S = real_D.clone()

            ## We randomly select a new destination image
            idx = rd.randint(len(model.train_loader.dataset))
            real_D = model.train_loader.dataset[idx][0].to(model.bibae_net.device)

            ## We get the encodings of each image
            encode_S = model.bibae_net.AE_net.encode(real_S.unsqueeze(0))[0]
            encode_D = model.bibae_net.AE_net.encode(real_D.unsqueeze(0))[0]

            decode_S = model.bibae_net.AE_net.decode(encode_S)[0]
            decode_D = model.bibae_net.AE_net.decode(encode_D)[0]

            im_S.set_data( myPL.trans(real_S, model.unorm_trans) )
            im_D.set_data( myPL.trans(real_D, model.unorm_trans) )

            ## We calculate the vector pointing between the two encodings
            step_vec = ( encode_D - encode_S ) / steps #/ T.norm( encode_D - encode_S )
            vec = encode_S
            dist = (encode_D - vec).norm().item()
            old_dist = 100
            end = False

            while True:
                vec += step_vec
                old_dist = dist
                dist = (encode_D - vec).norm().item()

                if dist > old_dist:
                    vec = encode_D
                    end = True

                decode_M = model.bibae_net.AE_net.decode(vec)[0]
                im_M.set_data( myPL.trans(decode_M, model.unorm_trans) )
                fig.canvas.draw()
                fig.canvas.flush_events()

                if end: break
            input("Next")

if __name__ == '__main__':
    main()
