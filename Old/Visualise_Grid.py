import sys
home_env = '../'
sys.path.append(home_env)

from Resources import Model
from Resources import Utils as myUT
from Resources import Plotting as myPL

import numpy as np
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torchvision as TV
import torch.nn.functional as F

def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main():

    ## Initialise the model
    model = Model.BIBAE_Agent( name = "BiB-AE", save_dir = "Saved_Models" )

    ## Load up the dataset
    model.initiate_dataset( dataset_name = "MNIST",
                            data_dims = [1,32,32], flatten_data = False, clamped = False,
                            class_dims = 10, class_onehot = True,
                            n_workers = 12, batch_size = 200 )

    ## Initialise the generative VAE
    model.initiate_AE( variational = True, KLD_weight = 5e-2, cyclical = None,
                       latent_dims = 2, use_cond = False,
                       act = nn.LeakyReLU(0.2),
                       mlp_layers = [64],
                       cnn_layers = [ [8,3,1,2], ## Chan,Kern,Pad,Pool
                                      [16,3,1,2],
                                      [32,3,1,2],
                                      [64,3,1,2],
                                      [128,3,1,2] ],
                       drpt = 0.0, lnrm = False, pnrm = True )

    ## Initialise the Latent Space Discriminator
    model.initiate_LSD( GRLambda = 1, weight = 1, flag = "disk", loss_type = "BCE",
                        act = nn.LeakyReLU(0.2),
                        mlp_layers = [128,128],
                        drpt = 0.2, lnrm = False )

    # Initialise the IO Discriminator
    model.initiate_IOD( GRLambda = 1, weight = 1e-4, use_cond = True, loss_type = "EMD",
                        act = nn.LeakyReLU(0.2),
                        mlp_layers = [32],
                        cnn_layers = [ [8,3,1,2],  ## Chan,Kern,Pad,Pool
                                       [16,3,1,2],
                                       [32,3,1,2],
                                       [32,3,1,2],
                                       [32,3,1,2] ],
                        drpt = 0.3, lnrm = False, pnrm = False )

    ## Initialise the model
    model.load_models( "latest" )
    model.bibae_net.eval()

    # load a network that was trained with a 2d latent space
    if model.latent_dims != 2:
        print('Please change the parameters to two latent dimensions.')

    cond_info = 8 * T.ones( 400, dtype=T.int64, device=model.bibae_net.device)

    with T.no_grad():

        ## Giving the conditional information
        if model.class_dims > 0:
            cond_info = F.one_hot(cond_info, num_classes = model.class_dims)
        else:
            cond_info = None

        # create a sample grid in 2d latent space
        latent_x = np.linspace(-1.0,1.0,20)
        latent_y = np.linspace(-1.0,1.0,20)
        latents = T.zeros((len(latent_y), len(latent_x), 2))
        for i, lx in enumerate(latent_x):
            for j, ly in enumerate(latent_y):
                latents[j, i, 0] = lx
                latents[j, i, 1] = ly
        latents = latents.view(-1, 2) # flatten grid into a batch

        # reconstruct images from the latent vectors
        latents = latents.to(model.bibae_net.device)
        image_recon = model.bibae_net.AE_net.decode(latents, cond_info)
        images = T.zeros(image_recon.shape)

        for i, img in enumerate(image_recon):
            images[i] = myPL.trans(img, unorm_trans=model.unorm_trans, make_image=False )

        fig, ax = plt.subplots(figsize=(10, 10))
        show_image(TV.utils.make_grid(images.data,20,5))
        plt.show()

if __name__ == '__main__':
    main()
