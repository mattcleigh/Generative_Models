import sys
home_env = '../'
sys.path.append(home_env)

import matplotlib.pyplot as plt
from Resources import Plotting as myPL
from itertools import count

def Fit( agent, load_flag, patience, vis_z_space, dim_reduction ):
    """ A function which performs the fitting of most of the VAE/Generative type models
    """

    ## For loading previous states of the network
    if load_flag is not None:
        agent.load_models( load_flag )

    ## Creating an updating plot of the loss function
    plt.ion()
    lp = myPL.loss_plot( agent.name )
    rp = myPL.recreation_plot( agent.recon_data, agent.name )

    ## Visualising the latent space can be costly at high dimensions, so it is optional
    if vis_z_space > 0:
        zp = myPL.latent_plot( agent.name, dim_reduction )

    ## The number of bad_epochs and the min test loss to be used for early stopping
    bad_epochs = 0
    min_test_loss = 1e6

    for epoch in count(1):
        print( "\nEpoch: {}".format(epoch) )

        ## Run the test/train cycle
        agent.test()
        agent.train()

        ## Print running loss values
        print("Average loss on testing data = {:.4f}".format(agent.tst_loss_hist[-1]))
        print("Average loss on training data = {:.4f}".format(agent.trn_loss_hist[-1]))

        ## Update the animated graphs
        lp.update( agent.tst_loss_hist, agent.trn_loss_hist )
        rp.update( agent.visualise_recreation() )
        if vis_z_space>0 and epoch%vis_z_space==0:
            zp.update( agent.latent_targets, agent.latent_means )

        ## We save the latest version of the networks
        agent.save_models("latest")

        ## Check if the test loss has decreased
        if agent.tst_loss_hist[-1] < min_test_loss:
            min_test_loss = agent.tst_loss_hist[-1]
            agent.save_models("best")
            bad_epochs = 0

        ## If it hasnt, the number of bad_epochs goes up
        else:
            bad_epochs += 1
            print("Epoch was bad: ", bad_epochs)

        ## When the number of bad epochs exceeds the patience then the training is stopped, but the graphs remain
        if bad_epochs > patience:
            print("\nPatience exceeded: stopping training!\n")
            plt.ioff()
            plt.show()
            return 1
