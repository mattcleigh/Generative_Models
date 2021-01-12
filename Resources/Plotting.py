import sys
home_env = '../'
sys.path.append(home_env)

import torch as T
import numpy as np
import torchvision as TV
import matplotlib.pyplot as plt

from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA

class loss_contribution_plot(object):
    def __init__(self, title = ""):

        self.fig = plt.figure( figsize = (5,5) )
        self.ax  = self.fig.add_subplot(111)
        self.fig.suptitle(title)

        self.loss_line, = self.ax.step( [], "-r", where="mid" )

    def update(self, vals):

        self.loss_line.set_data( np.arange(len(vals)), vals )

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class loss_plot(object):
    def __init__(self, title = ""):

        self.fig = plt.figure( figsize = (5,5) )
        self.ax  = self.fig.add_subplot(111)
        self.fig.suptitle(title)

        self.trn_line, = self.ax.plot( [], "-r", label="Train" )
        self.tst_line, = self.ax.plot( [], "-g", label="Test" )

    def update(self, trnl, tstl):

        self.trn_line.set_data( np.arange(len(trnl)), np.clip(trnl,-100,100) )
        self.tst_line.set_data( np.arange(len(tstl)), np.clip(tstl,-100,100) )

        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def trans(tensor, unorm_trans=None, make_image=True):

    if unorm_trans is not None:
        tensor = unorm_trans(tensor)

    tensor = T.clamp(tensor, 0, 1 )

    if make_image:
        topil = TV.transforms.ToPILImage()
        tensor = topil(tensor)

    return tensor


class recreation_plot(object):
    def __init__(self, examples, title = "", unorm_trans=None):

        self.n = len(examples)
        self.unorm_trans = unorm_trans

        self.fig = plt.figure( figsize = (2.5*self.n,5) )
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        self.ex_axes = [ self.fig.add_subplot(2,self.n,i) for i in range(1,self.n+1) ]
        self.ot_axes = [ self.fig.add_subplot(2,self.n,i) for i in range(self.n+1,2*self.n+1) ]
        self.update_imgs = []

        for ax, exmpl in zip(self.ex_axes, examples):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.imshow( trans(exmpl, unorm_trans) )

        for ax, exmpl in zip(self.ot_axes, examples):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self.update_imgs.append( ax.imshow( trans(exmpl, unorm_trans) ) )

    def update(self, ex_outputs):
        for img, ex_out in zip(self.update_imgs, ex_outputs):
            img.set_data(trans(ex_out, self.unorm_trans))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class latent_plot(object):
    def __init__(self, title = "", dim_red="None"):

        self.fig = plt.figure( figsize = (5,5) )
        self.ax  = self.fig.add_subplot(111)
        self.fig.suptitle(title)
        self.scatter = self.ax.scatter( [], [], c = [], vmin=0, vmax=9, cmap="tab10" )
        plt.colorbar(self.scatter)

        ## The dimensionality reduction method needed
        if   dim_red == "TSNE": self.dim_red = TSNE(n_components=2, n_jobs=4)
        elif dim_red == "PCA":  self.dim_red = PCA(n_components=2)
        elif dim_red == "None": self.dim_red = None

    def update(self, targets, values):

        if values.shape[1] > 2:
            if self.dim_red is None:
                print( "Cant visualise latent space without dimensionality reduction method specified!")
                return 0
            values = self.dim_red.fit_transform(values)

        self.scatter.set_offsets( values )
        if len(targets.shape)==1:
            self.scatter.set_array( targets )
        else:
            self.scatter.set_array( np.ones(len(values)) )

        self.ax.ignore_existing_data_limits = True
        self.ax.update_datalim(self.scatter.get_datalim(self.ax.transData))
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
