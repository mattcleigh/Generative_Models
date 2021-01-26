import torch as T
import torch.nn as nn
import torch.nn.functional as F

from Resources import Utils as myUT
from Resources import Layers as myLY

class GAN_Loss( nn.Module ):
    """
        Standard Class for calculating the loss for GAN_Loss.

        Since many of the losses require their own calculations of output,
        this method will also return the discriminator outputs, it is advised
        that this is done together to save computation time.

        Three options are available via the type flag string:
            BCE - Calculates the origional GAN loss
            NST - Non-saturating variant of above, due to different functional
                  forms this method alternates between disc and gen optimisation
            EMD - The "Earth Mover Distance", used for WGANS

        Since the losses are calculated together, the min-max game is achived using a gradient
        reveral layer, parameterised by grl. NST is the only non-minmax loss, so we dont use it there

        Gradient penalty will also be added to the loss, either 0 centered as with
        most applications (default), but can be 1 by setting gp1=True as with GP-WGAN.
            - Lazy regularisation applies, so the GP will only be added in present intervals based on the
              reg_every flag. Set this to zero for no GP.
            - By default the GP will be calculated with respect to real inputs, to use random mindpoint between
              reals and gen samples set the gp_mid flag to True

    """
    def __init__( self, D, type = "BCE", grl = 1, reg_every = 16, gp_mid = False, gp1 = False, eps_drift = 0.001, gp_coef = 10, gp_gamma = 100 ):
        super(GAN_Loss, self).__init__()

        ## Checking that the type flag is supported
        assert type in [ "BCE", "NST", "EMD" ], "\n\nGAN Loss type flag not recognised!\n\n"

        self.D = D
        self.type = type
        self.grl = grl
        self.GRL = myLY.GRL(grl)
        self.reg_every = reg_every
        self.gp_mid = gp_mid
        self.gp1 = gp1
        self.eps_drift = eps_drift
        self.gp_coef = gp_coef
        self.gp_gamma = gp_gamma
        self.reg_counter = 0

    def forward( self, real_samples, fake_samples, c_info = None ):
        self.reg_counter += 1

        ## For the gradient penalty
        do_reg = False
        if self.reg_every>0:
            if self.reg_counter%self.reg_every==0:
                do_reg = True
                real_samples.requires_grad_(True)

        if self.type == "BCE":
            fake_samples = self.GRL( fake_samples )
            disc_real = self.D( real_samples, c_info )
            disc_fake = self.D( fake_samples, c_info )
            ones  = T.ones_like( disc_real )
            zeros = T.zeros_like( disc_fake )
            real_loss = F.binary_cross_entropy_with_logits(disc_real, ones, reduction="mean")
            fake_loss = F.binary_cross_entropy_with_logits(disc_fake, zeros, reduction="mean")
            total_loss = 0.5 * ( real_loss + fake_loss )

        if self.type == "NST":
            ## Because this type of loss if alternating, it is doubled
            ## Optimise the discriminator only
            if self.reg_counter%2==0:
                myUT.set_requires_grad( self.D, True )
                disc_real = self.D( real_samples, c_info )
                disc_fake = self.D( fake_samples.detach(), c_info )
                ones  = T.ones_like( disc_real )
                zeros = T.zeros_like( disc_fake )
                real_loss = F.binary_cross_entropy_with_logits(disc_real, ones, reduction="mean")
                fake_loss = F.binary_cross_entropy_with_logits(disc_fake, zeros, reduction="mean")
                total_loss = ( real_loss + fake_loss )

            ## Optimise the generator only
            ## We still multiply by the value of the GRL for consitancy
            else:
                myUT.set_requires_grad( self.D, False )
                disc_fake = self.D( fake_samples, c_info )
                ones = T.ones_like( disc_fake )
                total_loss = self.grl * F.binary_cross_entropy_with_logits(disc_fake, ones, reduction="mean")

        if self.type == "EMD":
            fake_samples = self.GRL( fake_samples )
            disc_real = self.D( real_samples, c_info )
            disc_fake = self.D( fake_samples, c_info )
            ws_loss = T.mean( disc_fake - disc_real )
            ws_drft = self.eps_drift * T.mean(disc_real*disc_real)
            total_loss = ws_loss + ws_drft

        ## Adding the gradient penalty
        if do_reg:

            if self.gp_mid:
                alpha = T.rand_like( real_samples )
                inpt_mid = ( alpha * real_samples + (1-alpha) * fake_samples ).detach()
                inpt_mid.requires_grad_(True)
                disc_real = self.D( inpt_mid, c_info )

            grad_outputs = T.ones_like( disc_real )
            gradients = T.autograd.grad( outputs=disc_real, inputs=real_samples,
                                         grad_outputs=grad_outputs,
                                         create_graph=True, retain_graph=True)[0]

            ## Gradients have shape (batch_size, sample_dims(c,h,w) )
            ## We flatten them and before we take the norm
            gradients = gradients.view( gradients.shape[0], -1 )
            grad_sq = T.sum(gradients**2, dim=1)

            if self.gp1:
                grad_sq = ( T.sqrt(grad_sq + 1e-12) / self.gp_gamma - 1 )**2

            gp_loss = self.gp_coef / 2 * ( grad_sq ).mean()

            ## And add it to the total loss
            total_loss += gp_loss

        return total_loss

def KLD_to_Norm_Loss( means, log_stds ):
    return 0.5 * T.mean( means*means + (2*log_stds).exp() - 2*log_stds - 1 )
