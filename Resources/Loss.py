import torch as T
import torch.nn as nn
import torch.nn.functional as F

def KLD_to_Norm_Loss( means, log_stds ):
    return 0.5 * T.mean( means*means + (2*log_stds).exp() - 2*log_stds - 1 )

def GAN_Loss( type, disc_real, disc_fake, inpt_real, inpt_fake, D,
              c_info = None, eps_drift = 0.001, gp_coef = 10, gp_gamma = 100 ):
    if type == "BCE":
        return BCE_GAN_Loss( disc_real, disc_fake )
    if type =="EMD":
        return EMD_GAN_Loss( disc_real, disc_fake, inpt_real, inpt_fake, D,
                             c_info, eps_drift, gp_coef, gp_gamma )

def BCE_GAN_Loss( disc_real, disc_fake ):
    ones  = T.ones_like( disc_real )
    zeros = T.zeros_like( disc_fake )
    real_loss = F.binary_cross_entropy_with_logits(disc_real, ones, reduction="mean")
    fake_loss = F.binary_cross_entropy_with_logits(disc_fake, zeros, reduction="mean")
    total_loss = 0.5 * ( real_loss + fake_loss )
    return total_loss

def EMD_GAN_Loss( disc_real, disc_fake, inpt_real, inpt_fake, D,
                  c_info = None, eps_drift = 0.001, gp_coef = 10, gp_gamma = 100 ):

    ## The Wasserstein loss is very simple
    ws_loss = T.mean( disc_fake - disc_real )

    ## We also calculate the regularisation term to prevent values from getting too big
    ws_reg = eps_drift * T.mean(disc_real*disc_real)

    ## For the gradient penalty we start by generating random alphas
    alpha = T.rand_like(inpt_real)

    ## We use the random values to interpolate the inputs via polyak averaging
    inpt_mid = (alpha * inpt_real + (1-alpha) * inpt_fake).detach()
    inpt_mid.requires_grad_(True)

    ## We need fake conditional information too
    if c_info is not None:
        n_classes = c_info.shape[-1]
        alpha_idx = T.randint( n_classes, size=(len(c_info),) )
        mid_c_info = F.one_hot( alpha_idx, num_classes=n_classes ).type(T.float32).to(inpt_mid.device)
        mid_c_info.requires_grad_(True)
    else:
        mid_c_info = None

    ## We pass the interpolated sample through the network
    disc_mid = D(inpt_mid, mid_c_info)

    ## We now calculate all gradients of the outputs with respect to the inputs
    grad_outputs = T.ones_like(disc_mid)
    gradients = T.autograd.grad( outputs=disc_mid, inputs=inpt_mid,
                                 grad_outputs=grad_outputs,
                                 create_graph=True, retain_graph=True)[0]

    ## Gradients have shape (batch_size, sample_dims(c,h,w) )
    ## We flatten them before we take the norm
    gradients = gradients.view(gradients.shape[0], -1)
    grad_norm = T.sqrt(T.sum(gradients**2, dim=1) + 1e-12)

    ## Now we calculate the gradient penalty
    gp_loss = gp_coef * ( ( grad_norm - gp_gamma )**2 / gp_gamma**2 ).mean()

    ## Finally we add them all together
    total_loss = ws_loss + ws_reg + gp_loss
    return total_loss
