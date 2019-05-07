import torch
import torch.distributions as dist
from torch.nn.functional import l1_loss


def loss_generator(image, text, negative_text, text_length, D, G, lambda_1, lambda_2):
    t1 = torch.log(D(image))
    gen, _, _ = G(image, negative_text, text_length)
    t2 = torch.log(D(gen, negative_text, text_length, negative=True))
    gen, mu, sigma = G(image, text, text_length)
    l_rec = l1_loss(image, gen)

    Sigma = sigma**2
    kl = (-torch.sum(torch.log(Sigma), dim=1) - sigma.size(1) + torch.sum(Sigma, dim=1) + torch.sum(mu**2, dim=1)) * 0.5

    return torch.mean(t1) + lambda_1 * torch.mean(t2) + lambda_2 * l_rec + torch.sum(kl)


def loss_discriminator(image, text, negative_text, text_length, D, G, lambda_1):
    tt1 = torch.log(D(image))
    tt2 = torch.log(D(image, text, text_length, negative=False))
    tt3 = torch.log(1.0 - D(image, negative_text, text_length, negative=True))
    t1 = torch.mean(tt1 + lambda_1 * (tt2 + tt3))
    t2 = torch.mean(torch.log(1.0 - D(G(image, negative_text, text_length)[0])))
    loss = t1 + t2
    return loss