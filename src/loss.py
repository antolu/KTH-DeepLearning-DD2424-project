import torch
from torch.nn.functional import l1_loss


def loss_generator(image, text, text_length, D, G, lambda_1, lambda_2):
    text_length = text_length.squeeze()
    uncond, pos_cond, neg_cond = D(image, text, text_length)
    t1 = uncond
    inds = torch.roll(torch.arange(text.size(0)), 1).squeeze()
    negative_text = text[inds]
    negative_text_length = text_length[inds]
    gen, _, _ = G(image, negative_text, negative_text_length)
    t2 = neg_cond
    gen, mu, log_sigma = G(image, text, text_length)
    l_rec = l1_loss(image, gen)

    log_Sigma = 2*log_sigma
    kl = (-torch.sum(log_Sigma, dim=1) - log_sigma.size(1) + torch.sum(torch.exp(log_Sigma), dim=1) + torch.sum(mu**2, dim=1)) * 0.5

    return torch.mean(t1) + lambda_1 * torch.mean(t2) + lambda_2 * l_rec + torch.sum(kl)


def loss_discriminator(image, text, text_length, D, G, lambda_1):
    uncond, cond, cond_n = D(image, text, text_length.squeeze())
    tt1 = uncond
    tt2 = cond
    tt3 = torch.log(1.0 - torch.exp(cond_n))
    t1 = torch.mean(tt1 + lambda_1 * (tt2 + tt3))
    inds = torch.roll(torch.arange(text_length.size(0)), 1)
    negative_text = text[inds]
    negative_text_length = text_length[inds]
    fake = G(image, negative_text, negative_text_length.squeeze())[0]
    inner = 1.0 - torch.exp(D(fake.detach()))
    inner[inner <= 0] = 1e-6
    t2 = torch.mean(torch.log(inner))
    loss = t1 + t2
    return loss
