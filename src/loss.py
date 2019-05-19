import torch
from torch.nn.functional import l1_loss, binary_cross_entropy_with_logits, binary_cross_entropy


def loss_generator(image, text, text_length, D, G, lambda_1, params):
    inds = torch.roll(torch.arange(text_length.size(0)), 1)
    negative_text = text[inds]
    negative_text_length = text_length[inds]
    fake, mu, log_sigma = G(image, negative_text, negative_text_length)
    uncond, cond, _ = D(fake, negative_text, negative_text_length)
    l1 = binary_cross_entropy_with_logits(uncond.detach(), torch.ones_like(uncond))
    params["uncond_gen"] += l1
    l2 = binary_cross_entropy(cond.detach(), torch.ones_like(cond))
    params["cond_p_gen"] += l2

    log_Sigma = 2 * log_sigma
    kl = torch.mean(-log_sigma + (torch.exp(log_Sigma) + mu**2 - 1.0) * 0.5)
    params["kl"] += kl * 0.5

    return 0.5 * kl + lambda_1 * l1 + l2, fake, negative_text


def loss_generator_reconstruction(image, text, text_length, D, G, lambda_2, params):
    reconstruction, mu, log_sigma = G(image, text, text_length)

    l1 = l1_loss(reconstruction, image)
    params["l1_reconstruction"] += l1

    log_Sigma = 2 * log_sigma
    kl = torch.mean(-log_sigma + (torch.exp(log_Sigma) + mu**2 - 1.0) * 0.5)
    params["kl"] += kl * 0.5

    return 0.5 * kl + l1 * lambda_2, kl


def loss_real_discriminator(image, text, text_length, D, G, lambda_1, params):
    uncond, cond, cond_n = D(image, text, text_length)
    l1 = binary_cross_entropy_with_logits(uncond, torch.ones_like(uncond))
    l2 = binary_cross_entropy_with_logits(cond, torch.ones_like(cond))
    l3 = binary_cross_entropy_with_logits(cond_n, torch.zeros_like(cond_n))
    params["uncond_disc_real"] += l1
    t = (l2 + l3) / 2.0
    params["cond_disc_real"] += t

    return l1 + lambda_1 * t


def loss_synthetic_discriminator(image, text, text_length, D, G, params):
    inds = torch.roll(torch.arange(text_length.size(0)), 1)
    negative_text = text[inds]
    negative_text_length = text_length[inds]

    fake, _, _ = G(image, negative_text, negative_text_length)
    uncond, _, _ = D(fake.detach(), negative_text, negative_text_length)

    t = binary_cross_entropy_with_logits(uncond, torch.zeros_like(uncond))
    params["cond_disc_fake"] += t
    return t
