import torch
from torch.nn.functional import l1_loss, binary_cross_entropy_with_logits


def loss_generator(image, text, text_length, D, G, lambda_1, lambda_2):
    inds = torch.roll(torch.arange(text_length.size(0)), 1)
    negative_text = text[inds]
    negative_text_length = text_length.squeeze()[inds]
    fake, mu, log_sigma = G(image, negative_text, negative_text_length.squeeze())
    uncond, cond, _ = D(fake, negative_text, negative_text_length)
    l1 = binary_cross_entropy_with_logits(uncond.detach(), torch.ones_like(uncond))
    l2 = binary_cross_entropy_with_logits(cond.detach(), torch.ones_like(cond))

    log_Sigma = 2 * log_sigma
    kl = torch.mean(-log_sigma + (torch.exp(log_Sigma) + mu**2 - 1.0) * 0.5)

    return 0.5 * kl + l1 + l2 * lambda_1, fake, negative_text


def loss_generator_reconstruction(image, text, text_length, D, G, lambda_1, lambda_2):
    reconstruction, mu, log_sigma = G(image, text, text_length.squeeze())

    l1 = l1_loss(reconstruction, image)

    log_Sigma = 2 * log_sigma
    kl = torch.mean(-log_sigma + (torch.exp(log_Sigma) + mu**2 - 1.0) * 0.5)

    return 0.5 * kl + l1 * lambda_2, kl


def loss_real_discriminator(image, text, text_length, D, G, lambda_1):
    uncond, cond, cond_n = D(image, text, text_length.squeeze())
    l1 = binary_cross_entropy_with_logits(uncond, torch.ones_like(uncond))
    l2 = binary_cross_entropy_with_logits(cond, torch.ones_like(cond))
    l3 = binary_cross_entropy_with_logits(cond_n, torch.zeros_like(cond_n))
    return l1 + lambda_1 * (l2 + l3) * 0.5


def loss_synthetic_discriminator(image, text, text_length, D, G, lambda_1):
    inds = torch.roll(torch.arange(text_length.size(0)), 1)
    negative_text = text[inds]
    negative_text_length = text_length.squeeze()[inds]

    fake, _, _ = G(image, negative_text, negative_text_length.squeeze())
    uncond, _, _ = D(fake.detach(), negative_text, negative_text_length.squeeze())

    return binary_cross_entropy_with_logits(uncond, torch.zeros_like(uncond))
