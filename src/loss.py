import torch
from torch.nn.functional import l1_loss as l1, binary_cross_entropy_with_logits as bce


class Loss:
    """Losses used for training TAGAN"""
    def __init__(self, image, text, text_length, generator, discriminator, params, lambda_1=10.0, lambda_2=0.2):
        """
        Keyword Arguments:
        image         -- Batch of images tensor
        text          -- Batch of text captions tensor
        text_length   -- Lengths of the captions for every element in the batch
        generator     -- Generator model
        discriminator -- Discriminator model
        params        -- Parameters dictionary
        lambda_1      -- Conditional losses weight parameter (default 10.0)
        lambda_2      -- Reconstruction loss weight parameter (default 0.2)
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.generator = generator
        self.discriminator = discriminator
        self.params = params
        self.image = image
        self.text = text
        self.text_length = text_length
        inds = torch.roll(torch.arange(self.text_length.size(0)), 1)
        self.negative_text = self.text[inds]
        self.negative_text_length = self.text_length[inds]

    def loss_generator(self):
        """Generator loss function (fake images evaluation)"""
        fake, mu, log_sigma = self.generator(self.image, self.negative_text, self.negative_text_length)
        uncond, cond, _ = self.discriminator(fake, self.negative_text, self.negative_text_length)
        l1 = bce(uncond, torch.ones_like(uncond))
        self.params["uncond_gen"] += l1
        l2 = bce(cond, torch.ones_like(cond))
        self.params["cond_p_gen"] += l2
        log_Sigma = 2 * log_sigma
        kl = torch.mean(-log_sigma + (torch.exp(log_Sigma) + mu**2 - 1.0) * 0.5)
        self.params["kl"] += kl * 0.5
        return 0.5 * kl + self.lambda_1 * l1 + l2, fake

    def loss_generator_reconstruction(self):
        """Generator reconstruction loss"""
        reconstruction, mu, log_sigma = self.generator(self.image, self.text, self.text_length)
        score = l1(reconstruction, self.image)
        self.params["l1_reconstruction"] += l1
        log_Sigma = 2 * log_sigma
        kl = torch.mean(-log_sigma + (torch.exp(log_Sigma) + mu**2 - 1.0) * 0.5)
        self.params["kl"] += kl * 0.5
        return 0.5 * kl + score * self.lambda_2, kl

    def loss_real_discriminator(self):
        """Discriminator loss for real images"""
        uncond, cond, cond_n = self.discriminator(self.image, self.text, self.text_length)
        l1 = bce(uncond, torch.ones_like(uncond))
        l2 = bce(cond, torch.ones_like(cond))
        l3 = bce(cond_n, torch.zeros_like(cond_n))
        self.params["uncond_disc_real"] += l1
        avg_l2_l3 = (l2 + l3) / 2.0
        self.params["cond_disc_real"] += avg_l2_l3
        return l1 + self.lambda_1 * avg_l2_l3

    def loss_synthetic_discriminator(self):
        """Discriminator loss for synthetic images"""
        fake, _, _ = self.generator(self.image, self.negative_text, self.negative_text_length)
        uncond, _, _ = self.discriminator(fake.detach(), self.negative_text, self.negative_text_length)
        loss = bce(uncond, torch.zeros_like(uncond))
        self.params["cond_disc_fake"] += loss
        return loss
