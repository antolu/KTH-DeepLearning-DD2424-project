# TAGAN implementation
from collections import OrderedDict

from torch import randn
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np


class ConditioningAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma):
        return torch.randn(mu.shape) * sigma + mu


class TemporalAverage(nn.Module):
    def forward(self, x):
        return torch.mean(x, 1)

class TextEncoderDiscriminator(nn.Module):
    pass


class TextEncoderGenerator(nn.Module):
    def __init__(self, num_words):
        super().__init__()
        self.gru_f = nn.GRUCell(input_size=300, hidden_size=512)
        self.gru_b = nn.GRUCell(input_size=300, hidden_size=512)
        self.avg = TemporalAverage()
        self.mu_cond_aug = nn.Sequential(
            OrderedDict(
                [
                    ("lin", nn.Linear(512, 128)),
                    ("act", nn.LeakyReLU(0.2)),
                ]
            )
        )

        self.sigma_cond_aug = nn.Sequential(
            OrderedDict(
                [
                    ("lin", nn.Linear(512, 128)),
                    ("act", nn.LeakyReLU(0.2)),
                ]
            )
        )

        self.cond_aug = ConditioningAugmentation()

    def forward(self, text, text_lengths):
        words_embs = encode_text(text, text_lengths, self.gru_f, self.gru_b)
        avg = self.avg(words_embs)
        mu = self.mu_cond_aug(avg)
        sigma = self.mu_cond_aug(avg)
        final = self.cond_aug(mu, sigma)
        return final


class ImageEncoderDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 4, 1, 2)),

            ('conv2', nn.Conv2d(64, 128, 4, 1, 2)),
            ('act2', nn.LeakyReLU(0.2)),

            ('conv3', nn.Conv2d(128, 256, 4, 1, 2)),
            ('act3', nn.LeakyReLU(0.2)),

            ('conv4', nn.Conv2d(256, 512, 4, 1, 2)),
            ('act4', nn.LeakyReLU(0.2)),

            ('conv5', nn.Conv2d(512, 512, 4, 1, 2)),
            ('act5', nn.LeakyReLU(0.2))]))

    def forward(self, im):
        return self.main(im)


class ImageEncoderGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, 1, 1)),
            ('act1', nn.ReLU()),

            ('conv2', nn.Conv2d(64, 128, 4, 2, 1)),
            ('bn2', nn.BatchNorm2d(128)),
            ('act2', nn.ReLU()),

            ('conv3', nn.Conv2d(128, 256, 4, 2, 1)),
            ('bn3', nn.BatchNorm2d(256)),
            ('act3', nn.ReLU()),

            ('conv4', nn.Conv2d(256, 512, 4, 2, 1)),
            ('bn4', nn.BatchNorm2d(512)),
            ('act4', nn.ReLU())
        ]))

    def forward(self, im):
        return self.main(im)


class ConcatABResidualBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(640, 512, 3, 1, 1)),
                    ("bn1", nn.BatchNorm2d(512)),
                    ("res1", ResidualBlock(512, 512)),
                    ("res2", ResidualBlock(512, 512)),
                    ("res3", ResidualBlock(512, 512)),
                    ("res4", ResidualBlock(512, 512))
                ]
            )
        )

    def forward(self, text_embed, image_embed):
        text_embed = text_embed[:, :, None, None].repeat(1, 1, image_embed.shape[2], image_embed.shape[3])
        x = torch.cat((image_embed, text_embed), 1)
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(512, 512, 3, 1, 1)),
                    ("bn1", nn.BatchNorm2d(512)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(512, 512, 3, 1, 1)),
                    ("bn2", nn.BatchNorm2d(512))
                ]
            )
        )

    def forward(self, x):
        c = self.main(x)
        return c + x


class UnconditionalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(512, 1, 4, 1, 0)),
                    ("act", nn.Softmax())
                ]
            )
        )
    def forward(self, x):
        return self.main(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("upsampling1", nn.Upsample(scale_factor=2)),

                    ("conv1", nn.Conv2d(512, 256, 3, 1, 1)),
                    ("bn1", nn.BatchNorm2d(256)),
                    ("relu1", nn.ReLU()),

                    ("upsampling2", nn.Upsample(scale_factor=2)),

                    ("conv2", nn.Conv2d(256, 128, 3, 1, 1)),
                    ("bn2", nn.BatchNorm2d(128)),
                    ("relu2", nn.ReLU()),

                    ("upsampling3", nn.Upsample(scale_factor=2)),

                    ("conv3", nn.Conv2d(128, 64, 3, 1, 1)),
                    ("bn3", nn.BatchNorm2d(64)),
                    ("relu3", nn.ReLU()),

                    ("conv4", nn.Conv2d(64, 3, 3, 1, 1)),
                    ("tanh", nn.Tanh())
                ]
            )
        )

    def forward(self, x):
        return self.main(x)


class TextAdaptiveDiscriminator(nn.Module):
    pass


class ConditionalDiscriminator(nn.Module):
    pass


class Generator(nn.Module):
    def __init__(self, num_words):
        super().__init__()
        self.a = TextEncoderGenerator(num_words)
        self.b = ImageEncoderGenerator()
        self.ab = ConcatABResidualBlocks()

        self.d = Decoder()
        self.apply(initialize_parameters)

        # We keep this here for training the conditioning augmentation (https://arxiv.org/pdf/1612.03242.pdf eq 2)
        self.cond_aug_params = self.a.mu_cond_aug, self.a.sigma_cond_aug

    def forward(self, ximage, xtext, xtext_lengths):
        # x includes both the text and the image
        a = self.a(xtext, xtext_lengths)
        b = self.b(ximage)
        ab = self.ab(a, b)
        c = b + ab
        d = self.d(c)
        return d


class Discriminator(nn.Module):
    def __init__(self, num_words):
        super().__init__()

        # IMAGE ENCODER
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True), # I'm including this 'inplace' part since the example I am following has used it
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )


        # UNCONDITIONAL DISCRIMINATOR
        self.un_disc = nn.Sequential(
            nn.Conv2d(512, 1, 4),
            nn.Softmax()
        )


        # TEXT ENCODER
        self.get_betas = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax()
        )

        self.gru_f = nn.GRUCell(input_size=300, hidden_size=512)
        self.gru_b = nn.GRUCell(input_size=300, hidden_size=512)
        self.avg = TemporalAverage()



        #
        self.GAP1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.GAP2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # This is same as self.GAP2
        self.GAP3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # CONDITIONAL DISCRIMINATOR contained in forward pass

        self.get_Wb1 = nn.Linear(512, 257)
        self.get_Wb2 = nn.Linear(512, 513)

        self.apply(initialize_parameters)

    def forward(self, image, text, len_text, negative=False):

        image1 = self.conv3(image)
        image2 = self.conv4(image1)
        image3 = self.conv5(image2)
        GAP_image1 = self.GAP1(image1)
        GAP_image2 = self.GAP2(image2)
        GAP_image3 = self.GAP3(image3)
        GAP_images = [GAP_image1, GAP_image2, GAP_image3]
        d = self.un_disc(GAP_image3).squeeze()

        # Get word embedding
        words_embs = encode_text(text, len_text, self.gru_f, self.gru_b)
        avg = self.avg(words_embs).unsqueeze(-1)

        # Calculate attentions
        u_dot_wi = torch.bmm(words_embs, avg).squeeze(-1)
        alphas = F.softmax(u_dot_wi)

        # Get weights
        betas = self.get_betas(words_embs)

        total = 0
        total_neg = 0

        idx = np.arange(0, image.size(0))
        idx_neg = torch.tensor(np.roll(idx, 1))

        for j in range(3):
            image = GAP_images[j]
            image = image.mean(-1).mean(-1).unsqueeze(-1)

            if j == 0:
                Wb = self.get_Wb1(words_embs)
            else:
                Wb = self.get_Wb2(words_embs)

            W = Wb[:, :, :-1]
            b = Wb[:, :, -1].unsqueeze(-1)

            if negative:
                W_neg = W[idx_neg]
                b_neg = b[idx_neg]
                betas_neg = betas.permute(2,0,1)
                f_neg = torch.sigmoid(torch.bmm(W_neg, image) + b_neg).squeeze(-1)
                total_neg += f_neg * betas_neg[j][idx_neg]
            f = torch.sigmoid(torch.bmm(W, image) + b).squeeze(-1)
            total += f * betas[:, :, j]

        if negative:
            alphas_neg = alphas[idx_neg, :]  # need to change this
            total_neg = total_neg.t().pow(alphas_neg.t()).prod(0)  # total_neg should be (batch_size)
        total = total.t().pow(alphas.t()).prod(0)  # total should be (batch_size)

        if negative:
            return d, total, total_neg
        return d, total


def initialize_parameters(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        if model.weight.requires_grad:
            model.weight.data.normal_(std=0.02)
        if model.bias is not None and model.bias.requires_grad:
            model.bias.data.fill_(0)
    elif isinstance(model, nn.BatchNorm2d) and model.affine:
        if model.weight.requires_grad:
            model.weight.data.normal_(1, 0.02)
        if model.bias.requires_grad:
            model.bias.data.fill_(0)

def encode_text(text, text_length, gru_f, gru_b):
    batch, seq_len, input_size = text.shape

    if text_length.size(0) != batch:
        raise ValueError

    hidden_f = torch.zeros(batch, gru_f.hidden_size)
    hidden_b = torch.zeros(batch, gru_b.hidden_size)

    text = text.permute(1, 0, 2)

    hidden_f_mat = torch.zeros(seq_len, batch, gru_f.hidden_size)
    hidden_b_mat = torch.zeros(seq_len, batch, gru_b.hidden_size)

    for i in range(seq_len):
        is_in_word = ((i < text_length)[:, None]).float()

        hidden_f = gru_f(text[i], hidden_f)
        hidden_f_mat[i] = hidden_f * is_in_word

        hidden_b = gru_b(text[- i - 1], hidden_b)  # -i-1 because the backward pass needs to incorporate both -i and and -i-1 words per hidden state i
        hidden_b_mat[-i] = hidden_b * is_in_word

    return ((hidden_f_mat + hidden_b_mat) / 2.0).permute(1, 0, 2)