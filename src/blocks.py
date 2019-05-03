# TAGAN implementation
from collections import OrderedDict

import torch
import torch.nn as nn


class ConditioningAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma):
        return torch.randn(mu.shape) * sigma + mu  # TODO: use matrix multiplications


class TemporalAverage(nn.Module):
    def forward(self, x):
        return x.mean()

class TextEncoderDiscriminator(nn.Module):
    pass


class TextEncoderGenerator(nn.Module):
    def __init__(self, num_words):
        super().__init__()
        self.first = nn.Sequential(
            nn.GRU(input_size=num_words*300, hidden_size=512, bidirectional=True),
            TemporalAverage()
        )
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

    def forward(self, text):
        first = self.first.forward(text)
        mu = self.mu_cond_aug(first)
        sigma = self.mu_cond_aug(first)
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
        return self.main.forward(im)


class ImageEncoderGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, 1, 1)),
            ('act1', nn.ReLU()),

            ('conv2', nn.Conv2d(64, 128, 4, 1, 2)),
            ('bn2', nn.BatchNorm2d(128)),
            ('act2', nn.ReLU()),

            ('conv3', nn.Conv2d(128, 256, 4, 1, 2)),
            ('bn3', nn.BatchNorm2d(256)),
            ('act3', nn.ReLU()),

            ('conv4', nn.Conv2d(256, 512, 4, 1, 2)),
            ('bn4', nn.BatchNorm2d(512)),
            ('act4', nn.ReLU())
        ]))

    def forward(self, im):
        return self.main.forward(im)


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
        self.main.forward()


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
        c = self.main.forward(x)
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
        return self.main.forward(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("upsampling1", nn.Upsample((512, 32, 32), 2.0)),

                    ("conv1", nn.Conv2d(512, 256, 3, 1, 1)),
                    ("bn1", nn.BatchNorm2d(256)),
                    ("relu1", nn.ReLU()),

                    ("upsampling2", nn.Upsample((256, 64, 64), 2.0)),

                    ("conv2", nn.Conv2d(256, 128, 3, 1, 1)),
                    ("bn2", nn.BatchNorm2d(128)),
                    ("relu2", nn.ReLU()),

                    ("upsampling", nn.Upsample((128, 128, 128), 2.0)),

                    ("conv3", nn.Conv2d(128, 64, 3, 1, 1)),
                    ("bn3", nn.BatchNorm2d(64)),
                    ("relu3", nn.ReLU()),

                    ("conv4", nn.Conv2d(64, 3, 3, 1, 1)),
                    ("tanh", nn.Tanh())
                ]
            )
        )

    def forward(self, x):
        return self.main.forward(x)


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
        self.c = ResidualBlock()
        self.d = Decoder()

        self.apply(initialize_parameters)

    def forward(self, xtext, ximage):
        a = self.a.forward(xtext)
        b = self.b.forward(ximage)
        ab = self.ab.forward(a, b)
        c = self.c.forward(ab)
        d = self.d.forward(b + c)
        return d


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.ie = ImageEncoderDiscriminator()
        self.ud = UnconditionalDiscriminator()
        self.te = TextEncoderDiscriminator()
        self.tad = TextAdaptiveDiscriminator()
        self.d = ConditionalDiscriminator()

    def forward(self, image, text):
        # x is the image
        ie = self.ie.forward(image)
        uc = self.ud.forward(ie)
        te = self.te.forward(text)
        tad = self.tad.forward(ie, te)
        d = self.d.forward(tad, uc)
        return d


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
