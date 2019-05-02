# TAGAN implementation
from collections import OrderedDict

import torch
import torch.nn as nn


class ConditioningAugmentation(nn.Module):
    def forward(self, x):
        m = torch.mean(x)
        s = torch.std(x)
        return torch.randn(x.shape) * s + m


class TemporalAverage(nn.Module):
    def forward(self, x):
        return x.mean()


class TextEncoder(nn.Module):
    def __init__(self, num_words):
        super().__init__()
        self.main = nn.Sequential(
            nn.GRU(input_size=num_words*300, hidden_size=512, bidirectional=True),
            TemporalAverage(),
            nn.LeakyReLU(),
            ConditioningAugmentation()
        )

    def forward(self, text):
        o = self.bgru.forward(text)


class ImageEncoderDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 4, 1, 2)),
            ('act1', nn.LeakyReLU(0.2)),

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


class ConcatABResidualBlocks(nn.Module):
    pass


class ResidualBlock(nn.Module):
    pass


class UnconditionalDiscriminator(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class TextAdaptiveDiscriminator(nn.Module):
    pass


class ConditionalDiscriminator(nn.Module):
    pass


class Generator(nn.Module):
    def __init__(self, num_words):
        super().__init__()
        self.a = TextEncoder(num_words)
        self.b = ImageEncoder()
        self.ab = ConcatABResidualBlocks()
        self.c = ResidualBlock()
        self.d = Decoder()

    def forward(self, xtext, ximage):
        # x includes both the text and the image
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
        self.te = TextEncoder()
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
