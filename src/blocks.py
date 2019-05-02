# TAGAN implementation

import torch
import torch.nn


class ConditioningAugmentation(torch.nn.Module):
    def forward(self, x):
        m = torch.mean(x)
        s = torch.std(x)
        return torch.randn(x.shape) * s + m


class TemporalAverage(torch.nn.Module):
    def forward(self, x):
        return x.mean()


class TextEncoder(torch.nn.Module):
    def __init__(self, num_words):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.GRU(input_size=num_words*300, hidden_size=512, bidirectional=True),
            TemporalAverage(),
            torch.nn.LeakyReLU(),
            ConditioningAugmentation()
        )

    def forward(self, text):
        o = self.bgru.forward(text)


class ImageEncoder(torch.nn.Module):
    pass


class ConcatABResidualBlocks(torch.nn.Module):
    pass


class ResidualBlock(torch.nn.Module):
    pass


class UnconditionalDiscriminator(torch.nn.Module):
    pass


class Decoder(torch.nn.Module):
    pass


class TextAdaptiveDiscriminator(torch.nn.Module):
    pass


class ConditionalDiscriminator(torch.nn.Module):
    pass


class Generator(torch.nn.Module):
    def __init__(self, num_words):
        super().__init__()
        self.main = torch.nn.Sequential(
            TextEncoder(num_words),
            ImageEncoder(),
            ConcatABResidualBlocks(),
            ResidualBlock(),
            Decoder()
        )

    def forward(self, xtext, ximage):
        # x includes both the text and the image
        a = self.a.forward(xtext)
        b = self.b.forward(ximage)
        ab = self.ab.forward(a, b)
        c = self.c.forward(ab)
        d = self.d.forward(b + c)
        return d


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ie = ImageEncoder()
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
