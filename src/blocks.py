import torch


class TextEncoder(torch.nn.Module):
    pass


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
    def __init__(self):
        self.a = TextEncoder()
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
        d = self.d.forward(c)
        return d


class Discriminator(torch.nn.Module):
    def __init__(self):
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
