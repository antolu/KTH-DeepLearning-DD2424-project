import torch
import torch.nn as nn


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


# # custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()


        # IMAGE ENCODER
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 64, 4, padding=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True), # I'm including this 'inplace' part since the example I am following has used it
            nn.Conv2d(64, 128, 4, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )


        # UNCONDITIONAL DISCRIMINATOR
        self.un_disc = nn.Sequential(
            nn.Conv2d(512, 1, 4),
            nn.Softmax()
        )


        # TEXT ENCODER
        self.bi_GRU = nn.GRU(300, 512, bidirectional=True)
        self.get_betas = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax()
        )



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

        # CONDITIONAL DISCRIMINATOR


    def forward(self, image, text):
        # x is the image

        image1 = self.conv3(image)
        image2 = self.conv4(image1)
        image3 = self.conv5(image2)
        GAP_image1 = self.GAP1(image1)
        GAP_image2 = self.GAP2(image2)
        GAP_image3 = self.GAP3(image3)
        #GAP_images = [GAP_image1, GAP_image2, GAP_image3]
        d = self.un_disc(GAP_image3)

        return d
