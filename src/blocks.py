# TAGAN implementation
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        self.text_encoder = RNN_ENCODER(num_words, ninput=300, drop_prob=0.5,
                                        nhidden=512, nlayers=1, bidirectional=True,
                                        n_steps=50, rnn_type="GRU")
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
        batch_size = text.size(0)
        hidden = self.text_encoder.init_hidden(batch_size)
        words_embs, _ = self.text_encoder(text, text_lengths, hidden)
        words_embs = words_embs.detach()
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


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=512, nlayers=1, bidirectional=True,
                 n_steps=50, rnn_type="GRU"):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = n_steps
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


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

    def forward(self, xtext, ximage):
        # x includes both the text and the image
        a = self.a(xtext)
        b = self.b(ximage)
        ab = self.ab(a, b)
        c = b + ab
        d = self.d(c)
        return d


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.ie = ImageEncoderDiscriminator()
        self.ud = UnconditionalDiscriminator()
        self.te = TextEncoderDiscriminator()
        self.tad = TextAdaptiveDiscriminator()
        self.d = ConditionalDiscriminator()
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
