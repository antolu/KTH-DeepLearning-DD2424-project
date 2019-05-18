# TAGAN implementation
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, num_words, device="cpu"):
        super().__init__()
        self.device = device
        self.a = TextEncoderGenerator(num_words, device)
        self.b = ImageEncoderGenerator()
        self.ab = ConcatABResidualBlocks()

        self.d = Decoder()
        self.apply(initialize_parameters)

    def forward(self, ximage, xtext, xtext_lengths):
        # x includes both the text and the image
        a, mu, sigma = self.a(xtext, xtext_lengths)
        b = self.b(ximage)
        ab = self.ab(a, b)
        # c = b + ab
        c = ab
        d = self.d(b + c)
        return d, mu, sigma


class ConditioningAugmentation(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, mu, sigma):
        res = torch.zeros_like(mu)
        for i in range(mu.size(0)):
            res[i] = torch.randn(
                mu[i].shape).to(self.device) * sigma[i] + mu[i]
        return res


class TemporalAverage(nn.Module):
    def forward(self, x, mask):
        return x.sum(1) / mask.sum(1).unsqueeze(-1)


class TextEncoderGenerator(nn.Module):
    def __init__(self, num_words, device="cpu"):
        super().__init__()
        self.device = device
        self.gru_f = nn.GRUCell(input_size=300, hidden_size=512).to(device)
        self.gru_b = nn.GRUCell(input_size=300, hidden_size=512).to(device)

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

        self.cond_aug = ConditioningAugmentation(device)

    def forward(self, text, text_lengths):
        words_embs, mask = encode_text(text, text_lengths, self.gru_f,
                                       self.gru_b, self.device)
        avg = self.avg(words_embs, mask)
        mu = self.mu_cond_aug(avg)
        log_sigma = self.sigma_cond_aug(avg)
        final = self.cond_aug(mu, torch.exp(log_sigma))
        return final, mu, log_sigma


class ImageEncoderGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3, 1, 1)),
            ('act1', nn.ReLU(inplace=True)),

            # BN already includes a bias
            ('conv2', nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(128)),
            ('act2', nn.ReLU(inplace=True)),

            ('conv3', nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(256)),
            ('act3', nn.ReLU(inplace=True)),

            ('conv4', nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(512)),
            ('act4', nn.ReLU(inplace=True))
        ]))

    def forward(self, im):
        return self.main(im)


class ConcatABResidualBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(640, 512, 3, 1, 1, bias=False)),
                    ("bn1", nn.BatchNorm2d(512)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("res1", ResidualBlock(512, 512)),
                    ("res2", ResidualBlock(512, 512)),
                    ("res3", ResidualBlock(512, 512)),
                    ("res4", ResidualBlock(512, 512))
                ]
            )
        )

    def forward(self, text_embed, image_embed):
        text_embed = text_embed[:, :, None, None].repeat(1, 1,
                                                         image_embed.size(2),
                                                         image_embed.size(3))
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
                    ("conv1", nn.Conv2d(512, 512, 3, 1, 1, bias=False)),
                    ("bn1", nn.BatchNorm2d(512)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("conv2", nn.Conv2d(512, 512, 3, 1, 1, bias=False)),
                    ("bn2", nn.BatchNorm2d(512))
                ]
            )
        )

    def forward(self, x):
        c = self.main(x)
        return c + x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            OrderedDict(
                [
                    ("upsampling1", nn.Upsample(scale_factor=2)),

                    ("conv1", nn.Conv2d(512, 256, 3, 1, 1, bias=False)),
                    ("bn1", nn.BatchNorm2d(256)),
                    ("relu1", nn.ReLU(inplace=True)),

                    ("upsampling2", nn.Upsample(scale_factor=2)),

                    ("conv2", nn.Conv2d(256, 128, 3, 1, 1, bias=False)),
                    ("bn2", nn.BatchNorm2d(128)),
                    ("relu2", nn.ReLU(inplace=True)),

                    ("upsampling3", nn.Upsample(scale_factor=2)),

                    ("conv3", nn.Conv2d(128, 64, 3, 1, 1, bias=False)),
                    ("bn3", nn.BatchNorm2d(64)),
                    ("relu3", nn.ReLU(inplace=True)),

                    ("conv4", nn.Conv2d(64, 3, 3, 1, 1)),
                    ("tanh", nn.Tanh())
                ]
            )
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, num_words, device="cpu"):
        super().__init__()
        self.device = device

        # IMAGE ENCODER
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

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
            nn.Conv2d(512, 1, 4, padding=0, stride=1),
            # nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

        # TEXT ENCODER
        self.get_betas = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(dim=2)
        )

        self.gru_f = nn.GRUCell(input_size=300, hidden_size=512)
        self.gru_b = nn.GRUCell(input_size=300, hidden_size=512)
        self.avg = TemporalAverage()

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

        # self.get_Wb1 = nn.Linear(512, 257)
        # self.get_Wb2 = nn.Linear(512, 513)
        self.Wb1 = nn.Linear(512, 257)
        self.Wb2 = nn.Linear(512, 513)
        self.Wb3 = nn.Linear(512, 513)
        self.Wb = [self.Wb1, self.Wb2, self.Wb3]

        self.apply(initialize_parameters)

    def forward(self, image, text=None, len_text=None):

        image1 = self.conv3(image)
        image2 = self.conv4(image1)
        image3 = self.conv5(image2)
        GAP_image1 = self.GAP1(image1)
        GAP_image2 = self.GAP2(image2)
        GAP_image3 = self.GAP3(image3)
        GAP_images = [GAP_image1, GAP_image2, GAP_image3]
        d = self.un_disc(GAP_image3).squeeze()
        # d = torch.log(d)
        if text is None:
            return d.squeeze()

        # Get word embedding
        words_embs, mask = encode_text(text, len_text, self.gru_f, self.gru_b,
                                       self.device)
        avg = self.avg(words_embs, mask).unsqueeze(-1)

        # Calculate attentions
        u_dot_wi = torch.bmm(words_embs, avg).squeeze(-1)
        alphas = F.softmax(u_dot_wi, dim=1).permute(0, 1)

        # Get weights
        betas = self.get_betas(words_embs)

        total = 0
        total_neg = 0
        

        idx = np.arange(0, image.size(0))
        idx_neg = torch.tensor(np.roll(idx, 1))

        for j in range(3):
            image = GAP_images[j]
            image = image.mean(-1).mean(-1).unsqueeze(-1)

            # if j == 0:
            #     Wb = self.get_Wb1(words_embs)
            # else:
            #     Wb = self.get_Wb2(words_embs)

            Wb = self.Wb[j](words_embs)

            W = Wb[:, :, :-1]
            b = Wb[:, :, -1].unsqueeze(-1)

            W_neg = W[idx_neg]
            b_neg = b[idx_neg]
            betas_neg = betas.permute(2, 0, 1)
            f_neg = torch.sigmoid(torch.bmm(W_neg, image) + b_neg).squeeze(-1)
            total_neg += f_neg * betas_neg[j][idx_neg]
            f = torch.sigmoid(torch.bmm(W, image) + b).squeeze(-1)
            total += f * betas[:, :, j]

        alphas_neg = alphas[idx_neg, :]  # need to change this
        # total_neg should be (batch_size)
        total_neg = torch.pow(total_neg, alphas_neg).prod(1)
        # total should be (batch_size)
        total = torch.pow(total, alphas).prod(1)

        unconditional = d
        cond_positive = total
        cond_negative = total_neg
        return unconditional, cond_positive, cond_negative


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


def encode_text(text, text_length, gru_f, gru_b, device="cpu"):
    batch, seq_len, input_size = text.shape

    if text_length.size(0) != batch:
        raise ValueError

    hidden_f = torch.zeros(batch, gru_f.hidden_size).to(device)
    hidden_b = torch.zeros(batch, gru_b.hidden_size).to(device)

    text = text.permute(1, 0, 2)

    hidden_f_mat = torch.zeros(seq_len, batch, gru_f.hidden_size).to(device)
    hidden_b_mat = torch.zeros(seq_len, batch, gru_b.hidden_size).to(device)

    mask = torch.zeros(batch, seq_len).to(device)

    for i in range(seq_len):
        is_in_word = ((i < text_length)[:, None]).float()
        mask[:, i] = is_in_word.squeeze()

        hidden_f = gru_f(text[i], hidden_f)
        hidden_f_mat[i] = hidden_f * is_in_word

        # - i - 1 because the backward pass needs to incorporate both
        # -i and -i - 1 words per hidden state i
        hidden_b = gru_b(text[- i - 1], hidden_b)
        hidden_b_mat[-i] = hidden_b * is_in_word

    return ((hidden_f_mat + hidden_b_mat) / 2.0).permute(1, 0, 2), mask


class ResidualBlock_authors(nn.Module):
    def __init__(self, ndim):
        super(ResidualBlock_authors, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(ndim, ndim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ndim),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndim, ndim, 3, padding=1, bias=False),
            nn.BatchNorm2d(ndim)
        )

    def forward(self, x):
        return x + self.encoder(x)


class Generator_authors(nn.Module):
    def __init__(self):
        super(Generator_authors, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # residual blocks
        self.residual_blocks = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock_authors(512),
            ResidualBlock_authors(512),
            ResidualBlock_authors(512),
            ResidualBlock_authors(512)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

        # conditioning augmentation
        self.mu = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.apply(init_weights)

    def forward(self, img, txt):
        # image encoder
        e = self.encoder(img)

        # text encoder
        if type(txt) is not tuple:
            raise TypeError('txt must be tuple (txt_data, txt_len).')

        txt_data = txt[0]
        txt_len = txt[1]

        hi_f = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
        hi_b = torch.zeros(txt_data.size(1), 512, device=txt_data.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt_data.size(0)):
            mask_i = (txt_data.size(0) - 1 - i < txt_len).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt_data[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt_data[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        h = (h_f + h_b) / 2
        cond = h.sum(0) / mask.sum(0)

        z_mean = self.mu(cond)
        z_log_stddev = self.log_sigma(cond)
        z = torch.randn(cond.size(0), 128, device=txt_data.device)
        cond = z_mean + z_log_stddev.exp() * z

        # residual blocks
        cond = cond.unsqueeze(-1).unsqueeze(-1)
        merge = self.residual_blocks(torch.cat((e, cond.repeat(1, 1, e.size(2), e.size(3))), 1))

        # decoder
        d = self.decoder(e + merge)

        return d, (z_mean, z_log_stddev)


class Discriminator_authors(nn.Module):
    def __init__(self):
        super(Discriminator_authors, self).__init__()
        self.eps = 1e-7

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.GAP_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # text feature
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.gen_filter = nn.ModuleList([
            nn.Linear(512, 256 + 1),
            nn.Linear(512, 512 + 1),
            nn.Linear(512, 512 + 1)
        ])
        self.gen_weight = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(-1)
        )

        self.classifier = nn.Conv2d(512, 1, 4)

        self.apply(init_weights)

    def forward(self, img, txt, len_txt, negative=False):
        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)
        img_feats = [self.GAP_1(img_feat_1), self.GAP_2(img_feat_2), self.GAP_3(img_feat_3)]
        D = self.classifier(img_feat_3).squeeze()

        # text attention
        u, m, mask = self._encode_txt(txt, len_txt)
        att_txt = (u * m.unsqueeze(0)).sum(-1)
        att_txt_exp = att_txt.exp() * mask.squeeze(-1)
        att_txt = (att_txt_exp / att_txt_exp.sum(0, keepdim=True))

        weight = self.gen_weight(u).permute(2, 1, 0)

        sim = 0
        sim_n = 0
        idx = np.arange(0, img.size(0))
        idx_n = torch.tensor(np.roll(idx, 1), dtype=torch.long, device=txt.device)

        for i in range(3):
            img_feat = img_feats[i]
            W_cond = self.gen_filter[i](u).permute(1, 0, 2)
            W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
            img_feat = img_feat.mean(-1).mean(-1).unsqueeze(-1)

            if negative:
                W_cond_n, b_cond_n, weight_n = W_cond[idx_n], b_cond[idx_n], weight[i][idx_n]
                sim_n += torch.sigmoid(torch.bmm(W_cond_n, img_feat) + b_cond_n).squeeze(-1) * weight_n
            sim += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * weight[i]

        if negative:
            att_txt_n = att_txt[:, idx_n]
            sim_n = torch.clamp(sim_n + self.eps, max=1).t().pow(att_txt_n).prod(0)
        sim = torch.clamp(sim + self.eps, max=1).t().pow(att_txt).prod(0)

        if negative:
            return D, sim, sim_n
        return D, sim

    def _encode_txt(self, txt, len_txt):
        hi_f = torch.zeros(txt.size(1), 512, device=txt.device)
        hi_b = torch.zeros(txt.size(1), 512, device=txt.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt.size(0)):
            mask_i = (txt.size(0) - 1 - i < len_txt).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        u = (h_f + h_b) / 2
        m = u.sum(0) / mask.sum(0)
        return u, m, mask


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)
