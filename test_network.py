
from blocks import Generator, Discriminator
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from loss import loss_discriminator, loss_generator


def test_network():
    G = Generator(50)
    D = Discriminator(50)
    # torch.save(G.state_dict(), "output_dict.pth")
    #
    # model = torch.load("output_dict.pth")
    #
    # model_keys = model.keys()
    #
    # with open("layers_output.txt", "w") as f:
    #     for key in model_keys:
    #         f.write(str(key) + "\t\t\t\t\t " + str(model[key].shape) + "\n")
    input_img = Variable(torch.randn(64, 3, 128, 128))
    input_text = Variable(torch.randn(64, 50, 300))
    input_text_size = Variable(torch.randint(5, 15, size=(64,)))
    for i in range(1000):
        l = loss_generator(input_img, input_text, input_text, input_text_size, D, G, 0.5, 0.5)
        # l = loss_discriminator(input_img, input_text, input_text, input_text_size, D, G, 0.5)
