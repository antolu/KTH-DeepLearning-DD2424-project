from blocks import Generator
import torch.nn as nn
import torch.optim as optim


def test_network():
    G = Generator(10)
    Loss = nn.MSELoss()
    Optim = optim.Adam()
