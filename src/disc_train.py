# Create the Discriminator
netD = Discriminator().to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function
netD.apply(weights_init)



# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Parameters
num_epochs = 600
lr = 0.0002
momentum = 0.5
batch_size = 64
lambda1 = 10
lambda2 = 2

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(momentum, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(momentum, 0.999))