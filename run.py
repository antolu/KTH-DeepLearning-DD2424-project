import sys
sys.path.append("src")

import matplotlib
import numpy as np
import torch.optim as optim
import visdom
import torch
import time
from loss import Loss
from blocks import Discriminator
from math import ceil
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange
from utils import Utils
from blocks import Generator
from image_io import disp_sidebyside, save_img
from load_dataset import ParseDatasets
from preprocess_caption import PreprocessCaption


matplotlib.use('tkagg')

# torch.manual_seed(0)

args = Utils.parse_args()

# Check arguments
supported_datasets = ["cub", "oxford", "coco"]
supported_coco_sets = ["train", "val", "test", None]
if args.dataset not in supported_datasets:
    raise Exception(f"The supplied dataset parameter {args.dataset} is not supported.")
if args.coco_set not in supported_coco_sets:
    raise Exception(f"The supplied coco set parameter {args.coco_set} is not supported.")

# Load fastText
pc = PreprocessCaption(args.fasttext_model)
pc.load_fasttext()

# Initialise transform
if args.mode == "train":
    tf = transforms.Compose([
        transforms.Resize(136),
        transforms.RandomCrop(128),
        transforms.RandomRotation((-10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
elif args.mode == "test":
    tf = transforms.Compose([
        transforms.Resize(136),
        transforms.RandomCrop(128),
        transforms.ToTensor()
    ])

# Load blacklisted images
if args.blacklist is not None:
    blacklist = Utils.read_blacklist(args.blacklist)
else:
    blacklist = None

# Parse datasets
print("Parsing datasets")
pd = ParseDatasets(dataset=args.dataset, images_root=args.images_root,
                   annotations_root=args.annotations_root,
                   preprocess_caption=pc, transform=tf, blacklist=blacklist)

# Load datasets
train_set, val_set, test_set = pd.get_datasets()

# Check if GPU can be used, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {device}.")

# Initialise models
generator = Generator(device).to(device)
discriminator = Discriminator(device).to(device)

# Load pretrained models
if args.pretrained_generator is not None:
    print("Loading pretrained generator")
    pretrained_generator = torch.load(args.pretrained_generator, map_location=device)
    generator.load_state_dict(pretrained_generator, strict=False)

if args.mode == "train" and args.pretrained_discriminator is not None:
    print("Loading pretrained discriminator")
    pretrained_discriminator = torch.load(args.pretrained_discriminator, map_location=device)
    discriminator.load_state_dict(pretrained_discriminator, strict=False)

if args.mode == "train":
    # Initialise train routine

    # Use a visualizer of training samples if available
    vis = visdom.Visdom()

    # Set networks to train mode
    generator.train()
    discriminator.train()

    # Initialise optimizers
    od = optim.Adam(discriminator.parameters(),
                    lr=args.learning_rate,
                    betas=(args.momentum, 0.999))

    og = optim.Adam(generator.parameters(), lr=args.learning_rate,
                    betas=(args.momentum, 0.999))

    # Load pretrained optimizers
    if args.pretrained_optimizer_discriminator is not None:
        pretrained_optimizer_discriminator = torch.load(
            args.pretrained_optimizer_discriminator)
        od.load_state_dict(pretrained_optimizer_discriminator)

    if args.pretrained_optimizer_generator is not None:
        pretrained_optimizer_generator = torch.load(
            args.pretrained_optimizer_generator)
        og.load_state_dict(pretrained_optimizer_generator)

    # Initialise data loader
    dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4,
                            shuffle=True)

    # Initialise losses file
    losses = open(f"losses_{int(time.time())}.csv", 'w')
    losses.write("epoch,cond_disc_fake,cond_disc_real,uncond_disc_real,"
                 "l1_reconstruction,kl,cond_p_gen,uncond_gen\n")

    try:
        with trange(args.no_epochs) as t:
            for epoch in t:
                params = {
                    "cond_disc_fake": 0.0,
                    "cond_disc_real": 0.0,
                    "uncond_disc_real": 0.0,
                    "l1_reconstruction": 0.0,
                    "kl": 0.0,
                    "cond_p_gen": 0.0,
                    "uncond_gen": 0.0
                }
                if ((epoch + 1) % 100) == 0:
                    optimizers = [od, og]
                    for o in optimizers:
                        for param_group in o.param_groups:
                            param_group['lr'] /= 2.0

                for i_batch, (img, caption, no_words) in enumerate(dataloader):
                    # Do training

                    img, caption, no_words = img.to(device), caption.to(device), no_words.to(device)

                    # Images normalization
                    img = img.mul(2)
                    img = img.sub(1)

                    loss = Loss(image=img, text=caption,
                                text_length=no_words.squeeze(),
                                generator=generator,
                                discriminator=discriminator,
                                params=params, lambda_1=args.lambda_1,
                                lambda_2=args.lambda_2)

                    # Train step discriminator
                    discriminator.zero_grad()
                    lrd = loss.loss_real_discriminator()
                    lrd.backward()
                    lsd = loss.loss_synthetic_discriminator()
                    lsd.backward()
                    od.step()

                    # Train step generator
                    generator.zero_grad()
                    lgs, fake = loss.loss_generator()
                    lgs.backward()
                    lgr, kld = loss.loss_generator_reconstruction()
                    lgr.backward()
                    og.step()

                    # Denominator for the averages
                    den = (i_batch + 1)

                    # Calculate averages
                    cond_disc_fake = params["cond_disc_fake"] / den
                    cond_disc_real = params["cond_disc_real"] / den
                    l1_reconstruction = params["l1_reconstruction"] / den
                    kl = params["kl"] / den
                    cond_p_gen = params["cond_p_gen"] / den
                    uncond_gen = params["uncond_gen"] / den
                    uncond_disc_real = params["uncond_disc_real"] / den

                    # Show losses information in the progress bar
                    t.set_description(
                        f"E:{epoch}|"
                        f"B:{den}/{ceil(len(train_set) / args.batch_size)}|"
                        f"uD:{uncond_disc_real:.4}|"
                        f"c+D:{cond_disc_real:.4}|"
                        f"c-D:{cond_disc_fake:.4}|"
                        f"L1:{l1_reconstruction:.4}|"
                        f"uG:{uncond_gen:.4}|"
                        f"cG:{cond_p_gen:.4}|"
                        f"k:{kl:.4}"
                    )

                # Log the losses in the file
                losses.write(f"{epoch},{cond_disc_fake},{cond_disc_real},{uncond_disc_real}"
                             f",{l1_reconstruction},{kl},{cond_p_gen},{uncond_gen}\n")

                # Save the partial models and optimizers
                if (epoch + 1) % args.save_models_frequency == 0:
                    torch.save(generator.state_dict(),
                               f"./models/run_G_dataset_{args.dataset}_epoch_{epoch + 1}.pth")
                    torch.save(discriminator.state_dict(),
                               f"./models/run_D_dataset_{args.dataset}_epoch_{epoch + 1}.pth")
                    torch.save(od.state_dict(),
                               f"./models/run_od_dataset_{args.dataset}_epoch_{epoch + 1}.pth")
                    torch.save(og.state_dict(),
                               f"./models/run_og_dataset_{args.dataset}_epoch_{epoch + 1}.pth")

                # Rescale images for visualising and visualise
                img_vis = img.mul(0.5).add(0.5)
                vis.images(img_vis.cpu().detach().numpy(), nrow=4, opts=dict(title=f'original_e{epoch + 1}'))
                fake_vis = fake.mul(0.5).add(0.5)
                vis.images(fake_vis.cpu().detach().numpy(), nrow=4, opts=dict(title=f'generated_e{epoch + 1}'))

    except KeyboardInterrupt:
        pass
    finally:
        # Save the last models before terminating training
        torch.save(od.state_dict(),
                   f"./models/run_od_dataset_{args.dataset}_before_terminating.pth")
        torch.save(og.state_dict(),
                   f"./models/run_og_dataset_{args.dataset}_before_terminating.pth")
        torch.save(generator.state_dict(),
                   f"./models/run_G_dataset_{args.dataset}_before_terminating.pth")
        torch.save(discriminator.state_dict(),
                   f"./models/run_D_dataset_{args.dataset}_before_terminating.pth")
        losses.close()

elif args.mode == 'test':
    # Set generator in test mode
    generator.eval()

    # Start a testing loop
    while True:
        # Pick a random sample from the training set
        i = np.random.choice(len(train_set))
        tensor, caption_vec, no_words, caption, img = train_set.get(i)

        print("""
The original caption is \"{}\". Enter your modified caption.
Leave blank for the original caption""".format(caption))

        # Wait for user input
        try:
            cap = input()
        except EOFError:
            exit()

        if cap != "":
            caption_vec, no_words = pc.string_to_vector(cap)

        # Generate a modified image
        generated, _, _ = generator(tensor.unsqueeze(0).to(device), caption_vec.unsqueeze(0).to(
            device), no_words.to(device))

        # Display image
        disp_sidebyside([img, tensor.cpu().squeeze(), generated.cpu().squeeze()], caption=cap)

        # Save image to file
        save_img(img, caption, "{}_orig".format(i), "results")
        save_img(generated.cpu().squeeze(), cap, "{}_gen".format(i), "results")
