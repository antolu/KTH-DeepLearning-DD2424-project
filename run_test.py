import matplotlib
import sys
sys.path.append("src")
from blocks import Discriminator
from utils import Utils
from loss import loss_real_discriminator, loss_synthetic_discriminator, loss_generator, loss_generator_reconstruction
import visdom
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
from tqdm import trange
from math import ceil
from src.blocks import Generator
from src.image_io import *
from src.load_dataset import ParseDatasets, Dataset
from src.preprocess_caption import PreprocessCaption

import torch, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

torch.manual_seed(0)

args = Utils.parse_args()

# Check arguments
supported_datasets = ["cub", "oxford", "coco"]
supported_coco_sets = ["train", "val", "test", None]
if args.dataset not in supported_datasets :
    raise Exception("The supplied dataset parameter {} is not supported.".format(args.dataset))
if args.coco_set not in supported_coco_sets :
    raise Exception("The supplied coco set parameter {} is not supported.".format(args.coco_set))

# Load fastText
pc = PreprocessCaption(args.fasttext_model)
pc.load_fasttext()

# Initialise transform
tf = transforms.Compose([
    transforms.Resize(136),
    transforms.RandomCrop(128),
    transforms.RandomRotation((-10, 10)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

if args.blacklist is not None :
    blacklist = Utils.read_blacklist(args.blacklist)
else :
    blacklist = None

# Parse datasets
print("Parsing datasets")
pd = ParseDatasets(dataset=args.dataset, images_root=args.images_root, annotations_root=args.annotations_root, preprocess_caption=pc, transform=tf, blacklist=blacklist)

train_set, val_set, test_set = pd.get_datasets()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {}.".format(device))


print("Loading pretrained model")
generator = Generator(args.max_no_words, device).to(device)
discriminator = Discriminator(args.max_no_words, device).to(device)

# Load pretrained models
if args.pretrained_generator is not None:
    pretrained_generator = torch.load(args.pretrained_generator)
    generator.load_state_dict(pretrained_generator, strict=False)
if args.pretrained_discriminator is not None:
    pretrained_discriminator = torch.load(args.pretrained_discriminator)
    discriminator.load_state_dict(pretrained_discriminator, strict=False)

if args.runtype == "train":
    vis = visdom.Visdom()
    generator.train()
    discriminator.train()

    od = optim.Adam(generator.parameters(),
                    lr=0.0002/16,
                    betas=(0.5, 0.999))
    og = optim.Adam(discriminator.parameters(), lr=0.0002/16,
                    betas=(0.5, 0.999))

    # Load pretrained optimizers
    if args.pretrained_optimizer_discriminator is not None:
        pretrained_optimizer_discriminator = torch.load(
            args.pretrained_optimizer_discriminator)
        od.load_state_dict(pretrained_optimizer_discriminator)

    if args.pretrained_optimizer_generator is not None:
        pretrained_optimizer_generator = torch.load(
            args.pretrained_optimizer_generator)
        og.load_state_dict(pretrained_optimizer_generator)

    dataloader = DataLoader(train_set, batch_size=64, num_workers=4,
                            shuffle=True)

    lg = lgr = lsd = lrd = -1
    generator_losses = open("generator_losses.csv", 'w')
    discriminator_losses = open("discriminator_losses.csv", 'w')
    generator_losses.write("epoch,batch,loss\n")
    discriminator_losses.write("epoch,batch,loss\n")
    total_steps = 0
    try:
        with trange(args.no_epochs) as t:
            for epoch in t:
                for i_batch, (img, caption, no_words) in enumerate(dataloader):
                    t.set_description('Epoch: {} | Batch: {}/{} | LG: {} | LD: {}'.format(
                        epoch, i_batch + 1, ceil(len(train_set)/64), lg + lgr, lrd + lsd))
                    # Do training

                    if ((total_steps + 1) % 100) == 0:
                        optimizers = [od, og]
                        for o in optimizers:
                            for param_group in o.param_groups:
                                param_group['lr'] /= 2.0

                    img, caption, no_words = img.to(device), caption.to(device), no_words.to(device)
                    img = img.mul(2)
                    img = img.sub(1)

                    discriminator.zero_grad()
                    lrd = loss_real_discriminator(img, caption, no_words, discriminator, generator, 10.0)
                    lrd.backward()
                    lsd = loss_synthetic_discriminator(img, caption, no_words, discriminator, generator, 10.0)
                    lsd.backward()
                    od.step()

                    generator.zero_grad()
                    lg, fake = loss_generator(img, caption, no_words, discriminator, generator, 10.0, 0.2)
                    lg.backward()
                    lgr = loss_generator_reconstruction(img, caption, no_words, discriminator, generator, 10.0, 0.2)
                    lgr.backward()
                    generator_losses.write("{},{},{}\n".format(epoch, i_batch + 1, lg.detach().cpu().numpy().squeeze() + lgr.detach().cpu().numpy().squeeze()))
                    discriminator_losses.write("{},{},{}\n".format(epoch, i_batch + 1, lrd.detach().cpu().numpy().squeeze() + lsd.detach().cpu().numpy().squeeze()))
                    og.step()

                    total_steps += 1
                    
                if (epoch + 1) % 50 == 0:
                    torch.save(generator.state_dict(), "./models/run_G_dataset_{}_epoch_{}.pth".format(args.dataset, epoch))
                    torch.save(discriminator.state_dict(), "./models/run_D_dataset_{}_epoch_{}.pth".format(args.dataset, epoch))
                    torch.save(od.state_dict(), "./models/run_od_dataset_{}_epoch_{}.pth".format(args.dataset, epoch))
                    torch.save(og.state_dict(), "./models/run_og_dataset_{}_epoch_{}.pth".format(args.dataset, epoch))
                img_vis = img.mul(0.5).add(0.5)
                vis.images(img_vis.cpu().detach().numpy(), nrow=4, opts=dict(title='original'))
                fake_vis = fake.mul(0.5).add(0.5)
                vis.images(fake_vis.cpu().detach().numpy(), nrow=4, opts=dict(title='generated'))
    except KeyboardInterrupt:
        pass
    finally:
        torch.save(od.state_dict(), "./models/run_od_dataset_{}_before_dying.pth".format(args.dataset))
        torch.save(og.state_dict(), "./models/run_og_dataset_{}_before_dying.pth".format(args.dataset))
        torch.save(generator.state_dict(), "./models/run_G_dataset_{}_before_dying.pth".format(args.dataset))
        torch.save(discriminator.state_dict(), "./models/run_D_dataset_{}_before_dying.pth".format(args.dataset))
        discriminator_losses.close()
        generator_losses.close()

elif args.runtype == 'test':

    # How to call generator
    print("Calling generator")
    generator.eval()

    while True :
        i = np.random.choice(len(test_set))
        tensor, caption_vec, no_words, caption, img = test_set.get(i)

        # print("Generating for sample with caption \"{}\"".format(sample["caption"]))

        print("The original caption is \"{}\". Enter your modified caption. \nLeave blank for the original caption".format(caption))
        cap = input()
        if cap != "" :
            caption_vec, no_words = pc.string_to_vector(cap, args.max_no_words)

        print("running")
        generated, _, _ = generator(tensor.unsqueeze(0).to(device), caption_vec.unsqueeze(0).to(device), no_words.to(device))

        disp_sidebyside([img, tensor.cpu().squeeze(), generated.cpu().squeeze()], caption=cap)

        prompt = input("Do you want to keep generating more images? (y/n) ")
        if prompt != "y" :
            break
