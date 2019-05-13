import matplotlib
import sys
sys.path.append("src")
from blocks import Discriminator
from loss import loss_discriminator, loss_generator
from utils import Utils

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
from tqdm import trange
from math import ceil
from src.argparser import parse_args
from src.blocks import Generator
from src.image_io import *
from src.load_dataset import ParseDatasets, Dataset
from src.preprocess_caption import PreprocessCaption

import torch, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

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
if args.pretrained_generator is not None :
    pretrained_generator = torch.load(args.pretrained_generator)
    generator.load_state_dict(pretrained_generator, strict=False)
if args.pretrained_discriminator is not None :
    pretrained_discriminator = torch.load(args.pretrained_discriminator)
    discriminator.load_state_dict(pretrained_discriminator, strict=False)


if args.runtype == "train" :
    generator.train()
    discriminator.train()

    od = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    og = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    dataloader = DataLoader(train_set, batch_size=64, num_workers=4, shuffle=True)

    lg = ld = -1
    try:
        with trange(args.no_epochs) as t:
            for epoch in t: 
                for i_batch, (img, caption, no_words) in enumerate(dataloader):
                    t.set_description('Epoch: {} | Batch: {}/{} | LG: {} | LD: {}'.format(epoch, i_batch + 1, ceil(len(train_set)/64), lg, ld))
                    # Do training

                    img, caption, no_words = img.to(device), caption.to(device), no_words.to(device)

                    discriminator.zero_grad()
                    ld = loss_discriminator(img, caption, no_words, discriminator, generator, 10.0)
                    ld.backward()
                    od.step()

                    generator.zero_grad()
                    lg = loss_generator(img, caption, no_words, discriminator, generator, 10.0, 2.0)
                    lg.backward()
                    og.step()
                if epoch % 50 == 0:
                    torch.save(generator.state_dict(), "./models/run_G_dataset_{}_epoch_{}.pth".format(args.dataset, epoch))
                    torch.save(discriminator.state_dict(), "./models/run_D_dataset_{}_epoch_{}.pth".format(args.dataset, epoch))
    except KeyboardInterrupt:
        pass
    finally:
        torch.save(generator.state_dict(), "./models/run_G_dataset_{}_before_dying.pth".format(args.dataset))
        torch.save(discriminator.state_dict(), "./models/run_D_dataset_{}_before_dying.pth".format(args.dataset))
   
elif args.runtype == 'test' :

    # How to call generator
    print("Calling generator")
    generator.eval()

    while True :
        i = np.random.choice(len(test_set))
        tensor, caption_vec, no_words, caption, img = test_set[i]

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
