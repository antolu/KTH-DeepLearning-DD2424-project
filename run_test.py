
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
from src.argparser import parse_args
from src.blocks import Generator
from src.image_io import *
from src.load_dataset import ParseDatasets, Dataset
from src.preprocess_caption import PreprocessCaption

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

args = parse_args()

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

# Parse datasets
print("Parsing datasets")
pd = ParseDatasets(dataset=args.dataset, images_root=args.images_root, annotations_root=args.annotations_root, preprocess_caption=pc, transform=tf)

train_set, val_set, test_set = pd.get_datasets()

# Load pretreined model
if args.pretrained_model is not None :
    pretrained_model = torch.load(args.pretrained_model)

print("Loading pretrained model")
generator = Generator(args.max_no_words)

if pretrained_model is not None :
    generator.load_state_dict(pretrained_model, strict=False)
generator.eval()

# How to call generator
print("Calling generator")

while True :
    i = np.random.choice(len(test_set))
    sample = test_set[i]

    # print("Generating for sample with caption \"{}\"".format(sample["caption"]))

    print("The original caption is \"{}\". Enter your modified caption. \nLeave blank for the original caption".format(sample["caption"]))
    cap = input()
    if cap == "" :
        caption_vec = sample["caption_vector"]
        no_words = sample["no_words"]
        cap = sample["caption"]
    else :
        caption_vec, no_words = pc.string_to_vector(cap, args.max_no_words)

    generated_img = generator(sample["tensor"], caption_vec, torch.Tensor([no_words]))

    disp_sidebyside([sample["img"], sample["tensor"].squeeze(), generated_img.squeeze()], caption=cap)

    prompt = input("Do you want to keep generating more images? (y/n) ")
    if prompt != "y" :
        break