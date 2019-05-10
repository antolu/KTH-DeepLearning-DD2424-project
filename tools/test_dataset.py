import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from src.load_dataset import ParseDatasets, Dataset
from src.preprocess_caption import PreprocessCaption
from torch.utils.data import Dataloader

import torch
from torchvision import transforms

pc = PreprocessCaption("fasttext/wiki.en.bin")
pc.load_fasttext()

tf = transforms.Compose([
    transforms.RandomRotation((-10, 10)),
    transforms.Resize(136),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# pd = ParseDatasets(dataset="coco", dataset_root="datasets/coco", keywords_file="caption_keywords.txt", data_set="train", preprocess_caption=pc, transform=tf)
pd = ParseDatasets(dataset="cub", images_root="datasets/cub/CUB_200_2011", captions_root="datasets/cub/cub_icml", preprocess_caption=pc, transform=tf)
# pd = ParseDatasets(dataset="oxford", images_root="datasets/oxford/jpd", captions_root="datasets/oxford/flowers_icml", preprocess_caption=pc, transform=tf)

train, val, test = pd.get_datasets()

def disp_tensor(tnsr) :
    img = tnsr.permute(1, 2, 0).numpy()

    plt.axis('off')
    plt.imshow(img)
    plt.show()
    plt.clf()

def disp_sidebyside(img, tnsr) :
    img_tnsr = tnsr.permute(1, 2, 0).numpy()
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(img)
    plt.axis('off')
    f.add_subplot(1,2, 2)
    plt.imshow(img_tnsr)
    plt.axis('off')
    plt.show(block=True)
    plt.clf()

sample = train[10]
disp_sidebyside(sample["img"], sample["tensor"])

dataloader = DataLoader(train, batch_size=4, shuffle=True, num_workers=4)

# Retrieve a local image
# img = pd.load_image(565004)
# plt.axis('off')
# plt.imshow(img)
# plt.show()