import torchfile
import torch
from src.load_dataset import *

cap = torchfile.load("datasets/cub/cub_icml/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.t7")
cap_char_vec = cap.char

tools = CaptionTools()

for i in range(cap_char_vec.shape[1]) :
    chars = tools.num2char(cap_char_vec[:, i])
    print(chars.rstrip())

captions_root = "datasets/cub/cub_icml"
images_root = "datasets/cub/CUB_200_2011"

# captions_root = "datasets/oxford/flowers_icml"
# images_root = "datasets/oxford/jpg"

pd = ParseDatasets(images_root, captions_root, dataset="cub")

# caps, classes = pd.read_all_captions()
# img_paths = pd.find_image_paths(caps)
# imgs, allclasses = pd.read_all_images()

# merged = pd.merge_captions_images(caps, imgs)