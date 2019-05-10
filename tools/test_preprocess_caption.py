import numpy as np
from src.load_dataset import CaptionTools, ParseCoco, ParseDatasets
from src.preprocess_caption import PreprocessCaption

pc = PreprocessCaption("fasttext/wiki.en.bin")
pc.load_fasttext()

captions_root = "datasets/cub/cub_icml"
images_root = "datasets/cub/CUB_200_2011"

pd = ParseDatasets(images_root, captions_root, dataset="cub")

# pc.string_to_vector(string)