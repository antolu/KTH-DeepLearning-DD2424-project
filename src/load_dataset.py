from pycocotools.coco import COCO
import numpy as np
import scipy.ndimage
import os
import torchfile

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

class CaptionTools:
    def __init__(self) :
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

    def num2char(self, nums) :
        char = ""
        for num in nums :
            char += self.alphabet[num-1]

        return char

class ParseCoco:
    """
    Reads and parses captions and images of the COCO dataset
    """
    def __init__(self, data_root="datasets/coco") :
        self.data_root = data_root

    def read_keywords(self):
        """
        Reads keywords we want the captions to contain from "caption_keywords.txt"
        :return: A set of the keywords
        """
        with open("caption_keywords.txt", "r") as f:
            keywords = [line[:len(line) - 1] for line in f]

        return keywords


    def read_coco(self, set='train'):
        """
        Reads the training dataset annotations from disk
        :return: COCO objects for annotations and captions
        """
        if set == 'train':
            dataType = 'train2017'
        elif set == 'val':
            dataType = 'val2017'
        else :
            raise "set option " + set + " was not recognized!"

        captionFile = '{}/annotations/captions_{}.json'.format(self.data_root, dataType)
        annFile = '{}/annotations/instances_{}.json'.format(self.data_root, dataType)

        coco = COCO(annFile)
        coco_caps = COCO(captionFile)

        return coco, coco_caps


    def filter_coco(self, keywords, coco, coco_caps):
        """
        Finds the images in COCO dataset that contains one or several of the relevant keywords
        :param keywords: A set of keywords, and each image caption should contain at least one of the keywords
        :param coco: The COCO dataset annotations
        :param coco_caps: The COCO dataset captions
        :return: A dictionary with key corresponding to a an image ID, and the value is another dictionary where the "captions" key yields a list of the image captions
        """
        imgIds = coco.imgs.keys()

        annIds = coco_caps.getAnnIds(imgIds=imgIds)  # [21616, 53212]
        anns = coco_caps.loadAnns(
            annIds)  # [{'image_id': 540372, 'id': 300754, 'caption': 'A cow standing in an empty city block '}]

        filteredImgs = {}
        sortedImgs = {}

        # One image might have several captions, concatenate all captions of one image id to a single dict
        for img in anns:
            imgID = img['image_id']

            if imgID in sortedImgs:
                savedImg = sortedImgs[imgID]
            else:
                savedImg = {'image_id': imgID, 'id': img['id'], 'captions': []}

            caption = img['caption']
            savedImg['captions'].append(caption)
            sortedImgs[imgID] = savedImg

        # Go through all image captions, if at least one of the captions of an image contains a keyword, save the image and all captions of that image
        for imgID in sortedImgs:

            for caption in sortedImgs[imgID]['captions']:
                if len(set(caption.split()).intersection(keywords)) > 0:
                    filteredImgs[imgID] = sortedImgs[imgID]
                    break

        return filteredImgs

    def load_image(self, image_id, dataset='train'):
        """
        Loads the image in `image_id` into memory as a 3D rgb array
        :param image_id The image id (in integer form)
        :param dataset, the dataset to load the image from. Valid values are 'train' and 'val'
        :return Returns a 3D rgb array of the image
        """

        if dataset=='train' :
            dataType = 'train2017'
        elif dataset=='val' :
            dataType = 'val2017'
        else:
            raise "set option " + dataset + " was not recognized!"

        image_id = str(image_id)
        image_id = (12-len(image_id)) * "0" + image_id

        image_file = '{}/{}/{}.jpg'.format(self.data_root, dataType, image_id)

        image_array = scipy.ndimage.imread(image_file)

        return image_array

class ParseDatasets(CaptionTools) :
    """
    Class for reading and parsing CUB and oxford datasets
    """
    def __init__(self, images_root, captions_root) :
        """
        Initialises this class for one dataset
        :param images_root The root directory of the images of this dataset
        :param captions_root The root directory of then captions of this dataset
        """
        super().__init__()
        self.images_root = images_root
        self.captions_root = captions_root

    def read_all_captions(self) :
        """
        Reads all captions for this dataset (defined in init) into memory
        :return Returns a dictionary where the key is the filename of the image, and the value is 
                a dictionary with a key 'captions' whose value is list containing the image captions.
                In essence: data_dict["<filename>"]["captions"] == <list of captions>
                Also returns a list of all the classes the images are from.
        """

        if not os.path.exists(self.captions_root) :
            raise "Path" + self.captions_root + "does not exist!"

        # Find all folders of the captions
        with open(os.path.join(self.captions_root, "allclasses.txt")) as f :
            allclasses = [line.rstrip() for line in f]

        data_dict = {}

        # Iterate over all classes, convert num representation to chars
        progress = 0.0
        for dataclass in allclasses :
            dataclass_dir = os.path.join(self.captions_root, dataclass)
            for filename in os.listdir(dataclass_dir) :
                filepath = os.path.join(dataclass_dir, filename)

                captions = self.load_caption(filepath)

                data_dict[os.path.splitext(filename)[0]] = {"captions": captions}
            
            progress += 1.0
            print("Progress: ", round(progress/len(allclasses) * 100, 2), "%")

        return data_dict, allclasses

    def load_caption(self, filepath) :
        """
        Loads captions from a single file and returns them in a list
        :param filepath The path to the caption file (.t7 format)
        :return Returns a list of all the captions contained in the file
        """
        caption_data = torchfile.load(filepath)

        caption_char_vec = caption_data.char # get the char vector in nums

        captions = []
        for i in range(caption_char_vec.shape[1]) :
            caption = self.num2char(caption_char_vec[:, i]).rstrip()
            captions.append(caption)

        return captions

    def read_all_images(self) :
        """
        Reads all images for this dataset (defined in init) into memory
        :return Returns a dictionary where the key is the filename of the image, and the value is 
                a dictionary with a key 'image' whose value is a 3D rgb array representing the image.
                In essence: data_dict["<filename>"]["image"] == image array
                Also returns a list of all the classes the images are from.
        """
        if not os.path.exists(self.images_root) :
            raise "Path" + self.images_root + "does not exist!"

        # Find all folders of the images
        with open(os.path.join(self.images_root, "classes.txt")) as f :
            allclasses = [line.split()[1].rstrip() for line in f]

        data_dict = {}

        # Iterate over all classes, convert num representation to chars
        progress = 0.0
        for dataclass in allclasses :
            dataclass_dir = os.path.join(self.images_root, "images", dataclass)
            for filename in os.listdir(dataclass_dir) :
                filepath = os.path.join(dataclass_dir, filename)

                image = self.load_image(filepath)

                data_dict[os.path.splitext(filename)[0]] = {"image": image}
            
            progress += 1.0
            print("Progress: ", round(progress/len(allclasses) * 100, 2), "%")

        return data_dict, allclasses

    def find_image_paths(self, data_dict) :
        """
        Finds path of all images for this dataset (defined in init) into memory
        :param The data dictionary where each key is the name of a image/caption file
        :return Returns a dictionary where the key is the filename of the image, just as the input
                with an additional key-value pair in the child dictionary accessible as
                data_dict["<filename>"]["imgpath"] = "/path/to/file.jpg"
        """
        if not os.path.exists(self.images_root) :
            raise "Path" + self.images_root + "does not exist!"

        # Find all folders of the images
        with open(os.path.join(self.images_root, "classes.txt")) as f :
            allclasses = [line.split()[1].rstrip() for line in f]

        data_dict = {}

        # Iterate over all classes, convert num representation to chars
        progress = 0.0
        for dataclass in allclasses :
            dataclass_dir = os.path.join(self.images_root, "images", dataclass)
            for filename in os.listdir(dataclass_dir) :
                filepath = os.path.join(dataclass_dir, filename)

                data_dict[os.path.splitext(filename)[0]] = {"imgpath": filepath}
            
            progress += 1.0
            print("Progress: ", round(progress/len(allclasses) * 100, 2), "%")

        return data_dict

    def load_image(self, filepath) :
        """
        Reads all the image in `filepath` into memory
        :param filepath The path of the file
        :return An array representation of the image
        """
        img = scipy.ndimage.imread(filepath)
        return img

    def merge_captions_images(self, caps_dict, imgs_dict) :
        """
        Merges captions and data dicts
        :param caps_dict The dictionary containing captions, as generated by read_all_captions above
        :param imgs_dict The dictionary containing images, as generated by read_all_images above
        """
        merged_dict = {}
        i = 0
        for filename in caps_dict.keys() :
            merged_dict[str(filename)] = caps_dict[str(filename)]
            for key in imgs_dict[filename].keys() :
                merged_dict[str(filename)][key] = imgs_dict[str(filename)][key]

        return merged_dict

