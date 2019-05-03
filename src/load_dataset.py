from pycocotools.coco import COCO
import numpy as np
import scipy.ndimage
import os
import torchfile
from scipy.io import loadmat

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

class CaptionTools:
    def __init__(self) :
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

    def num2char(self, nums) :
        """
        Converts a vector in numerical for to character vector (for CUB and Oxford)
        """
        char = ""
        for num in nums :
            char += self.alphabet[num-1]

        return char

class ParseCoco:
    """
    Reads and parses captions and images of the COCO dataset
    The finished data dictionaries have the following form
    self.data_dict = {<image id>:{"image_id":<image id>, "id":<real id>, "captions":<list of captions>}}
    This class does NOT load images into the data dictionary due to the size of the dataset. Instead
    call the load_image method with the image id to load individual files at runtime. 
    """
    def __init__(self, data_root="datasets/coco", keywords_file="caption_keywords.txt") :
        self.data_root = data_root
        self.keywords_file = keywords_file

    def read_keywords(self):
        """
        Reads keywords we want the captions to contain from "caption_keywords.txt"
        :return: A set of the keywords
        """
        with open(self.keywords_file, "r") as f:
            keywords = [line[:len(line) - 1] for line in f]

        self.caption_keywords = keywords


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

        self.coco = COCO(annFile)
        self.coco_caps = COCO(captionFile)


    def filter_coco(self):
        """
        Finds the images in COCO dataset that contains one or several of the relevant keywords.
        Stores the resulting data dictionary as a member variable of this class called `data_dict`
        :param keywords: A set of keywords, and each image caption should contain at least one of the keywords
        :param coco: The COCO dataset annotations
        :param coco_caps: The COCO dataset captions
        :return: A dictionary with key corresponding to a an image ID, and the value is another dictionary where the "captions" key yields a list of the image captions
        """
        imgIds = self.coco.imgs.keys()

        annIds = self.coco_caps.getAnnIds(imgIds=imgIds)  # [21616, 53212]
        anns = self.coco_caps.loadAnns(
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
                if len(set(caption.split()).intersection(self.caption_keywords)) > 0:
                    filteredImgs[imgID] = sortedImgs[imgID]
                    break

        self.data_dict = filteredImgs

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
    The resulting data dictionaries have the following structure
    self.train_data = {<filename1>:{"imgpath":<path>, "captionpath":<path>, "img":<img>, "captions":<list of captions>} ... more filenames}
    """
    def __init__(self, images_root, captions_root, dataset="cub") :
        """
        Initialises this class for one dataset
        :param images_root The root directory of the images of this dataset
        :param captions_root The root directory of then captions of this dataset
        """
        super().__init__()
        self.images_root = images_root
        self.captions_root = captions_root
        self.dataset = dataset

        if not os.path.exists(self.captions_root) :
            raise "Path" + self.captions_root + "does not exist!"
        
        if not os.path.exists(self.images_root) :
            raise "Path" + self.images_root + "does not exist!"

        self.read_data_paths()


    def read_data_paths(self) :
        """
        Finds path of all images and captions for this dataset (defined in init) into memory
        Saves the dictionaries `train_data` and `test_data` as member variables
        """

        # Find all folders of the images
        if self.dataset=="cub" :
            with open(os.path.join(self.images_root, "classes.txt")) as f :
                allclasses = [line.split()[1].rstrip() for line in f]

            # Be able to convert image id to image name
            imgId_to_filename = {}
            filename_to_imgId = {}
            with open(os.path.join(self.images_root, "images.txt")) as f :
                for line in f:
                    line = line.rstrip()
                    imgId, path = line.split()
                    filename = os.path.splitext(os.path.split(path)[1])[0]
                    imgId_to_filename[imgId] = filename
                    filename_to_imgId[filename] = imgId

            # Find data splits
            data_split = {}
            with open(os.path.join(self.images_root, "train_test_split.txt")) as f :
                for line in f :
                    imgId, traintest = line.split()
                    data_split[imgId] = traintest

        elif self.dataset=="oxford" :
            data_split = {}
            imgId_to_filename = {}
            filename_to_imgId = {}

            split = loadmat(os.path.join(self.images_root, "setid.mat"))

            train_ids = split["trnid"][0]
            val_ids = split["valid"][0]
            test_ids = split["tstid"][0]

            for imgId in train_ids :
                imgId = str(imgId)

                filename = "image_" + ((5-len(imgId)) * "0") + imgId
                imgId_to_filename[str(imgId)] = filename
                filename_to_imgId[filename] = str(imgId)

                data_split[str(imgId)] = "1"
            
            for imgId in val_ids :
                imgId = str(imgId)

                filename = "image_" + ((5-len(imgId)) * "0") + imgId
                imgId_to_filename[imgId] = filename
                filename_to_imgId[filename] = imgId

                data_split[imgId] = "2"

            for imgId in test_ids :
                imgId = str(imgId)

                filename = "image_" + ((5-len(imgId)) * "0") + imgId
                imgId_to_filename[imgId] = filename
                filename_to_imgId[filename] = imgId

                data_split[imgId] = "0"

        train_data = {}
        val_data = {}
        test_data = {}

        # Find paths of the image files
        if self.dataset == "cub" :
            for dataclass in allclasses :
                dataclass_dir = os.path.join(self.images_root, "images", dataclass)
                for filename in os.listdir(dataclass_dir) :
                    filepath = os.path.join(dataclass_dir, filename)

                    filename = os.path.splitext(filename)[0]
                    imgId = filename_to_imgId[filename]
                    if data_split[imgId] == "1" :
                        train_data[filename] = {"imgpath": filepath}
                    elif data_split[imgId] == "2" :
                        val_data[filename] = {"imgpath": filepath}
                    elif data_split[imgId] == "0" :
                        test_data[filename] = {"imgpath": filepath}
        elif self.dataset == "oxford" :
            for filename in os.listdir(self.images_root) :
                filepath = os.path.join(self.images_root, filename)

                if filename == "setid.mat":
                    continue

                filename = os.path.splitext(filename)[0]
                
                imgId = filename_to_imgId[filename]
                if data_split[imgId] == "1" :
                    train_data[filename] = {"imgpath": filepath}
                elif data_split[imgId] == "2" :
                    val_data[filename] = {"imgpath": filepath}
                elif data_split[imgId] == "0" :
                    test_data[filename] = {"imgpath": filepath}

        with open(os.path.join(self.captions_root, "allclasses.txt")) as f :
            allclasses = [line.rstrip() for line in f]

        # Find paths of the captions files
        for dataclass in allclasses :
            dataclass_dir = os.path.join(self.captions_root, dataclass)
            for filename in os.listdir(dataclass_dir) :
                filepath = os.path.join(dataclass_dir, filename)

                filename = os.path.splitext(filename)[0]
                imgId = filename_to_imgId[filename]
                if data_split[imgId] == "1":
                    train_data[filename]["captionpath"] = filepath
                elif data_split[imgId] == "2":
                    val_data[filename]["captionpath"] = filepath
                elif data_split[imgId] == "0":
                    test_data[filename]["captionpath"] = filepath

        self.data_split = data_split
        self.imgId_to_filename = imgId_to_filename
        self.filename_to_imgId = filename_to_imgId
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def read_all_captions(self) :
        """
        Reads all captions for this dataset (defined in init) into memory
        Saves the captions into the member variables `train_data` and `test_data`
        """

        progress = 0.0
        for filename in self.train_data.keys() :
            self.train_data[filename]["captions"] = self.load_caption(self.train_data[filename]["captionpath"])
            progress += 1.0
            if progress % 100 == 0:
                print("Training caption loading progress: ", round(progress/len(self.train_data) * 100, 2), "%")

        progress = 0.0
        for filename in self.val_data.keys() :
            self.val_data[filename]["captions"] = self.load_caption(self.val_data[filename]["captionpath"])
            progress += 1.0
            if progress % 100 == 0:
                print("Training caption loading progress: ", round(progress/len(self.val_data) * 100, 2), "%")

        progress = 0.0
        for filename in self.test_data.keys() :
            self.test_data[filename]["captions"] = self.load_caption(self.test_data[filename]["captionpath"])
            progress += 1.0
            if progress % 100 == 0:
                print("Test caption loading progress: ", round(progress/len(self.test_data) * 100, 2), "%")


    def read_all_images(self) :
        """
        Reads all images for this dataset (defined in init) into memory
        Saves the images into the member variables `train_data` and `test_data`
        """
        progress = 0.0
        for filename in self.train_data.keys() :
            self.train_data[filename]["img"] = self.load_image(self.train_data[filename]["imgpath"])
            progress += 1.0
            if progress % 100 == 0:
                print("Training image loading progress: ", round(progress/len(self.train_data) * 100, 2), "%")

        progress = 0.0
        for filename in self.val_data.keys() :
            self.val_data[filename]["img"] = self.load_image(self.val_data[filename]["imgpath"])
            progress += 1.0
            if progress % 100 == 0:
                print("Validation image loading progress: ", round(progress/len(self.val_data) * 100, 2), "%")

        progress = 0.0
        for filename in self.test_data.keys() :
            self.test_data[filename]["img"] = self.load_image(self.test_data[filename]["imgpath"])
            progress += 1.0
            if progress % 100 == 0:
                print("Test image loading progress: ", round(progress/len(self.test_data) * 100, 2), "%")


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


    def load_image(self, filepath) :
        """
        Reads all the image in `filepath` into memory
        :param filepath The path of the file
        :return An array representation of the image
        """
        img = scipy.ndimage.imread(filepath)
        return img
