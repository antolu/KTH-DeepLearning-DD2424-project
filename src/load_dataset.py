from pycocotools.coco import COCO
import numpy as np
import os
import torchfile
from scipy.io import loadmat
from PIL import Image

import torch.utils.data as data
from torchvision import transforms

class ParseDatasets :
    """
    Class for reading and parsing CUB and oxford datasets
    The resulting data dictionaries have the following structure
    self.train_data = {<filename1>:{"imgpath":<path>, "captionpath":<path>, "img":<img>, "captions":<list of captions>} ... more filenames}
    """
    def __init__(self, images_root="", annotations_root="", dataset="cub", keywords_file="caption_keywords.txt", data_set="train", max_no_words=50, preprocess_caption=None, transform=None) :
        """
        Initialises this class for one dataset
        :param images_root The root directory of the images of this dataset
        :param captions_root The root directory of then captions of this dataset
        """
        super().__init__()
        self.images_root = images_root
        self.annotations_root = annotations_root
        self.dataset = dataset
        self.data_set = data_set

        self.max_no_words = max_no_words
        self.preprocess_caption = preprocess_caption
        
        if transform == None :
            self.transform = transforms.ToTensor()
        else :
            self.transform = transform

        self.keywords_file = keywords_file

        self.train = None
        self.val = None
        self.test = None

        self.data_is_parsed = False


    def parse_datasets(self) :
        if self.dataset != "coco" :
            if not os.path.exists(self.annotations_root) :
                raise "Path" + self.annotations_root + "does not exist!"
            
            if not os.path.exists(self.images_root) :
                raise "Path" + self.images_root + "does not exist!"

            self.read_data_paths()
        else :
            self.__read_coco_keywords()
            self.__read_coco_annotations(set=self.data_set)
            self.__filter_coco_annotations()

        self.data_is_parsed = True


    def __read_coco_keywords(self):
        """
        Reads keywords we want the captions to contain from "caption_keywords.txt"
        :return: A set of the keywords
        """
        with open(self.keywords_file, "r") as f:
            keywords = [line[:len(line) - 1] for line in f]

        self.caption_keywords = keywords


    def __read_coco_annotations(self, set='train'):
        """
        Reads the training dataset annotations from disk
        Saves coco objects in member variables for later use
        """
        if set == 'train':
            dataType = 'train2017'
        elif set == 'val':
            dataType = 'val2017'
        elif set == 'test' :
            dataType =  'test2017'
        else :
            raise "set option " + set + " was not recognized!"

        captionFile = '{}/captions_{}.json'.format(self.annotations_root, dataType)
        annFile = '{}/instances_{}.json'.format(self.annotations_root, dataType)

        self.coco = COCO(annFile)
        self.coco_caps = COCO(captionFile)


    def __filter_coco_annotations(self):
        """
        Finds the images in COCO dataset that contains one or several of the relevant keywords.
        Stores the resulting data dictionary as a member variable of this class called `data_dict`
        :param keywords: A set of keywords, and each image caption should contain at least one of the keywords
        :param coco: The COCO dataset annotations
        :param coco_caps: The COCO dataset captions
        Saves filtered data as member variable
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
                savedImg = {'image_id': imgID, 'id': img['id'], 'captions': [], 'imgpath':self.__get_imagepath(imgID)}

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

        if self.data_set == 'train':
            self.train = Dataset(filteredImgs, self.preprocess_caption, self.max_no_words, self.transform)
        elif self.data_set == 'val':
            self.val = Dataset(filteredImgs, self.preprocess_caption, self.max_no_words, self.transform)
        elif self.data_set == 'test' :
            self.test = Dataset(filteredImgs, self.preprocess_caption, self.max_no_words, self.transform)
        

    def __get_imagepath(self, image_id, dataset="train") :
        if dataset=='train' :
            dataType = 'train2017'
        elif dataset=='val' :
            dataType = 'val2017'
        elif dataset=='test' :
            dataType = 'test2017'
        else:
            raise "set option " + dataset + " was not recognized!"

        image_id = str(image_id)
        image_id = (12-len(image_id)) * "0" + image_id

        image_path = '{}/{}/{}.jpg'.format(self.images_root, dataType, image_id)

        return image_path

    # def __initiate_transforms(self) :


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

        with open(os.path.join(self.annotations_root, "allclasses.txt")) as f :
            allclasses = [line.rstrip() for line in f]

        # Find paths of the captions files
        for dataclass in allclasses :
            dataclass_dir = os.path.join(self.annotations_root, dataclass)
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

        self.train = Dataset(train_data, self.preprocess_caption, self.max_no_words, self.transform)
        self.val = Dataset(val_data, self.preprocess_caption, self.max_no_words, self.transform)
        self.test = Dataset(test_data, self.preprocess_caption, self.max_no_words, self.transform)

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
            caption = self.pc.num2char(caption_char_vec[:, i]).rstrip()
            captions.append(caption)

        return captions


    def load_image(self, filepath) :
        """
        Reads all the image in `filepath` into memory
        :param filepath The path of the file
        :return An array representation of the image
        """
        img = Image.open(filepath)
        return img


    def get_datasets(self) :
        if not self.data_is_parsed :
            self.parse_datasets()
            
        return self.train, self.val, self.test


class Dataset(data.Dataset, ParseDatasets) :
    """
    Base class for datasets inheriting the pytorch dataloader Dataset interface
    """

    def __init__(self, data_dict, preprocess_caption, max_no_words=50, transform=None) :
        self.data = data_dict
        self.keys = list(data_dict.keys())
        self.max_no_words = max_no_words

        self.pc = preprocess_caption
        self.transform = transform

    def __len__(self) :
        return len(self.data)

    def __getitem__(self, i) :
        imgId = self.keys[i]

        data = self.data[imgId]

        if "captions" not in data :
            captions = self.load_caption(data["captionpath"])
        else :
            captions = data["captions"]

        if "img" not in data : 
            img = self.load_image(data["imgpath"])
        else :
            img = data["img"]

        tnsr_img = self.transform(img)

        rand_caption = captions[np.random.choice(len(captions))]

        caption_vector, no_words = self.pc.string_to_vector(rand_caption, self.max_no_words)

        # ret = {"img":img, "tensor":tnsr_img, "caption_vector":caption_vector, "no_words":no_words, "caption":rand_caption}

        return tnsr_img, caption_vector, no_words