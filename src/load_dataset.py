from pycocotools.coco import COCO
import numpy as np
import scipy.ndimage

class parsecoco:
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


# class READCUB :


