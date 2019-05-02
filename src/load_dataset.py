from pycocotools.coco import COCO
import numpy as np
import scipy.ndimage

class READCOCO:
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
        dataDir = 'coco'
        if set == 'train':
            dataType = 'train2017'
        elif set == 'val':
            dataType = 'val2017'
        else :
            raise "set option " + set + " was not recognized!"

        captionFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

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

        for img in anns:
            imgID = img['image_id']

            if imgID in sortedImgs:
                savedImg = sortedImgs[imgID]
            else:
                savedImg = {'image_id': imgID, 'id': img['id'], 'captions': []}

            caption = img['caption']
            savedImg['captions'].append(caption)
            sortedImgs[imgID] = savedImg

        for imgID in sortedImgs:

            for caption in sortedImgs[imgID]['captions']:
                if len(set(caption.split()).intersection(keywords)) > 0:
                    filteredImgs[imgID] = sortedImgs[imgID]
                    break

        return filteredImgs

    def load_image(self, image_id, set='train'):
        dataDir = 'coco'
        if set=='train' :
            dataType = 'train2017'
        elif set=='val' :
            dataType = 'val2017'
        else:
            raise "set option " + set + " was not recognized!"

        image_id = str(image_id)
        image_id += (13-len(image_id)) * "0" + ".jpg"

        image_file = '{}/{}/{}.json'.format(dataDir, dataType, image_id)

        image_array = scipy.ndimage.imread(image_file)

        return image_array


# class READCUB :

c = READCOCO()

keywords = set(c.read_keywords())
print("Keywords: ", keywords)
coco, coco_caps = c.read_coco()

filteredImgs = c.filter_coco(keywords, coco, coco_caps)
print("Number of images containing one or several keywords:", len(filteredImgs))
