from pycocotools.coco import COCO
import numpy as np


def read_keywords():
    """
    Reads keywords we want the captions to contain from "caption_keywords.txt"
    :return: A set of the keywords
    """
    with open("caption_keywords.txt", "r") as f:
        keywords = [line[:len(line) - 1] for line in f]

    return keywords


def read_coco():
    """
    Reads the training dataset from disk
    :return: COCO objects for annotations and capions
    """
    dataDir = 'coco'
    dataType = 'train2017'
    captionFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)
    coco_caps = COCO(captionFile)

    return coco, coco_caps


def filter_coco(keywords, coco, coco_caps):
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


keywords = set(read_keywords())
print("Keywords: ", keywords)
coco, coco_caps = read_coco()

filteredImgs = filter_coco(keywords, coco, coco_caps)
print("Number of images containing one or several keywords:", len(filteredImgs))
