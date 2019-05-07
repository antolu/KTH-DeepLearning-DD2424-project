import argparse

def parse_args() :
    """
    Parser base code for train and test scripts
    :return parser arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="The dataset we're running on")
    parser.add_argument("--images_root", type=str, required=True, help="The root directory of the images")
    parser.add_argument("--annotations_root", type=str, required=True, help="The root directory of the annotations/captions")

    parser.add_argument("--coco_set", type=str, required=False, help="train/val/test for coco")
    parser.add_argument("--keywords-file", type=str, required=False, help="Path to the keywords file required by COCO dataset filtering")

    parser.add_argument("--fasttext_model", type=str, required=True, help="The path to the fasttext word embedding")
    parser.add_argument("--pretrained_model", type=str, required=True, help="The path to the pretrained generator model")

    parser.add_argument("--max_no_words", type=int, required=True, help="Maximum number of words for the model")

    return parser.parse_args()