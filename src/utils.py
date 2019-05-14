import argparse

class Utils :
    """
    Utilities for the rest of the runnable code
    """
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
        parser.add_argument("--keywords_file", type=str, required=False, help="Path to the keywords file required by COCO dataset filtering")

        parser.add_argument("--fasttext_model", type=str, required=True, help="The path to the fasttext word embedding")
        parser.add_argument("--blacklist", type=str, required=False, help="Path to the blacklist file")
        parser.add_argument("--pretrained_generator", type=str, required=False, help="The path to the pretrained generator model")
        parser.add_argument("--pretrained_discriminator", type=str, required=False, help="The path to the pretrained generator model")

        parser.add_argument("--max_no_words", type=int, required=True, help="Maximum number of words for the model")

        parser.add_argument("--runtype", type=str, required=True, help="Run train or test")
        parser.add_argument("--no_epochs", type=int, required=False, help="Number of training epochs to run")


        return parser.parse_args()

    def read_blacklist(path) :

        blacklist = set()

        with open(path, "r") as f :
            for line in f :
                if line.startswith("#") :
                    continue
                blacklist.add(line.rstrip())

        return blacklist
