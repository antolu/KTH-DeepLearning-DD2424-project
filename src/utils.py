import argparse


class Utils:
    """
    Utilities for the rest of the runnable code
    """
    def parse_args():
        """
        Parser base code for train and test scripts
        :return parser arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True,
                            help="The dataset we're running on")
        parser.add_argument("--images_root", type=str, required=True,
                            help="The root directory of the images")
        parser.add_argument("--annotations_root", type=str, required=True,
                            help="The root directory of the annotations/captions")
        parser.add_argument("--coco_set", type=str, required=False,
                            help="train/val/test for coco")
        parser.add_argument("--keywords_file", type=str, required=False,
                            help="Path to the keywords file required by COCO dataset filtering")
        parser.add_argument("--fasttext_model", type=str, required=True,
                            help="The path to the fasttext word embedding")
        parser.add_argument("--blacklist", type=str, required=False,
                            help="Path to the blacklist file")
        parser.add_argument("--pretrained_generator", type=str, required=False,
                            help="The path to the pretrained generator model")
        parser.add_argument("--pretrained_discriminator", type=str, required=False,
                            help="The path to the pretrained generator model")
        parser.add_argument("--mode", type=str, required=True, help="Run train or test")
        parser.add_argument("--no_epochs", type=int, required=False,
                            help="Number of training epochs to run")
        parser.add_argument("--pretrained_optimizer_discriminator",
                            type=str, required=False,
                            help="Load a pretrained optimizer for the discriminator "
                                 "with a certain learning rate and parameters")
        parser.add_argument("--pretrained_optimizer_generator",
                            type=str, required=False, help="Load a pretrained optimizer "
                            "for the generator with a certain learning rate and parameters")
        parser.add_argument("--batch_size",
                            type=int, required=False, help="Batch size for training")
        parser.add_argument("--momentum",
                            type=float, required=False, default=0.5,
                            help="Momentum term of optimizers")
        parser.add_argument("--lambda_1",
                            type=float, required=False, default=10.0,
                            help="Weight of conditional probabilities on loss function")
        parser.add_argument("--save_models_frequency",
                            type=int, required=False, default=50,
                            help="Save the models every this number of epochs")
        parser.add_argument("--learning_rate", required=False, default=0.0002,
                            type=float, help="Initial learning_rate for the Adam optimzer")
        return parser.parse_args()

    def read_blacklist(path):
        blacklist = set()
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                blacklist.add(line.rstrip())
        return blacklist
