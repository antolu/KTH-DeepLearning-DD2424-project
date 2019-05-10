################################################################
# Maps the pretrained parameters to our structure and saves it
################################################################

import sys
sys.path.append("src")
sys.path.append("tools")
from model_mapping import ParseMapping
from blocks import Generator, Discriminator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--pretrained", type=str, required=True)
parser.add_argument("--mapping", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--gendisc", type=str, required=True)

args = parser.parse_args()

if args.gendisc.lower() == "g" :
    g = Generator(50)
    our_model = g.state_dict()
elif args.gendisc.lower() == "d" :
    d = Discriminator(50)
    our_model = d.state_dict()

pm = ParseMapping(args.mapping, their_model_file=args.pretrained, our_model=our_model)

pm.parse()
pm.write_mapping(args.output)
