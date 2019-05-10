import torch
import argparse
from src.blocks import Generator, Discriminator

parser = argparse.ArgumentParser()

parser.add_argument("--gendisc", type=str, required=False)
parser.add_argument("--input", type=str, required=False)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

if args.gendisc is not None :
    if args.gendisc.lower() == "g" :
        g = Generator(50)
        model = g.state_dict()
    elif args.gendisc.lower() == "d" :
        d = Discriminator(50)
        model = d.state_dict()
else :
    model = torch.load(args.input)

model_keys = model.keys()

with open(args.output, "w") as f :
    for key in model_keys :
        f.write(str(key) + "\t\t\t\t\t " + str(model[key].shape) + "\n")
