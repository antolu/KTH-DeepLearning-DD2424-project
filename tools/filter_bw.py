import sys
sys.path.append("src")
sys.path.append("tools")
import os
import argparse
from PIL import Image

class MutableInt :
    def __init__(self) :
        self.i = 0

def filter(directory, bw, counter=None) : 
    for filename in os.listdir(directory) :
        path = os.path.join(directory, filename)
        if os.path.isdir(path) :
            filter(path, bw)
        elif filename.endswith(".jpg"):
            img = Image.open(path)
            if img.getbands()[0] == 'L':
                if counter is not None :
                    counter.i += 1
                    print(filename, counter.i)
                bw.add(filename)

parser = argparse.ArgumentParser()

parser.add_argument("--images_root", type=str, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

bw = set()

counter = MutableInt()
filter(args.images_root, bw, counter)

with open(args.output, "w") as f :
    for filename in bw :
        f.write(filename + "\n")