import torch
from collections import OrderedDict
# from src.blocks import *

class ParseMapping :
    def __init__(self, mapping_file, their_model_file=None, our_model_file=None) :
        self.mapping_file = mapping_file
        self.their_model_file = their_model_file
        self.our_model_file = our_model_file
    
    def parse(self) :
        
        their_model = torch.load(self.their_model_file)
        our_model = torch.load(self.our_model_file)

        # Read mapping
        with open(self.mapping_file, "r") as f :
            our_to_their = {}
            their_to_our = {}
            for line in f :
                line = line.rstrip()
                if (line == "") or (line.startswith("#")) :
                    continue
                    
                contents = line.split()

                our = contents[0]
                their = contents[1]

                if their == "None" :
                    continue

                our_to_their[our] = their
                their_to_our[their] = our

        mapped_parameters = OrderedDict()

        n = 0
        for layer in our_model.keys() :
            if layer not in our_to_their :
                print("Layer {} not in our model".format(layer))
                mapped_parameters[layer] = our_model[layer]
                n += 1
                continue

            # print(layer, our_to_their[layer])
            mapped_parameters[layer] = their_model[our_to_their[layer]]

        print("Number of unmatched parameters: ", n)

        self.mapped_parameters = mapped_parameters

    def write_mapping(self, filepath="mapped_model.pth") :
        if self.mapped_parameters is None :
            raise "Parameters not yet mapped. Run .parse() first."

        torch.save(self.mapped_parameters, filepath)

    def get_state_dict(self) :
        if self.mapped_parameters is None :
            raise "Parameters not yet mapped. Run .parse() first."

        return self.mapped_parameters

# Usage example 
pm = ParseMapping("mapping.txt", their_model_file="models/flowers_G.pth", our_model_file="our_model.pth")

pm.parse()
pm.write_mapping()

# mapping = pm.mapped_parameters

# generator = Generator(50)

# generator.load_state_dict(mapping)
