import torch

# model = torch.load("models/flowers_G.pth")
model = torch.load("new_model.pth")

model_keys = model.keys()

with open("new_layers_output.txt", "w") as f :
    for key in model_keys :
        f.write(str(key) + "\t\t\t\t\t " + str(model[key].shape) + "\n")
