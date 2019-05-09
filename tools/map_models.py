################################################################
# Maps the pretrained parameters to our structure and saves it
################################################################

from src.model_mapping import ParseMapping

models=[
    {
        "their_model":"models/flowers_G.pth",
        "output_name":"oxford_G.pth"
    },
    {
        "their_model":"models/birds_G.pth",
        "output_name":"cub_G.pth"
    }
]

for model in models :
    pm = ParseMapping("mapping.txt", their_model_file=model["their_model"], our_model_file="our_model.pth")

    pm.parse()
    pm.write_mapping(model["output_name"])
