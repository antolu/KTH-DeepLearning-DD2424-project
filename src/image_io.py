import matplotlib.pyplot as plt
import numpy as np
import torch

def disp_tensor(tnsr) :
    """
    Display a PyTorch tensor as an image
    """
    img = tnsr.permute(1, 2, 0).numpy()

    plt.axis('off')
    plt.imshow(img)
    plt.show()
    plt.clf()

def disp_sidebyside(images, caption=None) :
    """
    Display a numpy image side-by-side with a PyTorch image
    :param images A list of images, either in numpy or PyTorch format
    :param caption The title/caption of the image
    """
    parsed = []
    for image in images :
        if type(image) is torch.Tensor :
            parsed.append(image.detach().permute(1, 2, 0).numpy())
        else :
            parsed.append(image)

    f = plt.figure()

    for i in range(len(parsed)) :
        f.add_subplot(1,len(parsed), i+1)
        plt.axis('off')
        plt.imshow(parsed[i])

    if caption is not None :
        plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)

    plt.show(block=True)
    f.clear()
    plt.close(f)

