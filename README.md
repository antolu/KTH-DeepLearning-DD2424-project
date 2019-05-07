# DD2424-project
Project in DD2424 - Deep Learning: Text Adaptive Generative Adversarial Networks: Manipulating Images with Natural Language

## Dependency installation instructions

The following third party packages need to be installed to be able to run the code:

* [fastText](https://github.com/facebookresearch/fastText) for transforming captions to vector space representations.  
(Clone repo and run `pip install .`. Do NOT use download with pip or conda).
* [Pytorch](https://pytorch.org/) v1.0 or later
* [Torchvision](https://github.com/pytorch/vision) for transforming image tensors.
* [torchfile](https://github.com/bshillingford/python-torchfile) for reading the captions from .t7 files.
* [Pillow](https://pillow.readthedocs.io/en/4.2.x/) for reading image files in PyTorch compatible formats. 
* [OpenCV](https://github.com/opencv/opencv) for manipulating images.
* [COCOPAPI](https://github.com/cocodataset/cocoapi) for reading the COCO dataset annotations. See below for installation instructions.

### COCO dataset

To load the COCO dataset, the `pycocotools` API must be installed, and the annotations and images put into a folder called `coco` in the root of the repository. The repo should then have the following structure
```
|--coco
    |-- annotations
        |-- captions_train2017.json
        |-- ...
    |-- train2017
        |-- 0000000000000.jpg
        |-- ...
|--src
    |-- load_dataset.py
```

an alternative way of installing `pycocotools` is to use Anaconda:
```
conda install -c hcc pycocotools
```
or doing it manually
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
sudo make install
python setup.py install
```

### Datasets
The code is written to support the COCO (Common Objects in Context), CUB-200 birds, and Oxford-102 flowers datasets.

Links to the datasets are the following:

* COCO: [download images and annotations here](http://cocodataset.org/#download).
* Oxford: [images and data splits (setid.mat)](http://www.robots.ox.ac.uk/~vgg/data/flowers/102) and [captions]https://drive.google.com/file/d/0B0ywwgffWnLLMl9uOU91MV80cVU/view?usp=sharing)
* CUB [images](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [captions](https://drive.google.com/file/d/0B0ywwgffWnLLLUc2WHYzM0Q2eWc/view?usp=sharing)

The captions for the CUB and Oxford datasets are from [this repository](https://github.com/reedscot/icml2016)

At this time the code does not support adaptive paths for the datasets.