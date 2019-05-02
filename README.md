# DD2424-project
Project in DD2424 - Deep Learning: Text Adaptive Generative Adversarial Networks: Manipulating Images with Natural Language

# Dependency installation instructions

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