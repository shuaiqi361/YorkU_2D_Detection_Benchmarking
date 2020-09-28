# YorkU_2D_Detection_Benchmarking
This is the Repository for benchmarking some of the representative one-stage 2D object detection frameworks. These chosen frameworks may not rank the top on COCO leaderboard now, but they were recognized as key innovations in their time. In this Repo, you will find the code to train and evaluate these models, you will also be able to download the checkpoints trained by us on benchmark datasets.

## Introduction

By using this Repo, you will be able to train and evaluate object detection frameworks on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [MS COCO](https://cocodataset.org/#home), and [UA-DETRAC](http://detrac-db.rit.albany.edu/) dataset. Several modern detection frameworks are available for training, these will include:
- [SSD](https://arxiv.org/abs/1512.02325)
- [RetinaNet](https://arxiv.org/abs/1708.02002)
- [RefineDet](https://arxiv.org/abs/1711.06897)
- [ATSS](https://arxiv.org/abs/1912.02424)
- [NETNet](https://arxiv.org/abs/2001.06690)

This Repo is self-standing without the need of compiling any other sources, and it offers a unified pipeline to train and evaluate these models. In order to have a fair comparison among these baselines, the version in this Repo may be different from the original published paper, for example, all input images are resized to have the same size 512-by-512, all the output bounding boxes are percentage regarding the input image, only the VGG16 and ResNet50 are available for backbones, multi-GPU training is currently not supported, and etc. Therefore, you may find that the final mAP doesn't match the published results.

In addition, the trained models are saved and released for evaluation. Given the settings, the current best model on COCO and DETRAC dataset is RefineDet, you can download the saved checkpoints of this model in the last section of this Repo.


## Environment Setup
The code is written in Python 3.6, the Anaconda for setting up the environments is recommended. Some sample commands for installation are available:

Create Python conda env:
```
conda create -n py36 python=3.6 anaconda
```
Install Pytorch:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
Install opencv and other packages:
```
pip install opencv-python
pip install opencv-contrib-python
conda install yaml
conda install easydict
pip install pycocotools
```
Clone the Repo:
```
git clone https://github.com/shuaiqi361/YorkU_2D_Detection_Benchmarking.git
```
Activate the conda env:
```
source activate py36
```


## Experiments

All the models could be trained and evaluated on a single RTX 2080ti GPU with 11GB RAM. The batch size varies from 6-16 due to the varying size of models and training processes. There are 3 datasets that are currently available for using with the provided code:
- Pascal VOC: VOC12 + VOC07 training set for training, and VOC07 test set for testing
- COCO: COCO2017 training set for training and COCO2017 validation set for testing(test set is not used for offline evaluation purpoose)
- UA-DETRAC: training set for training, and test set for testing

Again, this train-test split servers as a baseline which is designed to compare different frameworks rather than duplicating the published results, for example, the current protocol for training on COCO is to use trainval35k for training, and evaluate on the withheld test set, which is different from here.

### Dataset
In the dataset folder, parsing files are available to create proper forms of input for training, you can download the dataset, and specify the paths of the dataset in your local computer, create the output data folder, then run the parsing scripts, e.g. [coco\_data\_parsing.py](https://github.com/shuaiqi361/YorkU_2D_Detection_Benchmarking/blob/master/dataset/coco_data_parsing.py).

### Train a model
If you want to train a model on COCO dataset, simply navigate to the experiment folder, and choose a model. In each of the model folder, there is a [train.sh] file, you can run the file to start training. Before training, make sure that the paths in this file, and the settings in config.yaml files are correct. Training logs will be saved under [logs] folder, and the saved model will be saved under [snapshots] folder.

For example, if you want to train the RefineDet model on COCO dataset, specify paths of you [local folder](https://github.com/shuaiqi361/YorkU_2D_Detection_Benchmarking/tree/master/experiment/RefineDet_exp_001) in [RefineDet\_exp\_001/config.yaml] and [RefineDet\_exp\_001/train.sh] and run [RefineDet\_exp\_001/train.sh] by:
```
sh train.sh
```

### Evaluate a model checkpoint
If you want to evaluate a saved checkpoint on COCO dataset, run the [eval.sh](https://github.com/shuaiqi361/YorkU_2D_Detection_Benchmarking/blob/master/experiment/SSD512_exp_001/eval.sh) file under each experiment folder(create one if missing), the trained checkpoint for RefineDet is provided [here](https://drive.google.com/file/d/1o-O50gHJ-FVGbugzRrAXGiRos60Ndexn/view?usp=sharing). Do
```
sh eval.sh
```

If you want to evaluate a model and show the detection results, you can run the [detect\_script/detect_bbox.py](https://github.com/shuaiqi361/YorkU_2D_Detection_Benchmarking/blob/master/detect_script/detect_bbox.py), this file can accept single image, a folder of images, and video as input, specify the paths of saved checkpoints in the main() function, and the output images and text files containing bboxes and labels will be saved under detect_results. Specify pathes and then,
```
python detect_bbox.py --image/--video/--folder
```

If you try to evaluate on DETRAC traffic dataset and generate the output format for online evaluation on their website, you can run [detect\_script/detect_detrac.py](https://github.com/shuaiqi361/YorkU_2D_Detection_Benchmarking/blob/master/detect_script/detect_detrac.py). The checkpoint is available [here](https://drive.google.com/file/d/1JleKAvcMtsJPT1hopABEgWg8jV9NMEwC/view?usp=sharing).
```
python detect_detrac.py
```
If you want to evaluate the output files locally, you will need to download their evaluation code [online](http://detrac-db.rit.albany.edu/download)
