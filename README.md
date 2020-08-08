# YorkU_2D_Detection_Benchmarking
This is the Repo for benchmarking some of the representative one-stage 2D object detection frameworks. These chosen frameworks may not rank the top on COCO leader now, but they are recognized as key innovation during the past few years.

## Introduction

By using this Repo, you will be able to train and evaluate object detection frameworks on Pascal VOC, COCO, and UA-DETRAC dataset. Several modern detection frameworks are available for training, these will include:
- SSD
- RetinaNet
- RefineDet
- ATSS
- NETNet

This Repo is self-stand without the need of compiling any other sources(cocoapi is required), and it offers a unified pipeline to train and evaluate these models.


## Experiments

All the models are trained and evaluated on a single RTX2060 GPU with 11GB RAM. The batch size varies from 2-16 due to the varying size of models and training processes. There are 3 datasets that are currently available for using:
- Pascal VOC: VOC12 + VOC07 training set for training, and VOC07 test set for testing
- COCO: COCO2017 training set for training and COCO2017 validation set for testing(test set is not used for evaluation)
- UA-DETRAC: training set for training, and test set for testing
