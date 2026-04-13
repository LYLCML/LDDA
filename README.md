# Learning Discriminative Directional Anchors for Open-Set Recognition (LDDA)

Official PyTorch implementation of **Learning Discriminative Directional Anchors for Open-Set Recognition (LDDA)**.

---

## Overview

LDDA is a prototype-based open-set recognition framework that introduces **anchor-direction pairs** to model class-specific directional structures in the embedding space.  
Unlike conventional distance-based prototype learning methods, LDDA determines class membership not only by the distance to an anchor point, but also by whether the sample is aligned with the corresponding class direction.

This repository provides the implementation for:

- **LDDA**
- **R-LDDA**

---

## Requirements
The current implementation requires the following packages:
- Python >= 3.6
- PyTorch >= 1.4
- torchvision >= 0.5
- CUDA >= 10.1
- scikit-learn >= 0.22

### Datasets
For Tiny-ImageNet, please download the following datasets to ./data/tiny_imagenet.

---

## Training & Evaluation
- Running the code: Simply run the main file.
- Dataset settings：datasetName='cifar10'.   Change the dataset here.  
- Dataset partitioning ['a','b','c','d','e']: for x in ['a']:
