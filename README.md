﻿# Road Segmentation with Convolutional Neural Networks
##### Niccolò Sacchi, Théo Deprelle, Valentin Nigolian
This project is part of the course "Pattern Classification and Machine Learning" taught at EPFL in 2017. It presents a technique for segmenting satellite images by using convolutional neural networks (CNNs).

Our team obtained an F1 score of 93.85% on Kaggle. The model and the techniques used are explained in `groupRoadSegmentationFault-submission.pdf`.

Due to the size of the network, both the model and the dataset have been uploaded [here](https://drive.switch.ch/index.php/s/it5ylw3afG8Lg2R).

Before anything you should change the paths of the images in the `run.py` and `train.py` files.

### Libraries
In order to correctly to run the scripts the following libraries must be installed:

- torch 0.2.0_4
- numpy
- scipy
- python 3.6.4
- Cudnn 8.0

### Hardware
The model has been trained on a GPU acceleration, with the following setup:

- NVIDIA Titan X (with 12 GB of RAM)

### How to reproduce the predictions
To generate the predictions, it is only necessary to run the script `run.py`. It won't train the model, instead it will just load the weights and produce a file with the predictions.

### How to train
To train the model from scratch run instead `train.py`. With our configuration the training took 3 hours.

### Description of the files
- `__init__.py`       : init the network
- `network.py`        : definition of the network
- `setup.py`          : setting the parameter for the training
- `summary.py`        : used to save the outputs 
- `trainer.py`        : train the network
- `transformation.py` : preprocessing and data augmentation
- `utils.py`          : utility functions

### Report
For a detailed description of the implemented solution please refer to the [Report](https://github.com/NiccoloSacchi/road-segmentation/blob/master/report/groupRoadSegmentationFault-submission.pdf).
