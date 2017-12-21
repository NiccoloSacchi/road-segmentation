# Road Segmentation with Convolutional Neural Networks
##### Théo Deprelle, Valentin Nigolian, Niccolo Sacchi
This project is part of the course "Pattern Classification and Machine Learning" taught at EPFL in 2017. It presents a technique for segmenting satellite images by using convolutional neural networks (CNNs). The CNN we used implements classify each pixel either as road or background, but can be adapted to detect any kind of feature depending on the provided ground-truth.

Our team obtained an F1 score of 93.38% on the test set. The model and the techniques used are explained in `groupRoadSegmentationFault-submission.pdf`.

The model and the dataset that we used can be found at : 
https://drive.switch.ch/index.php/s/it5ylw3afG8Lg2R

Before anything you should change the paths of the images in the run.py and train.py files.


### Libraries
In order to correctly to run the scripts the following libraries must be installed:

- torch 0.2.0_4
- numpy,scipy

### Hardware
The model has been trained on a GPU acceleration, with the following setup:

- NVIDIA Titan X (with 12 GB of RAM)
- python 3.6.4
- Cudnn 8.0

### How to reproduce the predictions
To generate the predictions, it is only necessary to run the script `run.py`. It won't train the model, instead it load the weights.

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