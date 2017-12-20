# Road Segmentation with Convolutional Neural Networks
##### Théo Deprelle, Valentin Nigolian, Niccolo Sacchi
This project is part of the course "Pattern Classification and Machine Learning" taught at EPFL in 2017. It presents a technique for segmenting satellite images by using convolutional neural networks (CNNs). The CNN we used implements classify each pixel either as road or background, but can be adapted to detect any kind of feature depending on the provided ground-truth.

Our team obtained an F1 score of 93.5% on the test set. The model and the techniques used are explained in `groupRoadSegmentationFault-submission.pdf`.

This repository includes both a script to train the model from scratch and a pre-trained model that can be used direcly to generate predictions. The original training and test sets can be downloaded from the Release section.

### Libraries
In order to correctly to run the scripts the following libraries must be installed:

- Keras 2.1.1
- Tensorlow 1.4.0
- others?

Keras is a  high-level neural networks API that uses TensorFlow as backend.

### Hardware
The model has been trained on a GPU acceleration, with the following setup:
VVV CHANGE VVV
- Windows 8.1 x64
- Intel Core i5-4460 @3.2 GHz
- NVIDIA GeForce GTX 960 (with 2 GB of RAM)
- 16 GB of system memory
- GPU Drivers: ForceWare 369.30
- Keras 1.1.2 with Theano 0.8.2 backend + CuDNN 5.1
- Theano flags: fastmath = True, optimizer = fast_run, floatX = float32

### How to reproduce the predictions
To generate the predictions, it is only necessary to run the script `run.py`. It won't train the model, instead it load the weights from `weights.h5`. Notice that this script expect to fin the test image in the folder `dataset/test_set_images`.

### How to train
To train the model from scratch run instead `train.py`. Depending on the available computational power and whether the model is being trained on the CPU or the GPU, the process may take some hours.

### Description of the files
vvv CHANGE vvv
- `asd.py`: asd description
- ...