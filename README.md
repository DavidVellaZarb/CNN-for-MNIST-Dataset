# CNN-for-MNIST-Dataset

This app creates and trains a convolutional neural network for classifying handwritten digits (using the famous MNIST dataset), 
with ~99% accuracy on the training set and ~95% accuracy on the test set.

It displays plots of various learning metrics, and saves the resulting neural network.

## Running Instructions

Prerequisites:
- Python 3
- Tensorflow
- Keras

Tensorflow and Keras can be installed using `pip install`.  To run the programme, simply cd into the directory and run 
`python CNN-for-MNIST-Dataset.py`.

Please note that training the neural network takes a long time (~25 minutes on my computer).  In real life, you would only need to 
do this once.
