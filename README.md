# Supplemental Code for ICML Submission

This directory contains files associated with the ICML submission "Memory-Optimal Direct Convolutions for Maximizing Classification Accuracy in Embedded Applications".

## Dependencies
The main dependencies are shown below.
1. Python 2.7
1. TensorFlow 1.13
1. Keras 2.2
1. pyserial

## Main Contents
The contents are best understood as being split by different segments of the (relatively linear) work flow. Thus, understanding the files in the same order may make it easier to understand.
1. `MNIST - Data Generator.ipynb` - generates the augmented dataset.
1. `MNIST - Sweeps.ipynb` - finds the optimal architecture through a brute-force sweep of hyperparameters.
1. `MNIST - Training.ipynb` - runs training on the network identified in `Sweeps` using data generated in `Data Generator`.
1. `MNIST - Arduino.ipynb` - interfaces to and verifies accuracy of the Arduino. It uses the final network parameters from `Training`.
1. `MNIST - Plots.ipynb` - generates plots for the paper, including interesting relationships between network size and accuracy.
1. `Arduino/cnn.cpp` - The Arduino code for reading in a full neural network specification and a stream of images and outputting the image class, all via serial communications.

In addition, see the helper files:
1. `network_parameterization.py` - establishes an easy way of defining networks so they can be analyzed for memory usage and also built very easily.
1. `quantization_layers.py` - defines custom Keras layers for easy quantization during training.

Some results have also been included for convenience:
1. `sweeps` - contains sweeps saved during the sweep process.
1. `models` - contains models saved during the training process.

However, data files have not been included, because they are too large.
1. `augmented_x_200k_v2.npy`
1. `augmented_y_200k_v2.npy`

## Arduino Workflow
The Arduino code uses [Arduino-Makefile](https://github.com/sudar/Arduino-Makefile). To run it:
1. Connect your Arduino (we used an Arduino nano with the ATmega328P chip).
1. Navigate to `<root>/Arduino`.
1. Run `make`, then `make upload`.
1. Open `MNIST - Arduino.ipynb` for an example of Arduino communications to Python.

## Credits
[Albert Gural](https://github.com/agural/memory-optimal-direct-convolutions)

