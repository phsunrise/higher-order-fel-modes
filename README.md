# Detecting Higher-Order Modes in FEL Pulses
This is a deep neural network project to detect and characterize 
higher-order modes in free-electron laser (FEL) images. Our goal
is to understand the performance of the FEL and its relation to 
various electron bunch parameters.

This repo consists of two parts: 1) supervised learning, where
we train and test on images that are labelled, and 2) unsupervised 
learning, where we let the neural network to automatically cluster
the images.

The keras library is required to use this repo.

## Supervised learning
The supervised learning codes include the following:

* [GUI_label.py](GUI_label.py), a python GUI for users to label the images.

* [combine_runs.py](combine_runs.py), a code to combine the images (from different
experiment runs), randomly select a given number of images with 
a certain label, and shuffle the final dataset. This is a necessary
step because in the original dataset, 0th-order modes (label=0)
greatly outnumbers higher-order modes (label=1).
This code also includes cropping and resizing of the images.

* [residue_networks.py](residue_networks.py), a residual deep neural network.

* [simple.py](simple.py), a simpler convolutional neural network.

## Unsupervised learning
We will use an autoencoder network for unsupervised learning.
To be continued...
