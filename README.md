# Self-Driving Car Engineer Nanodegree
# Deep Learning
# Project 3: Behavior Cloning

## Overview
This project aims to use CNN to training the car to drive itself in a simulation environment. The video of the final result in the following link: [YouTube](https://www.youtube.com/watch?v=JLVWC8iJfss)

## Dependencies
This project requires Python 3.5 and the following Python libraries installed:

- numpy
- opencv
- pandas
- flask-socketio
- eventlet
- pillow
- keras
- tensorflow
- h5py

## Files:
- model.ipynb: the ipython notebook generates the CNN model . The detailed is described in the next section. 
- drive.py: the python script predicts the steering angle from the image in the simulation in real time and send the steering angle back to the simulation.
- model.joson: the final model
- model.h5: the weight of the final model

## Detail Description:
### Preprocess
1. select region of interest 
2. resize
3. normalize

### CNN Architecture

Initially, I used a simple model CNN structure, with 2 convolution layers and 2 fully connected layer. However, it is difficult for the model to learn anything useful. The result is a constant steering angle. In addition, even with more epochs,  the validation error not decreasing.

Then I increase the number of layers to 4 convolution layers and 5 fully connected layer. The model starts to learning. The running loss starts to decrease. The car starts to make reasonable responses when it’s driving. However, it still drives off the track very often. The problem is the validation error is still big, which indicate overfitting. I add dropout layer after every convolution and fully connected layers, to make the model more robust. The new model takes much epoch to learn (about 110 epochs), but the validation error decreases further. 

In the result, the behavor is more smooth, most of the time the car in the middle of the road with rather large margin. In a sharp corner. It drives to edge of the track, but it immediate adjust to keep it back to the track.

### Data Generation

### Training

### Reflection
