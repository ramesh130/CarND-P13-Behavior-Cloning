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
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cn1 (Convolution2D)              (None, 9, 39, 24)     672         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 9, 39, 24)     0           cn1[0][0]                        
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 9, 39, 24)     0           dropout_1[0][0]                  
____________________________________________________________________________________________________
cn2 (Convolution2D)              (None, 4, 19, 12)     2604        activation_1[0][0]               
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 4, 19, 12)     0           cn2[0][0]                        
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 4, 19, 12)     0           dropout_2[0][0]                  
____________________________________________________________________________________________________
cn3 (Convolution2D)              (None, 2, 17, 8)      872         activation_2[0][0]               
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 2, 17, 8)      0           cn3[0][0]                        
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 2, 17, 8)      0           dropout_3[0][0]                  
____________________________________________________________________________________________________
cn4 (Convolution2D)              (None, 1, 16, 4)      132         activation_3[0][0]               
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 1, 16, 4)      0           cn4[0][0]                        
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 1, 16, 4)      0           dropout_4[0][0]                  
____________________________________________________________________________________________________
flatten (Flatten)                (None, 64)            0           activation_4[0][0]               
____________________________________________________________________________________________________
fc1 (Dense)                      (None, 20)            1300        flatten[0][0]                    
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 20)            0           fc1[0][0]                        
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 20)            0           dropout_5[0][0]                  
____________________________________________________________________________________________________
fc2 (Dense)                      (None, 20)            420         activation_5[0][0]               
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 20)            0           fc2[0][0]                        
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 20)            0           dropout_6[0][0]                  
____________________________________________________________________________________________________
fc3 (Dense)                      (None, 20)            420         activation_6[0][0]               
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 20)            0           fc3[0][0]                        
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 20)            0           dropout_7[0][0]                  
____________________________________________________________________________________________________
fc4 (Dense)                      (None, 20)            420         activation_7[0][0]               
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 20)            0           fc4[0][0]                        
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 20)            0           dropout_8[0][0]                  
____________________________________________________________________________________________________
fc5 (Dense)                      (None, 1)             21          activation_8[0][0]               
====================================================================================================
Total params: 6861
____________________________________________________________________________________________________

### Data Generation

### Training

### Reflection
Initially, I used a simple model CNN structure, with 2 convolution layers and 2 fully connected layer. However, it is difficult for the model to learn anything useful. The result is a constant steering angle. In addition, even with more epochs,  the validation error not decreasing.

Then I increase the number of layers to 4 convolution layers and 5 fully connected layer. The model starts to learning. The running loss starts to decrease. The car starts to make reasonable responses when it’s driving. However, it still drives off the track very often. The problem is the validation error is still big, which indicate overfitting. I add dropout layer after every convolution and fully connected layers, to make the model more robust. The new model takes much epoch to learn (about 110 epochs), but the validation error decreases further. 

In the result, the behavor is more smooth, most of the time the car in the middle of the road with rather large margin. In a sharp corner. It drives to edge of the track, but it immediate adjust to keep it back to the track.

