# Self-Driving Car Engineer Nanodegree
# Deep Learning
# Project 3: Behavior Cloning

## Overview
This project aims to train the car to drive itself in a simulation environment, with the images-steering-angle-pair data using a deep neural network. The video of the final result is shown in the following link: [YouTube](https://www.youtube.com/watch?v=JLVWC8iJfss)

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
- model.ipynb: the ipython notebook generates the CNN model. The detailed is described in the next section. 
- drive.py: the python script predicts the steering angle from the image in the simulation in real time and sends the steering angle back to the simulation.
- model.joson: the final model
- model.h5: the weight of the final model

## Detail Description:
### Preprocess
- Select region of interest: this step excludes the less relevant area, such as the sky and trees and focuses on the marks on the road.

- Resize: to reduce the size of the image can reduce to reduce the memory and speed up the training process. The images still have 3 channels. The size of the cropped image is (80, 320, 3) was to reduced to (20, 80, 3).

- Normalize: the original value of the image ranges from 0 to 255, normalization the new from -0.5 to 0.5.

### CNN Architecture

- Number of layers
The neural network contains 4 convolutional layers, followed by a flattened layer, then followed by 4 fully connected layers. The last layer only contains 1 node, so it produces the steering angle.

- Activation function: 
Relu is used as the activation function, compare to softmax it is more computationally efficient.

- Regularization:
A dropout layer is added after every convolution and fully connected layer. The layer eliminates a percentage of the output value to help the algorithm learn a more robust model.

The final parameters of the neural network have totally 6861 parameters. It is a reasonable size compared to the amount of data.

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

### Data Generation

The dataset contains 5 laps of the driving data. In 3 of the laps, the vehicle tries to stay in the center of the road. In the other 2 lap, it has many situations of recoveries. 320k image and steering angle pairs in the data. The preprocessing the size of the data size the training size is 1.2 Gb, can fit into memory.

The dataset is shuffled. 30% of the data is chosen as validation set and 70% is chosen as the training set.

### Training
The training model 110 epoch. The large learning rate is chosen at first. model is saved, and learning rate is gradually reduced from 0.001 to  0.00001 to fine-tune the data. The model and weight are saved for retraining. The final loss is about 0.053.

### Reflection
Initially, I used a simple model CNN structure, with 2 convolution layers and 2 fully connected layer. However, it is difficult for the model to learn anything useful. The result is a constant steering angle. Also, even with more epochs,  the validation error not decreasing.

Then I increase the number of layers to 4 convolution layers and 5 fully connected layer. The model starts to learning. The running loss starts to decrease. The car starts to make reasonable responses when it’s driving. However, it still drives off the track very often. The problem is the validation error is still big, which indicate overfitting. I add dropout layer after every convolution and fully connected layers, to make the model more robust. The new model takes much epoch to learn (about 110 epochs), but the validation error decreases further. 

In the final model, the vehicle behavior is more smooth. For most of the time the car drives in the middle of the road with a large margin. Only in one instance at a sharp corner, the car went to the edge of the track, but it immediate adjusts itself and went back to the center of the track.

