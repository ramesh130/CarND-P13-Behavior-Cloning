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

### 1. Model Architecture Design
- Overview: The model architecture is based on the Nvidia paper on end-to-end learning. In this paper, a CNN is implemented with 5 convolutional layers, followed by 1 flattened layers, and then by 3 fully connected layers. At the end of the network is a single neuron which generates the steering angle. My CNN follows the similar architecture, expect  It contains 4 convolutional layers and 4 fully connected layers.

- Reason for using CNN:
1. The end-to-end learning problem is highly nonlinar problem, the deep neuro network can handle very complex problem.
2. The convolutional network is suited for image data, it use shared weight to reduce the number of parameter in the model.
3. The driving data from human driving the simulation, can be obtained easlier. large amount of data is require by CNN.


### 2. Architecture Characteristics,

- Number of layers
The neural network contains 4 convolutional layers, followed by a flattened layer, then followed by 4 fully connected layers. The last layer only contains 1 node, so it produces the steering angle.

Determine the number of layers: I started with a simple network with 2 convolution layers and 2 fully connected layer. The result However, the result is constant value the model is not learning. Then gradually increase the number of layers, like the Nevid paper. The model start to learning. To further reduce the loss I add dropout layer after each layer to prevent overfitting.


- Activation function: 
Relu is used as the activation function, compare to softmax it is more computationally efficient.

- Regularization:
A dropout layer is added after every convolution and fully connected layer. The layer eliminates a percentage of the output value to help the algorithm learn a more robust model.

The final parameters of the neural network have totally 6861 parameters. It is a reasonable size compared to the amount of data.


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


### 3. Data Preprocessing 

- Data Generation: The dataset contains 5 laps of the driving data. In 3 of the laps, the vehicle tries to stay in the center of the road. In the other 2 lap, it has many situations of recoveries. 320k image and steering angle pairs in the data. The preprocessing the size of the data size the training size is 1.2 Gb, can fit into memory.

The dataset is shuffled. 20% is reserved for the training set.  24 % of is chosen as validation set, and 56% is chosen as the training set.

- Preprocess steps:
1. Select region of interest: this step excludes the less relevant area, such as the sky and trees and focuses on the marks on the road.

2. Resize: to reduce the size of the image can reduce to reduce the memory and speed up the training process. The images still have 3 channels. The size of the cropped image is (80, 320, 3) was to reduced to (20, 80, 3).

3. Normalize: the original value of the image ranges from 0 to 255, normalization the new from -0.5 to 0.5.


### 4. Model Training and hyperparameter tuning

- Performance measurement:
Least-min-sequare(LMS) error is choose as the performance measurement. Beasuese, the output steering angle is a continous number, and LMS can indicate the difference beteen the predicted value and the value from the data.

- Training iterations:
The training of the model take 50 epoches. The first 30 epoches use larger learning rate (0.001). The model and weights are saved for further fine tuning. The later 20 epoches use smaller learning rate (0.00001). The loss in validation set is converged to is about 0.062.


- Parameter tuning

The paramter is tuning by observation the loss of validation set and the training set. If the loss of the validation set stop decrease and training set further decrease, it indicates the model is overtrained. A response is to add dropout layer to prevent overfittingand more possible more layer to increase the complexity of the model.
I started with a simple network with 2 convolution layers and 2 fully connected layer. However, the result is constant value, the model does not learning. 
gradually increase the number of layers and add dropout layer after each layer to prevent overfitting.


### Reflection
Initially, I used a simple model CNN structure, with 2 convolution layers and 2 fully connected layer. However, it is difficult for the model to learn anything useful. The result is a constant steering angle. Also, even with more epochs, the validation error not decreasing.

So I increase the number of layers to 4 convolution layers and 5 fully connected layer. The model starts to learning. The running loss starts to decrease. The car starts to make reasonable responses when it’s driving. However, it still drives off the track very often. The problem is the validation error is still big, which indicate overfitting. I add dropout layer after every convolution and fully connected layers, to make the model more robust. The new model takes much epoch to learn (about 50 epochs), but the validation error decreases  further. And the model become more robust.

In the final model, the vehicle behavior is more smooth. For most of the time the car drives in the middle of the road with a large margin. If the car go to the edge of the track, it immediate adjusts itself and went back to the center of the track.

The final loss is 0.062, which is not very low. But the performance of the very robust. When I plot the output steering angle, the results has more output maxium(1) and minor(-1) values. I guess, it ajust the car more abrupt to keep it in the track.

### Reference
Bojarski, Mariusz, et al. "End to End Learning for Self-Driving Cars." arXiv preprint arXiv:1604.07316 (2016).


