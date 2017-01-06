# Self-Driving Car Engineer Nanodegree
# Deep Learning
## Project 3: Behavior Cloning

### Overview
This project aims to use CNN to training the car to drive itself in a simulation environment. The video of the final result in the following link: [YouTube](https://www.youtube.com/watch?v=JLVWC8iJfss)

### Dependencies
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

### Files:
- model.ipynb: the ipython notebook generates the CNN model . The detailed is described in the next section. 
- drive.py: the python script predicts the steering angle from the image in the simulation in real time and send the steering angle back to the simulation.
- model.joson: the final model
- model.hs: the weight of the final model
