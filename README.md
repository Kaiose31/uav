# uav
Drone/UAV path planning using Reinforcement Learning and Simulator

This project uses Airsim's `SimpleFlight` controller and runs reinforcement learning algorithms on the environment using the gym framework.

## install and Setup
1. create data/ and model/ directory for storing datasets and model checkpoints.
2. ```pip install -r requirements.txt```
3. Set environment variable: ```export AIRSIMHOST=<host_ip>```


## Tensorboard for visualization 
```pip install tensorboard```
```tensorboard --logdir data/<model_tensorboard>```

## usage

to use the drone object:
```
from sim.conn import client
client.move...
```
## General info 

airsim uses NED coords:  +X North, +Y EAST, +Z DOWN
