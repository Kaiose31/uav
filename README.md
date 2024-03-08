# uav
Drone/UAV path planning using Reinforcement Learning and Simulator

## install and Setup
1. create data/ and model/ directory for storing datasets and model checkpoints.
2. ```pip install -r requirements.txt```
3. Set environment variable: ```export AIRSIMHOST=<gcloud-external-ip>```

## usage

to use the drone object:
```
from sim.conn import client
client.move...
```

