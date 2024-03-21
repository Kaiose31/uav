from sim.drone_env import DroneEnv 
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from sim.conn import client
import numpy as np

IMG_SHAPE =  (480, 640, 3)
TARGET = [-30, -10, -20]
env = DroneEnv(IMG_SHAPE, client, target=np.array(TARGET))

# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=50,
  save_path="model/ddpg",
  name_prefix="ddpg",
  save_replay_buffer=True,
  save_vecnormalize=True,
)


model =  DDPG("CnnPolicy", env,tensorboard_log = "data/DDPG_tensorboard").learn(total_timesteps=1000, callback = checkpoint_callback) 

