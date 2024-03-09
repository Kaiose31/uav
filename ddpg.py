from sim.drone_env import DroneEnv 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG
from sim.conn import client
import numpy as np

IMG_SHAPE =  (480, 640, 3)
TARGET = [-30, -10, -20]
env = DroneEnv(IMG_SHAPE, client, target=np.array(TARGET))
check_env(env)

model =  DDPG("CnnPolicy", env,tensorboard_log = "data/DDPG_tensorboard").learn(total_timesteps=1000) 

