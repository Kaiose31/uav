from sim.drone_env import DroneEnv 
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG
from sim.conn import client
IMG_SHAPE =  (480, 640, 3)
env = DroneEnv(IMG_SHAPE, client)
check_env(env)


model =  DDPG("MlpPolicy", DroneEnv).learn(total_timesteps=1000)

