import airsim
import gymnasium 
import numpy as np
from gymnasium import spaces
from sim.utils import get_img, map_actions
import math 

class DroneEnv(gymnasium.Env):
    
    def __init__(self, img_shape: tuple, client: airsim.MultirotorClient, target: np.ndarray):
        super().__init__()
    
        self.target = target
        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            "dist_to_target": np.inf
        }

        self.action_space = spaces.Box(low = np.array([-1., -1., -1., -1.]),
                                       high = np.array([1., 1., 1., 1.]),
                                       dtype = np.float32)

        self.observation_space = spaces.Box(low = 0, high = 255, shape = img_shape, dtype = np.uint8)

        self.drone = client
        self._setup_flight()
    
    def _get_obs(self):
        image = get_img()
        self.drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["position"]
        pos = self.drone_state.kinematics_estimated.position
        self.state["position"] = pos 
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision
        self.state["dist_to_target"] = np.linalg.norm(pos.to_numpy_array() - self.target)
        return image
    

    def _setup_flight(self):

        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.takeoffAsync().join()

    def step(self, action):
        ac = map_actions(action)
        self.drone.moveByRollPitchYawrateThrottleAsync(**ac,duration=0.1).join()
        #TODO! Write a good reward function
        reward, done = self.reward()
        print(f"actions: {ac}, reward: {reward:.2f}, ep_done: {done}, dist_to_target: {self.state['dist_to_target']:.2f}")
        return self._get_obs(), reward, done, False, {}


    def reset(self, seed = None, options = None):
        self._setup_flight()
        return self._get_obs(), {}


    def reward(self):
        if self.drone.simGetCollisionInfo().has_collided:
            reward = -100
        else:
            reward = -self.state["dist_to_target"]
        done = False
        if reward <= -100: 
            done = True
        return reward, done
