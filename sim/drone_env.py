import airsim
import gymnasium 
import numpy as np
from gymnasium import spaces
from sim.utils import get_img

class DroneEnv(gymnasium.Env):
    
    yaw_min, yaw_max = -np.pi, np.pi  # Range for yaw angle in radians
    pitch_min, pitch_max = -np.pi, np.pi  # Range for pitch angle in radians    
    roll_min, roll_max = -np.pi, np.pi  # Range for roll angle in radians
    throttle_min, throttle_max = 0.0, 1.0  # Range for throttle


    def __init__(self, img_shape: tuple[int], client: airsim.MultirotorClient = None):
        super().__init__()
    

        yaw_min, yaw_max = -np.pi, np.pi  # Range for yaw angle in radians
        pitch_min, pitch_max = -np.pi, np.pi  # Range for pitch angle in radians    
        roll_min, roll_max = -np.pi, np.pi  # Range for roll angle in radians
        throttle_min, throttle_max = 0.0, 1.0  # Range for throttle
    
        self.action_space = spaces.Box(low = np.array([roll_min, pitch_min, yaw_min, throttle_min]),
                                       high = np.array([roll_max, pitch_max, yaw_max, throttle_max]),
                                       dtype = np.float32)

        self.observation_space = spaces.Box(low = 0, high = 255, shape = img_shape, dtype = np.uint8)

        self.drone = client
        self._setup_flight()



    def _setup_flight(self):

        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.takeoffAsync().join()
        self.drone.moveToPositionAsync(0, -5, 0, 10).join()

        

    def step(self, action):
        #TODO! Box action space -> move drone in sim
        self.drone.moveByRollPitchYawrateThrottleAsync(float(action[0]), float(action[1]), float(action[2]), float(action[3]), duration = 1).join()
        #TODO! Write a good reward function
        self.drone.simPause(True)
        reward = -1.0 if self.drone.simGetCollisionInfo().has_collided else 1.0
        self.drone.simPause(False)
        
        return get_img(), reward, True if reward == -1.0 else False, True, {} 


    def reset(self, seed = None, options = None):
        self._setup_flight()
        return get_img(), {}




