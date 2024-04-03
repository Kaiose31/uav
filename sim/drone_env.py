from .rotor import Rotor
import gymnasium
import numpy as np
from gymnasium import spaces
from sim.utils import get_img, map_actions


class DroneEnv(gymnasium.Env):

    def __init__(self, img_shape: tuple, client: Rotor, target: np.ndarray, step_size=0.1):
        super().__init__()

        self.step_size = step_size
        self.state = {"position": np.zeros(3), "prev_position": np.zeros(3), "dist_to_target": np.inf}
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)
        self.drone = client
        self.drone.target_position = target
        self._setup_flight()

    def _get_obs(self):
        image = get_img()
        drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["position"]
        pos = drone_state.kinematics_estimated.position
        self.state["position"] = pos
        self.state["dist_to_target"] = np.linalg.norm(pos.to_numpy_array() - self.drone.target_position)
        return image

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToPositionAsync(0, 0, -5, 5).join()

    def step(self, action):
        ac = map_actions(action)
        # APF action
        self.drone.apply_force()
        # model action
        self.drone.moveByRollPitchYawrateThrottleAsync(**ac, duration=self.step_size).join()
        # TODO! Write a good reward function
        reward, done = self.reward()
        print(f"actions: {ac}, reward: {reward:.2f}, ep_done: {done}, dist_to_target: {self.state['dist_to_target']:.2f}")
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        self._setup_flight()
        return self._get_obs(), {}

    def reward(self):
        distance = self.state["dist_to_target"]

        if self.drone.simGetCollisionInfo().has_collided:
            reward = -100
        elif distance <= 10:
            reward = 1
        else:

            reward = -self.state["dist_to_target"]
        done = False
        if reward <= -100:
            done = True
        return reward, done
