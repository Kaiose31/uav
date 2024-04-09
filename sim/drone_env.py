from .rotor import Rotor
import gymnasium
import numpy as np
from gymnasium import spaces
from sim.utils import get_img, map_actions
import os


class DroneEnv(gymnasium.Env):

    def __init__(self, img_shape: tuple, client: Rotor, target: np.ndarray, step_size=0.01, start_position=[0, 0, -5], goal_threshold=2.0):
        super().__init__()
        self.start_position = start_position
        self.step_size = step_size
        self.goal_threshold = goal_threshold
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
        self.state["position"] = pos.to_numpy_array()
        self.state["dist_to_target"] = np.linalg.norm(pos.to_numpy_array() - self.drone.target_position)
        return image

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToPositionAsync(self.start_position[0], self.start_position[1], self.start_position[2], 1).join()

    def step(self, action):
        ac = map_actions(action)
        # APF action
        self.drone.apply_force()
        # model action
        self.drone.moveByRollPitchYawrateThrottleAsync(**ac, duration=self.step_size).join()
        obs = self._get_obs()
        reward, done = self.reward()
        info = {"actions": ac, "reward": reward, "ep_done": done, "dist_to_target": self.state["dist_to_target"]}
        if os.environ.get("DEBUG"):
            print(info)
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self._setup_flight()
        return self._get_obs(), {}

    def reward(self):
        # Target is reached
        if self.state["dist_to_target"] < self.goal_threshold:
            return 1, True
        d_goal = np.linalg.norm(self.state["prev_position"] - self.drone.target_position)  # Previous position to target position
        d_goal_t_plus_one = self.state["dist_to_target"]  # Current position to target position
        # Dist from origin to goal
        d_max = np.linalg.norm(self.start_position - self.drone.target_position)
        d_cz = self.drone.d_cz  # Hyperparameter
        # d_mz = self.drone.d_mz
        cz_points = self.drone.get_cz_points()  # List of points that are in range of CZ

        min_distance = -1
        for _, point in enumerate(cz_points):  # Loop through all barrier points
            distance = np.linalg.norm(self.state["position"] - point)  # Calculate distance between current position and barrier point
            if distance < min_distance:
                min_distance = distance
        d_barrier_t_plus_one = min_distance  # Track the closest barrier point

        min_distance_curr = -1
        for _, point in enumerate(cz_points):  # Loop through all barrier points
            distance = np.linalg.norm(self.state["prev_position"] - point)  # Calculate distance between previous position and barrier point
            if distance < min_distance_curr:
                min_distance_curr = distance
        d_barrier = min_distance_curr  # Track the closest barrier point
        reward = 0
        done = False
        if self.drone.simGetCollisionInfo().has_collided and self.drone.is_failure():
            done = True
            reward = -1
        elif self.drone.points_mz() == 0 and self.drone.points_cz() == 0:
            reward = R_a(d_max, d_goal, d_goal_t_plus_one)
            done = False
        elif self.drone.points_cz() > 0 and self.drone.points_mz() == 0:
            reward = R_cz(d_cz, d_barrier, d_barrier_t_plus_one, d_max, d_goal, d_goal_t_plus_one)
            done = False
        elif self.drone.points_mz() > 0:
            reward = R_mz(d_cz, d_barrier, d_barrier_t_plus_one)
            done = False
        return reward, done


def R_a(d_max, d_goal_t, d_goal_t_plus_1):
    tanh_diff = np.tanh(d_max - d_goal_t)

    abs_tanh_diff = np.abs(tanh_diff)
    if d_goal_t - d_goal_t_plus_1 == 0 or abs_tanh_diff == 0:
        return 0
    return abs_tanh_diff * ((d_goal_t - d_goal_t_plus_1) / np.abs(d_goal_t - d_goal_t_plus_1))


def R_mz(D_cz, d_barrier_t, d_barrier_t_plus_1):
    tanh_diff = np.tanh(D_cz - d_barrier_t)

    abs_tanh_diff = np.abs(tanh_diff)

    if d_barrier_t_plus_1 - d_barrier_t == 0 or abs_tanh_diff == 0:
        return 0
    return abs_tanh_diff * ((d_barrier_t_plus_1 - d_barrier_t) / np.abs(d_barrier_t_plus_1 - d_barrier_t))


def R_cz(D_cz, d_barrier_t, d_barrier_t_plus_1, d_max, d_goal_t, d_goal_t_plus_1):
    tanh_diff_barrier = np.tanh(D_cz - d_barrier_t)

    abs_tanh_diff_barrier = np.abs(tanh_diff_barrier)

    if d_barrier_t_plus_1 - d_barrier_t == 0 or abs_tanh_diff_barrier == 0:
        term_barrier = 0
    else:
        term_barrier = abs_tanh_diff_barrier * ((d_barrier_t_plus_1 - d_barrier_t) / np.abs(d_barrier_t_plus_1 - d_barrier_t))

    tanh_diff_goal = np.tanh(d_max - d_goal_t)

    abs_tanh_diff_goal = np.abs(tanh_diff_goal)

    if d_goal_t - d_goal_t_plus_1 == 0 or abs_tanh_diff_goal == 0:
        term_goal = 0
    else:
        term_goal = abs_tanh_diff_goal * ((d_goal_t - d_goal_t_plus_1) / np.abs(d_goal_t - d_goal_t_plus_1))
    return term_barrier + term_goal
