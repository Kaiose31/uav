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
    
    def R_a(d_max, d_goal_t, d_goal_t_plus_1):
        tanh_diff = np.tanh(d_max - d_goal_t)
        
        abs_tanh_diff = np.abs(tanh_diff)
        
        if d_goal_t - d_goal_t_plus_1 == 0 or abs_tanh_diff == 0:
            return 0
        else:
            return abs_tanh_diff * ((d_goal_t - d_goal_t_plus_1) / np.abs(d_goal_t - d_goal_t_plus_1))
        
    def R_mz(D_cz, d_barrier_t, d_barrier_t_plus_1):
        tanh_diff = np.tanh(D_cz - d_barrier_t)
        
        abs_tanh_diff = np.abs(tanh_diff)
        
        if d_barrier_t_plus_1 - d_barrier_t == 0 or abs_tanh_diff == 0:
            return 0
        else:
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

    def reward(self):
        d_goal = self.state["dist_to_target"]
        d_max = self.drone.d_max
        d_cz = self.drone.d_cz
        # d_mz = self.drone.d_mz
        cz_points = self.drone.get_cz_points()

        min_distance = -1
        for _,point in enumerate(cz_points):
            distance = np.linalg.norm(self.state.position.to_numpy_array() - point)
            if distance < min_distance:
                min_distance = distance
        d_barrier = min_distance

        if self.drone.simGetCollisionInfo().has_collided:
            done = True
            reward = -1000
        elif (self.drone.points_mz() == 0 and self.drone.points_cz() == 0):
            reward = self.R_a(d_max,d_goal,1)
            done = False
        elif (self.drone.points_cz() > 0 and self.drone.points_mz() == 0):
            reward = self.R_cz(d_cz,d_barrier,1,d_max,d_goal,1)
            done = False
        elif (self.drone.points_mz() > 0):
            reward = self.R_mz(d_cz,d_barrier,1)
            done = False
        return reward, done
