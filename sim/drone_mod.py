import gymnasium
import numpy as np
from gymnasium import spaces
from sim.utils import to_euler_angles
import math
from typing import List, Tuple, Any
from sim.conn import client
from sim.utils import to_euler_angles
from airsim.types import Vector3r, Pose, Quaternionr
import math


class DroneMod(gymnasium.Env):

    def __init__(self, target: List[float]):
        self.target = Vector3r(*target)
        self.drone = client
        self.drone_pos = None
        self.observation_space = spaces.Box(
            low=np.array([-1000, -1000, -1000, -100, -100, -100, -np.pi, -np.pi / 2, -np.pi, -100, -100, -100, 0, 0, 0, -np.pi / 2, -np.pi, 0]),
            high=np.array([1000, 1000, 1000, 100, 100, 100, np.pi, np.pi / 2, np.pi, 100, 100, 100, 1000, 1000, 1000, np.pi / 2, np.pi, 1000]),
        )
        # Roll, pitch, yaw, throttle
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]))

    def _get_info(self):
        return {"distance": self.drone_pos.distance_to(self.target)}

    def _get_obs(self) -> Tuple[List[Any]]:
        pose = client.simGetVehiclePose()
        angular_vel = client.getImuData().angular_velocity
        linear_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
        self.prev_position = self.drone_pos
        pos = pose.position
        self.drone_pos = pos
        roll, pitch, yaw = to_euler_angles(pose.orientation)
        dist_to_target = pos.distance_to(self.target)
        position_diff = self.target - pos
        return np.array(
            [
                # Basic
                pos.x_val,
                pos.y_val,
                pos.z_val,
                linear_vel.x_val,
                linear_vel.y_val,
                linear_vel.z_val,
                roll,
                pitch,
                yaw,
                angular_vel.x_val,  # roll rate
                angular_vel.y_val,  # pitch rate
                angular_vel.z_val,  # yaw rate
                # Relative
                abs(self.target.x_val - pos.x_val),
                abs(self.target.y_val - pos.y_val),
                abs(self.target.z_val - pos.z_val),
                math.asin((self.target.z_val - pos.z_val) / dist_to_target),  # pitch angle to target
                math.atan2(position_diff.y_val, position_diff.x_val),  # heading angle to target
                dist_to_target,
            ],
            dtype=np.float32,
        )

    def _take_action(self, action):
        action_range = [-1, 1]
        roll = np.interp(action[0], action_range, [-np.pi / 2, np.pi / 2]).round(2)
        pitch = np.interp(action[1], action_range, [-np.pi / 2, np.pi / 2]).round(2)
        yaw = np.interp(action[2], action_range, [-np.pi / 4, np.pi / 4]).round(2)
        throttle = np.interp(action[3], action_range, [0, 1]).round(2)
        self.drone.moveByRollPitchYawThrottleAsync(roll, pitch, yaw, throttle, duration=0.5).join()

    def step(self, action):
        self._take_action(action)
        collided = self.drone.simGetCollisionInfo().has_collided
        dist_to_target = self.drone_pos.distance_to(self.target)
        obs = self._get_obs()
        terminated, reward = self.reward(obs)
        return obs, reward, terminated, False, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.simSetVehiclePose(Pose(Vector3r(0, 0, -2), Quaternionr()), ignore_collision=True)
        return self._get_obs(), self._get_info()

    def reward(self, obs):
        los = 0.666 * np.exp(-0.5 * (obs[14] / 0.5) ** 2)
        close = 0 if self.prev_position is None else self.prev_position - self.drone_pos
        if obs[17] < 5 and obs[14] < 1 and abs(obs[7]) < 0.08:
            return True, 5 + los + close
        elif self.drone.simGetCollisionInfo().has_collided:
            return True, -10 + los + close
        else:
            return False, 0 + los + close
