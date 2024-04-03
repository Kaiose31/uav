"""Quadrotor with Collision Detection and Avoidance based on APF"""

from airsim import MultirotorClient
from enum import Enum
import numpy as np


# Globals
D_MAX = 10
D_CZ = 5
D_MZ = 1

# potential field coefficient constants.
# k_att = strength of attraction, k_rep = strength of repulsion
k_att = 0.5
k_rep = 0.6

# Max range of influence
p0 = 5


# APF implementation
def apf_gravity(current_pos: np.ndarray, goal_pos: np.ndarray):
    assert current_pos.shape == goal_pos.shape == (3,)
    d_goal = np.linalg.norm(current_pos - goal_pos)
    return k_att * (goal_pos - current_pos)


def apf_repel(current_pos: np.ndarray, obs_pos: np.ndarray, goal_pos: np.ndarray):
    assert current_pos.shape == obs_pos.shape == (3,)
    d_barrier = np.linalg.norm(current_pos - obs_pos)
    return k_rep * ((1 / d_barrier) - (1 / p0)) * (1 / (d_barrier) ** 2) * np.gradient(d_barrier / current_pos)


class Rotor(MultirotorClient):

    def __init__(self, *args, **kwargs):
        self.target_position = kwargs.pop("target_position", np.array([20, 0, -5]))
        self.mass = kwargs.pop("drone_mass", 1)
        super().__init__(*args, **kwargs)
        self.d_max = D_MAX
        self.d_cz = D_CZ
        self.d_mz = D_MZ
        self.timestep = 0.1

    def is_collision_risk(self) -> bool:
        return self.points_cz() > 0

    def is_failure(self) -> bool:
        return self.points_mz() > 0

    def _lidar_dist(self):
        ld = super().getLidarData().point_cloud
        if len(ld) < 3:
            return np.empty(0), np.empty(0)
        ld_np = np.array(ld).reshape((len(ld) // 3, 3))
        return np.linalg.norm(ld_np - super().getMultirotorState().kinematics_estimated.position.to_numpy_array(), axis=1), ld_np

    def points_cz(self) -> int:
        distances = self._lidar_dist()[0]
        return len(distances[(distances < D_CZ) & (distances > D_MZ)]) if distances.size != 0 else 0

    def points_mz(self) -> int:
        distances = self._lidar_dist()[0]
        return len(distances[(distances < D_MZ)])

    def points_sz(self) -> int:
        distances = self._lidar_dist()[0]
        return len(distances[(distances < D_MAX) & (distances > D_CZ)])

    def get_cz_points(self):
        distances, ld_np = self._lidar_dist()
        col_idx = np.where((distances < D_CZ) & (distances > D_MZ))[0]
        return ld_np[col_idx]

    def debug_collision(self):
        print(f"collision points\t {self.get_cz_points()}")

    def apply_force(self):
        curr_pos = self.getMultirotorState().kinematics_estimated.position.to_numpy_array()
        curr_vel = self.getMultirotorState().kinematics_estimated.linear_velocity.to_numpy_array()
        f_total = sum(apf_repel(curr_pos, point, self.target_position) for point in self.get_cz_points()) + apf_gravity(curr_pos, self.target_position)
        accel = f_total / self.mass
        vel = curr_vel + accel * self.timestep
        self.moveByVelocityAsync(vel[0], vel[1], vel[2], duration=0.1).join()


if __name__ == "__main__":
    import os
    # Example
    r = Rotor(os.environ["AIRSIMHOST"], 41451, target_position=np.array([20, 0, -10]))
    r.reset()
    r.confirmConnection()
    r.armDisarm(True)
    r.enableApiControl(True)
    r.moveToPositionAsync(0, 0, -3, 3).join()
    r.moveToPositionAsync(10, -7, -3, 10).join()
    while 1:
        r.apply_force()
        print(f"distance to target: {np.linalg.norm(r.getMultirotorState().kinematics_estimated.position.to_numpy_array() -  r.target_position)}")
