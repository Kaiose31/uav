"""Quadrotor with Collision Detection and Avoidance based on APF"""

from airsim import MultirotorClient
import numpy as np

# Globals
D_MAX = 12
D_CZ = 9
D_MZ = 1

# potential field coefficient constants.
# k_att = strength of attraction, k_rep = strength of repulsion
k_att = 0.2
k_rep = 0.9

# Max range of influence
p0 = 5


# APF implementation
def apf_gravity(current_pos: np.ndarray, goal_pos: np.ndarray):
    assert current_pos.shape == goal_pos.shape == (3,)
    d_goal = np.linalg.norm(current_pos - goal_pos)
    return k_att * (goal_pos - current_pos)


def apf_repel(current_pos: np.ndarray, obs_pos: np.ndarray, goal_pos: np.ndarray):
    assert current_pos.shape == obs_pos.shape == (3,)
    delta = current_pos - obs_pos
    d_barrier = np.linalg.norm(current_pos - obs_pos)
    return k_rep * ((1 / d_barrier) - (1 / p0)) * (1 / (d_barrier) ** 2) * np.array([delta[0] / d_barrier, delta[1] / d_barrier, delta[2] / d_barrier])


class Rotor(MultirotorClient):

    def __init__(self, *args, **kwargs):
        self.target_position = kwargs.pop("target_position", np.array([20, 0, -5]))
        self.mass = kwargs.pop("drone_mass", 1)
        super().__init__(*args, **kwargs)
        self.d_max = D_MAX
        self.d_cz = D_CZ
        self.d_mz = D_MZ

        #RL and collision avoidance time diff
        self.timestep = 0.35

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

    def apply_force(self, state):
        curr_pos, curr_vel = state.position.to_numpy_array(), state.linear_velocity.to_numpy_array()
        f_total = sum(apf_repel(curr_pos, point, self.target_position) for point in self.get_cz_points()) + apf_gravity(curr_pos, self.target_position)
        accel = f_total / self.mass
        vel = curr_vel + accel * self.timestep
        self.moveByVelocityAsync(vel[0], vel[1], vel[2], duration=self.timestep).join()


# Example for tuning
if __name__ == "__main__":
    r = Rotor("34.68.92.182", 41451, target_position=np.array([-30, -10, -5]))
    r.reset()
    r.confirmConnection()
    r.enableApiControl(True)
    r.moveToPositionAsync(0, 0, -4, 1).join()
    state = r.getMultirotorState().kinematics_estimated
    while 1:
        r.apply_force(state)
        print(np.linalg.norm(state.position.to_numpy_array() - r.target_position))
