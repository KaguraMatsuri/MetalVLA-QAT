"""PokeBot absolute UR-base poses translated through the actual MuJoCo frames."""

import math

import mujoco
import numpy as np

from .api_policy import ApiError

POKE_PERIOD = 1./30.
POKE_CAMERA = "ur_hand_wrist"


class PokeExecutor:
    """No MCP runtime or second controller; use URTask's shared physics and IK."""

    def __init__(self, task, low, high):
        self.task = task
        base_id = task.model.body("ur_base").id
        self.origin = task.data.xpos[base_id].copy()
        self.rotation = task.data.xmat[base_id].reshape(3, 3).copy()
        self.low, self.high = np.asarray(low), np.asarray(high)

    def state(self):
        task = self.task
        position = self.rotation.T @ (task.tcp_position()-self.origin)
        orientation = self.rotation.T @ task.data.site_xmat[task.tcp_site].reshape(3, 3)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, orientation.ravel())
        # URTask measures closure, whereas the provider measures opening.
        opening = 1.-float(np.clip(task.state()[-1], 0., 1.))
        return [*position.tolist(), *quaternion[1:].tolist(), float(quaternion[0]), opening]

    def prepare(self, actions):
        """Validate the complete converted chunk before any command is applied."""
        converted = []
        for row in actions:
            position = self.origin + self.rotation @ np.asarray(row[:3])
            if not np.isfinite(position).all() or np.any(position < self.low) or np.any(position > self.high):
                raise ApiError("action_outside_workspace")
            local_rotation = np.empty(9)
            mujoco.mju_quat2Mat(local_rotation, np.asarray([row[6], *row[3:6]]))
            orientation = self.rotation @ local_rotation.reshape(3, 3)
            converted.append((position, orientation, row[7]))
        return converted

    def execute(self, action):
        position, orientation, opening = action
        yaw = math.atan2(orientation[1, 0], orientation[0, 0])
        return self.task.execute(position, yaw, opening < .5,
                                 target_rotation=orientation, gripper_open=opening, period=POKE_PERIOD)
