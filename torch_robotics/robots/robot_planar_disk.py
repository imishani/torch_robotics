import numpy as np
import torch

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.robots.robot_point_mass import RobotPointMass
from torch_robotics.torch_utils.torch_utils import to_numpy, tensor_linspace_v1, to_torch


class RobotPlanarDisk(RobotPointMass):

    def __init__(self,
                 name='RobotPlanarDisk',
                 radius=0.04,
                 q_limits=torch.tensor([[-1, -1], [1, 1]]),  # Confspace limits.
                 **kwargs):

        # Set the link margins for object collision checking to the radius of the disk here and in the parent.
        self.link_margins_for_object_collision_checking = [radius]

        super().__init__(
            name=name,
            q_limits=to_torch(q_limits, **kwargs['tensor_args']),
            **kwargs
        )

        ################################################################################################
        # Robot
        self.radius = radius
        self.link_margins_for_object_collision_checking = [radius]

    def check_rr_collisions(self, robot_q: torch.tensor) -> torch.tensor:
        """
        Check collisions between robots. (Robot-robot collisions).
        Args:
            robot_q: (..., n_robots, q_dim)
        Returns:
            collisions: (..., n_robots, n_robots), True if there is a collision between the robots.
        """
        # Check collisions between robots.
        assert robot_q.dim() >= 2
        robot_q1 = robot_q.unsqueeze(-2)
        robot_q2 = robot_q.unsqueeze(-3)
        # Pairwise distances between robots.
        robot_dq = robot_q1 - robot_q2
        # (..., n_robots, n_robots)
        robot_dq_norm = torch.norm(robot_dq, dim=-1)
        # (..., n_robots, n_robots)
        collisions = robot_dq_norm < 3 * self.radius
        # Set the trace to be False.
        collisions = collisions & ~torch.eye(collisions.shape[-1], device=collisions.device, dtype=collisions.dtype)
        return collisions

