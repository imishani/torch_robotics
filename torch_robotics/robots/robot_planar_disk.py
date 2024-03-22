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
