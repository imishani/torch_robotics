import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase, EnvEncoderBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.robots import RobotPointMass
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSquare2D(EnvBase):

    def __init__(self, tensor_args=None, **kwargs):
        obj_list = [
            MultiBoxField(
                np.array(
                    [[-0, -0],
                     ]
                ),
                np.array(
                    [[1.0, 1.0]
                     ]
                )
                ,
                tensor_args=tensor_args
            )
        ]

        super().__init__(
            name=self.__class__.__name__,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[ObjectField(obj_list, 'square2d')],
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.3,
            n_pre_samples=50000,
            max_time=5
        )
        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=200,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=9e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )
        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError


class EnvRandSquare2D(EnvBase):

    def __init__(self,
                 name='EnvRandSquare2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 square_side_length_range=(0.05, 0.5),
                 n_squares=np.random.randint(8, 15),
                 **kwargs
                 ):
        self.tensor_args = tensor_args
        obj_list = self.generate_random_env(n_squares=n_squares, side_length_range=square_side_length_range)

        object_field = ObjectField(obj_list, 'randsquare2d')

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[object_field],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def generate_random_env(self,
                            n_squares=5,
                            side_length_range=(0.05, 0.3)):
        obj_list = []
        centers = []
        sides = []

        for _ in range(n_squares):
            def sample():
                side = np.random.uniform(side_length_range[0], side_length_range[1])
                return (torch.tensor([np.random.uniform(-1, 1),
                                      np.random.uniform(-1, 1)],
                                     **self.tensor_args).view(1, -1),
                        torch.tensor([side, side],
                                     **self.tensor_args).view(1, -1))

            center, side_length = sample()
            valid = False
            while not valid:
                valid = True
                for c, s in zip(centers, sides):
                    x_dist = torch.abs(center[0, 0] - c[0, 0])
                    y_dist = torch.abs(center[0, 1] - c[0, 1])
                    x_overlap = side_length[0, 0] / 2 + s[0, 0] / 2 - x_dist
                    y_overlap = side_length[0, 0] / 2 + s[0, 0] / 2 - y_dist
                    if x_overlap > 0 and y_overlap > 0:
                        valid = False
                        center, side_length = sample()
                        break
            centers.append(center)
            sides.append(side_length)

        obj_list.append(MultiBoxField(
            torch.cat(centers, dim=0),
            torch.cat(sides, dim=0),
            tensor_args=self.tensor_args
        ))
        return obj_list

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.3,
            n_pre_samples=50000,
            max_time=50
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError


class EnvRandSquare2DEncoder(EnvEncoderBase):

    def __init__(self,
                 env: EnvRandSquare2D,
                 max_encoding_size=(14 * 2) + 14,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.encoding = None
        self.max_encoding_size = max_encoding_size

    def encode(self):
        if self.encoding is not None:
            return self.encoding

        objs = self.env.get_obj_list()
        centers = torch.Tensor().to(self.env.tensor_args["device"])
        sizes = torch.Tensor().to(self.env.tensor_args["device"])
        for obj_field in objs:
            for obj in obj_field.fields:
                if isinstance(obj, MultiBoxField):
                    centers = torch.cat([centers, obj.centers], dim=0)
                    sizes = torch.cat([sizes, obj.sizes[:, 0]], dim=0)
        encoded = torch.cat([centers, sizes.view(-1, 1)], dim=1)
        self.encoding = encoded.flatten()
        # padding
        self.encoding = torch.cat([self.encoding, torch.zeros(self.max_encoding_size - self.encoding.shape[0],
                                                              device=self.env.tensor_args["device"])])
        return self.encoding


if __name__ == '__main__':
    env = EnvRandSquare2D(
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    encder = EnvRandSquare2DEncoder(env)
    print(encder.encode())

    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
