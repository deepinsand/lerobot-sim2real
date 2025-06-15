from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Union, Sequence

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from sapien.render import RenderBodyComponent

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.so100.so_100 import SO100
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


# there are many ways to parameterize an environment's domain randomization. This is a simple way to do it
# with dataclasses that can be created and modified by the user and passed into the environment constructor.
@dataclass
class SO100AvoidCylinderDomainRandomizationConfig:
    ### task agnostic domain randomizations, many of which you can copy over to your own tasks ###
    initial_qpos_noise_scale: float = 0.02
    robot_color: Optional[Union[str, Sequence[float]]] = None
    """Color of the robot in RGB format in scale of 0 to 1 mapping to 0 to 255.
    If you want to randomize it just set this value to "random". If left as None which is
    the default, it will set the robot parts to white and motors to black. For more fine-grained choices on robot colors you need to modify
    mani_skill/assets/robots/so100/so100.urdf in the ManiSkill package."""
    randomize_table_color: bool = False # Example: can add more specific randomizations
    randomize_lighting: bool = True
    max_camera_offset: Tuple[float, float, float] = (0.025, 0.025, 0.025)
    """max camera offset from the base camera position in x, y, and z axes"""
    camera_target_noise: float = 1e-3
    """scale of noise added to the camera target position"""
    camera_view_rot_noise: float = 5e-3
    """scale of noise added to the camera view rotation"""
    camera_fov_noise: float = np.deg2rad(2)
    """scale of noise added to the camera fov"""

    ### Cylinder specific randomizations ###
    cylinder_speed_range: Tuple[float, float] = (0.05, 0.15) # m/s
    # Define a volume for cylinder's initial position and movement target
    cylinder_workspace_min: Tuple[float, float, float] = (0.1, -0.3, 0.05) # x, y, z
    cylinder_workspace_max: Tuple[float, float, float] = (0.5, 0.3, 0.4)  # x, y, z
    randomize_cylinder_orientation: bool = False # For now, keep orientation fixed or simple


@register_env("SO100AvoidCylinder-v1", max_episode_steps=256)
class SO100AvoidCylinderEnv(BaseDigitalTwinEnv):
    """
    **Task Description:**
    The objective is for the SO100 arm to avoid a white cylinder that moves randomly through the scene.
    The episode ends if the robot collides with the cylinder.

    **Randomizations:**
    - The cylinder's initial position is randomized within a defined workspace.
    - The cylinder's movement direction and speed are randomized.
    - Lighting, camera pose, and robot initial configuration are randomized.

    **Success Conditions:**
    - The robot successfully avoids collision with the cylinder for the duration of the episode.

    **Rewards:**
    - Dense reward proportional to the distance from the robot's TCP to the cylinder.
    - Large negative penalty upon collision with the cylinder.
    """

    # _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["so100"]
    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb+segmentation"]
    agent: SO100

    def __init__(
        self,
        *args,
        robot_uids="so100",
        control_mode="pd_joint_target_delta_pos",
        greenscreen_overlay_path=None,
        domain_randomization_config=SO100AvoidCylinderDomainRandomizationConfig(),
        domain_randomization=True,
        base_camera_settings=dict(
            fov=52 * np.pi / 180,
            pos=[0.5, 0.3, 0.35],
            target=[0.3, 0.0, 0.1],
        ),
        **kwargs,
    ):
        self.domain_randomization = domain_randomization
        """whether randomization is turned on or off."""
        self.domain_randomization_config = domain_randomization_config
        if not isinstance(self.domain_randomization_config, SO100AvoidCylinderDomainRandomizationConfig):
            logger.warning(f"domain_randomization_config is not of type SO100AvoidCylinderDomainRandomizationConfig, got {type(domain_randomization_config)}")
            if isinstance(domain_randomization_config, dict): # attempt to cast if it's a dict
                self.domain_randomization_config = SO100AvoidCylinderDomainRandomizationConfig(**domain_randomization_config)
        """domain randomization config"""
        self.base_camera_settings = base_camera_settings
        """what the camera fov, position and target are when domain randomization is off. DR is centered around these settings"""

        if greenscreen_overlay_path is None:
            logger.warning(
                "No greenscreen overlay path provided, no greenscreen will be used"
            )
            self.rgb_overlay_mode = "none"

        # set the camera called "base_camera" to use the greenscreen overlay when rendering
        else:
            self.rgb_overlay_paths = dict(base_camera=greenscreen_overlay_path)

        # Cylinder properties (convert inches to meters)
        self.cylinder_length = 6 * 0.0254  # meters
        self.cylinder_radius = (1 / 2) * 0.0254  # meters

        super().__init__(
            *args, robot_uids=robot_uids, control_mode=control_mode, **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20)

    @property
    def _default_sensor_configs(self):
        # we just set a default camera pose here for now. For sim2real we will modify this during training accordingly.
        # note that we pass in the camera mount which is created in the _load_scene function later. This mount lets us
        # randomize camera poses at each environment step. Here we just randomize some camera configuration like fov.
        if self.domain_randomization:
            camera_fov_noise = self.domain_randomization_config.camera_fov_noise * (
                2 * self._batched_episode_rng.rand() - 1
            )
        else:
            camera_fov_noise = 0
        return [
            CameraConfig(
                "base_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=camera_fov_noise + self.base_camera_settings["fov"],
                near=0.01,
                far=100,
                mount=self.camera_mount,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # this camera and angle is simply used for visualization purposes, not policy observations
        pose = sapien_utils.look_at([0.5, 0.3, 0.35], [0.3, 0.0, 0.1])
        return CameraConfig(
            "render_camera", pose, 512, 512, 52 * np.pi / 180, 0.01, 100
        )

    def _load_agent(self, options: dict):
        # load the robot arm at this initial pose
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.pi / 2)),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])

        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=False, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        # we use a predefined table scene builder which simply adds a table and floor to the scene
        # where the 0, 0, 0 position is the center of the table
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()

        # Cylinder visual material (white)
        cylinder_color = np.array([200 / 255., 180 / 255., 150 / 255., 1.0]) # RGBA

        # Build obstacle cylinder
        cylinders = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            # Using a capsule to represent the cylinder for collision and visual
            # SAPIEN uses half_length for capsules
            builder.add_capsule_collision(radius=self.cylinder_radius, half_length=self.cylinder_length / 2)
            builder.add_capsule_visual(
                radius=self.cylinder_radius,
                half_length=self.cylinder_length / 2,
                material=sapien.render.RenderMaterial(
                    base_color=cylinder_color,
                ),
            )
            # Initial pose will be set in _initialize_episode
            builder.initial_pose = sapien.Pose(p=[0, 0, -10]) # Initially far away
            builder.set_scene_idxs([i])
            # Cylinder is kinematic, its pose is set directly
            cylinder = builder.build_kinematic(name=f"obstacle_cylinder-{i}")
            cylinders.append(cylinder)
            self.remove_from_state_dict_registry(cylinder)

        self.obstacle_cylinder = Actor.merge(cylinders, name="obstacle_cylinder")
        self.add_to_state_dict_registry(self.obstacle_cylinder)

        # Objects to keep in render (not greenscreened)
        self.remove_object_from_greenscreen(self.agent.robot)
        self.remove_object_from_greenscreen(self.obstacle_cylinder)
        if self.domain_randomization and self.domain_randomization_config.randomize_table_color:
            # Example: if table color is randomized, ensure it's not greenscreened if needed
            pass
        else:
            # By default, table might be part of background for greenscreen
            # If you want the table to always be visible, remove it from greenscreen
            # self.remove_object_from_greenscreen(self.table_scene.table)
            pass

        # Define workspace for cylinder movement based on DR config
        self.cylinder_ws_min = common.to_tensor(self.domain_randomization_config.cylinder_workspace_min, device=self.device)
        self.cylinder_ws_max = common.to_tensor(self.domain_randomization_config.cylinder_workspace_max, device=self.device)

        # Initialize obstacle_cylinder_velocity here, as num_envs and device are available
        self.obstacle_cylinder_velocity = torch.zeros((self.num_envs, 3), device=self.device)

        # Ensure DR config tensors are on the correct device
        if self.domain_randomization:
            self.base_camera_settings["pos"] = common.to_tensor(
                self.base_camera_settings["pos"], device=self.device
            )
            self.base_camera_settings["target"] = common.to_tensor(
                self.base_camera_settings["target"], device=self.device
            )
            self.domain_randomization_config.max_camera_offset = common.to_tensor(
                self.domain_randomization_config.max_camera_offset, device=self.device
            )

        # a hardcoded initial joint configuration for the robot to start from
        self.rest_qpos = torch.tensor(
            [0, 0, 0, np.pi / 2, np.pi / 2, 0],
            device=self.device,
        )
        # hardcoded pose for the table that places it such that the robot base is at 0 and on the edge of the table.
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        # we build a 3rd-view camera mount to put cameras on which let us randomize camera poses at each timestep
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.camera_mount = builder.build_kinematic("camera_mount")

        # randomize or set a fixed robot color
        if self.domain_randomization_config.robot_color is not None:
            for link in self.agent.robot.links:
                for i, obj in enumerate(link._objs):
                    # modify the i-th object which is in parallel environment i
                    render_body_component: RenderBodyComponent = (
                        obj.entity.find_component_by_type(RenderBodyComponent)
                    )
                    if render_body_component is not None:
                        for render_shape in render_body_component.render_shapes:
                            for part in render_shape.parts:
                                if (
                                    self.domain_randomization
                                    and self.domain_randomization_config.robot_color
                                    == "random"
                                ):
                                    part.material.set_base_color(
                                        self._batched_episode_rng[i]
                                        .uniform(low=0.0, high=1.0, size=(3,))
                                        .tolist()
                                        + [1]
                                    )
                                else:
                                    part.material.set_base_color(
                                        list(
                                            self.domain_randomization_config.robot_color
                                        )
                                        + [1]
                                    )
    def sample_camera_poses(self, n: int):
        # a custom function to sample random camera poses
        # the way this works is we first sample "eyes", which are the camera positions
        # then we use the noised_look_at function to sample the full camera poses given the sampled eyes
        # and a target position the camera is pointing at
        if self.domain_randomization:
            eyes = randomization.camera.make_camera_rectangular_prism(
                n,
                scale=self.domain_randomization_config.max_camera_offset,
                center=self.base_camera_settings["pos"],
                theta=0,
                device=self.device,
            )
            return randomization.camera.noised_look_at(
                eyes,
                target=self.base_camera_settings["target"],
                look_at_noise=self.domain_randomization_config.camera_target_noise,
                view_axis_rot_noise=self.domain_randomization_config.camera_view_rot_noise,
                device=self.device,
            )
        else:
            return sapien_utils.look_at(
                eye=self.base_camera_settings["pos"],
                target=self.base_camera_settings["target"],
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # we randomize the pose of the cube accordingly so that the policy can learn to pick up the cube from
        # many different orientations and positions.
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # move the table back so that the robot is at 0 and on the edge of the table.
            self.table_scene.table.set_pose(self.table_pose)

            # sample a random initial joint configuration for the robot
            self.agent.robot.set_qpos(
                self.rest_qpos + torch.randn(size=(b, self.rest_qpos.shape[-1])) * 0.02
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )

            # Initialize cylinder pose and velocity
            initial_pos = torch.rand(b, 3, device=self.device) * (self.cylinder_ws_max - self.cylinder_ws_min) + self.cylinder_ws_min
            
            # Cylinder orientation (e.g., upright or along an axis)
            # For simplicity, let's align its length along the z-axis by default (for capsule)
            # Or make it random if configured
            if self.domain_randomization and self.domain_randomization_config.randomize_cylinder_orientation:
                qs = randomization.random_quaternions(b, lock_x=False, lock_y=False, device=self.device)
            else:
                # Default orientation (e.g. upright if capsule's half_length is along its local z)
                # Or aligned with world X axis: euler2quat(0, np.pi/2, 0)
                qs = torch.tensor([euler2quat(0, np.pi/2, 0, axes="rxyz")], device=self.device, dtype=torch.float32).repeat(b, 1)

            self.obstacle_cylinder.set_pose(Pose.create_from_pq(p=initial_pos, q=qs))

            # Sample random velocity for the cylinder
            rand_speeds = (torch.rand(b, device=self.device) *
                           (self.domain_randomization_config.cylinder_speed_range[1] - self.domain_randomization_config.cylinder_speed_range[0]) +
                           self.domain_randomization_config.cylinder_speed_range[0])
            
            # Random direction vector
            rand_directions = torch.randn(b, 3, device=self.device)
            rand_directions = rand_directions / torch.linalg.norm(rand_directions, dim=1, keepdim=True)
            
            self.obstacle_cylinder_velocity[env_idx] = rand_directions * rand_speeds.unsqueeze(1)

            # randomize the camera poses
            self.camera_mount.set_pose(self.sample_camera_poses(n=b))

    def _before_control_step(self):
        # Update cylinder pose based on its velocity
        dt = 1.0 / self.control_freq
        current_cylinder_pose = self.obstacle_cylinder.pose
        new_pos = current_cylinder_pose.p + self.obstacle_cylinder_velocity * dt
        
        # Optional: Keep cylinder within workspace (e.g. by reflecting velocity at boundaries)
        # For now, let it move freely. Can be refined later.
        # Example boundary check:
        # for i in range(3):
        #     hit_min = new_pos[:, i] < self.cylinder_ws_min[i]
        #     hit_max = new_pos[:, i] > self.cylinder_ws_max[i]
        #     self.obstacle_cylinder_velocity[hit_min, i] *= -1
        #     self.obstacle_cylinder_velocity[hit_max, i] *= -1
        # new_pos = torch.clamp(new_pos, self.cylinder_ws_min, self.cylinder_ws_max)

        self.obstacle_cylinder.set_pose(Pose.create_from_pq(p=new_pos, q=current_cylinder_pose.q))

        # Update camera poses
        if self.domain_randomization:
            self.camera_mount.set_pose(self.sample_camera_poses(n=self.num_envs))
        if self.gpu_sim_enabled: # ensure kinematic updates are applied
            self.scene._gpu_apply_all()

    def _get_obs_agent(self):
        # the default get_obs_agent function in ManiSkill records qpos and qvel. However
        # SO100 arm qvel are likely too noisy to learn from and not implemented.
        obs = dict(qpos=self.agent.robot.get_qpos())
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: Dict):
        # we ensure that the observation data is always retrievable in the real world, using only real world
        # available data (joint positions or the controllers target joint positions in this case).
        obs = dict(
            tcp_to_obstacle_pos=self.obstacle_cylinder.pose.p - self.agent.tcp_pose.p,
            obstacle_pose=self.obstacle_cylinder.pose.raw_pose, # Raw pose [pos (3), quat (4)]
        )
        if self.obs_mode_struct.state:
            # state based policies can gain access to more information that helps learning
            obs.update(
                tcp_pos=self.agent.tcp_pos,
                obstacle_velocity=self.obstacle_cylinder_velocity.clone(), # current velocity
            )
        return obs

    def evaluate(self):
        # evaluation function to generate some useful metrics/flags and evaluate the success of the task
        # Check for collision between robot and cylinder
        dist_tcp_to_cylinder_center = torch.linalg.norm(self.obstacle_cylinder.pose.p - self.agent.tcp_pose.p, axis=-1)

        # More accurate collision detection using contact forces
        collided_with_obstacle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Define robot links to check for collision.
        # These are common links for the SO100 end-effector. You might want to add more links
        # (e.g., forearm) if they are also likely to collide.
        # Actual link names depend on the robot's URDF file.
        # You can inspect all available link names via `self.agent.robot.links_map.keys()`
        # or by checking the SO100 URDF file directly.
        # The SO100 agent class defines:
        # - self.agent.finger1_link (maps to "Fixed_Jaw")
        # - self.agent.finger2_link (maps to "Moving_Jaw")
        # For other arm links, we use common naming conventions derived from joint names.
        robot_collision_links = [
            self.agent.finger1_link,
            self.agent.finger2_link,
            self.agent.robot.joints_map["wrist_roll"].get_child_link(),  # End-effector base / "palm"
            self.agent.robot.joints_map["wrist_flex"].get_child_link(),  # Part of the wrist/forearm
            self.agent.robot.joints_map["elbow_flex"].get_child_link(), # Part of the forearm/upper arm
        ]

        for link in robot_collision_links:
            contact_forces = self.scene.get_pairwise_contact_forces(link, self.obstacle_cylinder)
            # If the norm of contact forces is greater than a small threshold, consider it a collision.
            # The threshold (e.g., 1.0) might need tuning.
            collided_with_obstacle = torch.logical_or(collided_with_obstacle, torch.linalg.norm(contact_forces, dim=1) > 1.0)

        # Check if robot is touching the table (optional, but good for safety)
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.finger1_link, self.table_scene.table
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.agent.finger2_link, self.table_scene.table
        )
        lforce = torch.linalg.norm(l_contact_forces, dim=1) # Use dim=1 for batched envs
        rforce = torch.linalg.norm(r_contact_forces, dim=1)
        touching_table = torch.logical_or(
            lforce >= 1e-2,
            rforce >= 1e-2,
        )

        success = torch.logical_not(collided_with_obstacle) # Success is defined as not colliding

        return {
            "collided_with_obstacle": collided_with_obstacle,
            "dist_tcp_to_cylinder": dist_tcp_to_cylinder_center,
            "touching_table": touching_table,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Positive reward for surviving each step
        survival_bonus = 0.1

        # Reward for keeping distance from the cylinder
        distance_reward = torch.tanh(info["dist_tcp_to_cylinder"]) # Bounded reward, increases with distance

        reward = survival_bonus + distance_reward

        # Optional: Penalty for touching the table
        table_penalty_value = 0.5
        reward -= table_penalty_value * info["touching_table"].float()

        # Large negative penalty for collision
        collision_penalty_value = -10.0
        reward = torch.where(info["collided_with_obstacle"], torch.full_like(reward, collision_penalty_value), reward)

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        raw_reward = self.compute_dense_reward(obs=obs, action=action, info=info)

        # Normalize based on the potential range of non-collision rewards.
        # Max positive reward per step: survival_bonus (0.1) + max_distance_reward (tanh approaches 1.0) = 1.1
        # Min non-collision reward per step: survival_bonus (0.1) - table_penalty (0.5) = -0.4
        # We scale by the max possible positive component.
        max_expected_positive_reward_per_step = 0.1 + 1.0 # survival_bonus + max_tanh_dist

        # Scale all rewards by this factor.
        # Non-collision rewards will be roughly in:
        # Max: (0.1 + 1.0) / 1.1 = 1.0
        # Min: (0.1 - 0.5) / 1.1 = -0.4 / 1.1 approx -0.36
        # Collision penalty becomes: -10.0 / 1.1 approx -9.09
        normalized_reward = raw_reward / max_expected_positive_reward_per_step
        return normalized_reward

    # Override compute_terminated to end episode on collision
    def compute_terminated(self, obs: Any, action: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
        return info["collided_with_obstacle"]
