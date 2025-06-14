from pathlib import Path
import gymnasium as gym
from lerobot.common.robots.robot import Robot
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.robots.utils import make_robot_from_config
import numpy as np
from lerobot.common.cameras.opencv import ( # Assuming this is the correct import
    OpenCVCameraConfig,
)

def create_real_robot(uid: str = "so100") -> Robot:
    """Wrapper function to map string UIDS to real robot configurations. Primarily for saving a bit of code for users when they fork the repository. They can just edit the camera, id etc. settings in this one file."""
    if uid == "so100":
        robot_config = SO100FollowerConfig(
            port="/dev/tty.usbmodem59700726681",
            use_degrees=True,
            # for phone camera users you can use the commented out setting below
            # cameras={
            #     "base_camera": OpenCVCameraConfig(camera_index=1, fps=30, width=640, height=480)
            # }
            # for intel realsense camera users you need to modify the serial number or name for your own hardware
            cameras={
                "base_camera": OpenCVCameraConfig(
                index_or_path=0, fps=30, width=640, height=480)
            },
            id="follower",
        )
        real_robot = make_robot_from_config(robot_config)
        return real_robot