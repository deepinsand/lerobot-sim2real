from setuptools import find_packages, setup

setup(
    name="lerobot_sim2real",
    version="0.0.1",
    description="Sim2Real Manipulation with LeRobot",
    url="https://github.com/StoneT2000/lerobot-sim2real",
    packages=find_packages(include=["lerobot_sim2real*"]),
    python_requires=">=3.9",
    setup_requires=["setuptools>=62.3.0"],
    install_requires=[
        "mani-skill @ git+https://github.com/deepinsand/ManiSkill.git#egg=mani-skill",
        "tensorboard",
        "wandb"
    ]
)
