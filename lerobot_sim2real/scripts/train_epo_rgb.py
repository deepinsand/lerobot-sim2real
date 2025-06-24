"""Simple script to train a RGB EPO policy in simulation"""

from dataclasses import dataclass, field
import json
from typing import Optional
import tyro

from lerobot_sim2real.rl.epo_rgb import EPOArgs, train

@dataclass
class Args:
    env_id: str
    """The environment id to train on"""
    env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    epo: EPOArgs = field(default_factory=EPOArgs)
    """EPO training arguments"""

def main(args: Args):
    args.epo.env_id = args.env_id
    if args.env_kwargs_json_path is not None:
        with open(args.env_kwargs_json_path, "r") as f:
            env_kwargs = json.load(f)
        args.epo.env_kwargs = env_kwargs
    else:
        print("No env kwargs json path provided, using default env kwargs with default settings")
    train(args=args.epo)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)